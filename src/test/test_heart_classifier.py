"""
Unit tests for heart_classifier.py (PROTOCOL.md Sec. 5.3 Condition A) and
split.py's assign_hs_folds().

The properties that matter most: the class-group mapping covers every
HS.csv Heart Sound Type with the counts the class-grouping decision was
based on, and an HS.csv recording that leaks into a Mix.csv fold's held-out
mixtures never gets assigned a different fold for classifier training --
the same leak-safety property test_split.py checks for dictionary_pool(),
applied here to the classifier's own fold assignment.
"""
import numpy as np
import pandas as pd
import pytest

from heart_classifier import (
    CLASS_GROUPS,
    HEART_TYPE_TO_GROUP,
    add_class_group,
    assign_classifier_folds,
    confusion_counts,
    confusion_recall_pct,
    cross_validate_classifier,
    extract_features,
    top_confusions,
)
from load_dataset import load_audio, load_hs
from split import assign_folds


@pytest.fixture(scope="module")
def hs_df():
    return add_class_group(load_hs())


@pytest.fixture(scope="module")
def mix_df():
    return assign_folds(n_folds=5, seed=0)


@pytest.fixture(scope="module")
def classifier_hs_df():
    return assign_classifier_folds(n_folds=5, seed=0)


class TestClassGroupMapping:
    def test_every_heart_sound_type_is_mapped(self, hs_df):
        assert hs_df["class_group"].notna().all()
        assert set(hs_df["class_group"].unique()) == set(CLASS_GROUPS)

    def test_group_sizes_match_the_documented_decision(self, hs_df):
        counts = hs_df["class_group"].value_counts()
        assert counts["Normal"] == 9
        assert counts["Murmur"] == 24
        assert counts["Extra Sound"] == 7
        assert counts["Rhythm Disorder"] == 10

    def test_unmapped_type_raises(self):
        bad_df = pd.DataFrame({"Heart Sound Type": ["Not A Real Type"]})
        with pytest.raises(ValueError, match="Unmapped"):
            add_class_group(bad_df)

    def test_mapping_only_targets_the_four_groups(self):
        assert set(HEART_TYPE_TO_GROUP.values()) == set(CLASS_GROUPS)


class TestAssignClassifierFolds:
    def test_every_row_gets_exactly_one_fold(self, classifier_hs_df):
        assert classifier_hs_df["fold"].notna().all()
        assert set(classifier_hs_df["fold"].unique()) <= set(range(5))
        assert len(classifier_hs_df) == len(load_hs())

    def test_leaked_recordings_inherit_the_mix_fold(self, classifier_hs_df, mix_df):
        """A HS.csv recording byte-identical to a Mix.csv row's heart component
        must land in that leak group's fold -- the same boundary
        dictionary_pool() enforces for the separation baselines, applied here
        to the classifier's own fold assignment instead."""
        hash_to_fold = dict(zip(mix_df["heart_hash"], mix_df["fold"]))
        leaked = classifier_hs_df[classifier_hs_df["heart_hash"].isin(hash_to_fold)]
        assert len(leaked) > 0, "expected at least some HS.csv recordings to be reused in Mix.csv"
        for _, row in leaked.iterrows():
            assert row["fold"] == hash_to_fold[row["heart_hash"]]

    def test_reproducible_with_same_seed(self):
        a = assign_classifier_folds(n_folds=5, seed=7)
        b = assign_classifier_folds(n_folds=5, seed=7)
        pd.testing.assert_series_equal(a["fold"], b["fold"])

    def test_every_fold_has_training_examples_of_every_class_group(self, classifier_hs_df):
        """The whole point of grouping instead of using the raw 10 types: no
        fold's training set should be starved of a class group entirely."""
        for k in sorted(classifier_hs_df["fold"].unique()):
            train_groups = set(classifier_hs_df.loc[classifier_hs_df["fold"] != k, "class_group"])
            assert train_groups == set(CLASS_GROUPS), f"fold {k} training set missing: {set(CLASS_GROUPS) - train_groups}"


class TestFeatures:
    def test_feature_vector_shape(self):
        hs_df = load_hs()
        y, sr = load_audio(hs_df.loc[0, "audio_path"], sr=None)
        feats = extract_features(y, sr)
        assert feats.shape == (26,)
        assert np.all(np.isfinite(feats))


@pytest.fixture(scope="module")
def cv_result():
    return cross_validate_classifier(n_folds=5, seed=0)


class TestCrossValidateClassifier:
    def test_every_recording_predicted_exactly_once(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        assert len(predictions_df) == len(load_hs())
        assert predictions_df["Heart Sound ID"].is_unique

    def test_fold_summary_metrics_in_valid_range(self, cv_result):
        _predictions_df, fold_summary, _cv_summary = cv_result
        assert len(fold_summary) == 5
        assert fold_summary["accuracy"].between(0, 1).all()
        assert fold_summary["macro_f1"].between(0, 1).all()

    def test_cv_summary_has_ci_for_both_metrics(self, cv_result):
        _predictions_df, _fold_summary, cv_summary = cv_result
        assert set(cv_summary.index) == {"accuracy", "macro_f1"}
        assert (cv_summary["n_folds"] == 5).all()
        assert cv_summary["mean"].between(0, 1).all()


class TestConfusionMatrix:
    def test_counts_sum_to_class_group_sizes(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        counts = confusion_counts(predictions_df)
        assert list(counts.index) == CLASS_GROUPS
        assert list(counts.columns) == CLASS_GROUPS
        true_counts = predictions_df["true"].value_counts()
        pd.testing.assert_series_equal(
            counts.sum(axis=1).rename("true"), true_counts.reindex(CLASS_GROUPS).rename("true"), check_dtype=False
        )
        assert counts.to_numpy().sum() == len(predictions_df)

    def test_diagonal_matches_per_class_recall(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        counts = confusion_counts(predictions_df)
        for group in CLASS_GROUPS:
            true_mask = predictions_df["true"] == group
            n_correct = int((predictions_df.loc[true_mask, "pred"] == group).sum())
            assert counts.loc[group, group] == n_correct

    def test_recall_pct_rows_sum_to_100(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        counts = confusion_counts(predictions_df)
        pct = confusion_recall_pct(counts)
        np.testing.assert_allclose(pct.sum(axis=1).to_numpy(), 100.0)

    def test_top_confusions_excludes_the_diagonal(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        counts = confusion_counts(predictions_df)
        confusions = top_confusions(counts)
        for true_label, row in confusions.iterrows():
            assert row["most_confused_with"] != true_label

    def test_top_confusions_recall_matches_confusion_counts_diagonal(self, cv_result):
        predictions_df, _fold_summary, _cv_summary = cv_result
        counts = confusion_counts(predictions_df)
        confusions = top_confusions(counts)
        for group in CLASS_GROUPS:
            expected_recall = counts.loc[group, group] / counts.loc[group].sum()
            assert confusions.loc[group, "recall"] == pytest.approx(expected_recall)

    def test_a_small_synthetic_confusion_matrix_by_hand(self):
        predictions_df = pd.DataFrame({
            "true": ["Normal", "Normal", "Murmur", "Murmur", "Murmur"],
            "pred": ["Normal", "Murmur", "Murmur", "Murmur", "Rhythm Disorder"],
        })
        counts = confusion_counts(predictions_df)
        assert counts.loc["Normal", "Normal"] == 1
        assert counts.loc["Normal", "Murmur"] == 1
        assert counts.loc["Murmur", "Murmur"] == 2
        assert counts.loc["Murmur", "Rhythm Disorder"] == 1
        assert counts.loc["Extra Sound"].sum() == 0

        confusions = top_confusions(counts)
        assert confusions.loc["Normal", "most_confused_with"] == "Murmur"
        assert confusions.loc["Normal", "confusion_pct"] == pytest.approx(0.5)
