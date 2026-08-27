"""
Unit tests for condition_b.py (PROTOCOL.md Sec. 5.3/5.4 Condition B).
Deliberately cheap and self-contained: builds a tiny fake cache with
Baseline 1's real (bandpass, zero-training) output on a 4-row
native-additive subset, the same way sdr_sweep.py's build_sdr_sweep()
would, rather than depending on the full S6-02 generation run.

The property that matters most: Condition B must evaluate each row's
separated audio through the *same* fold-k classifier weights Condition A
would use for that row's isolated ground truth -- never a classifier that
saw this row's own recording during training.
"""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from baseline.baseline1 import bandpass_separate
from condition_b import (
    ISOLATED_LABEL,
    NO_SEPARATION_LABEL,
    build_condition_b_fold_basis,
    evaluate_condition_b,
    evaluate_no_separation,
    paired_delta_vs_isolated,
    summarize_by_baseline,
    summarize_by_fold,
)
from heart_classifier import extract_features, train_fold_classifiers
from load_dataset import load_audio, load_mix, verify_additive_triplets
from sdr_sweep import estimate_path
from split import assign_folds

BASELINE_LABEL = "Baseline 1 (bandpass)"


@pytest.fixture(scope="module")
def tiny_mix_df():
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    return valid.iloc[:4].reset_index(drop=True)


@pytest.fixture(scope="module")
def tiny_cache(tiny_mix_df, tmp_path_factory):
    cache_root = tmp_path_factory.mktemp("condition_b_cache")
    for _, row in tiny_mix_df.iterrows():
        mixed, sr = load_audio(row["mixed_audio_path"], sr=None)
        heart_est, lung_est = bandpass_separate(mixed, sr)
        heart_path = estimate_path(cache_root, BASELINE_LABEL, row["Mixed Sound ID"], "heart")
        lung_path = estimate_path(cache_root, BASELINE_LABEL, row["Mixed Sound ID"], "lung")
        heart_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(heart_path, heart_est, sr)
        sf.write(lung_path, lung_est, sr)
    return cache_root


@pytest.fixture(scope="module")
def tiny_predictions(tiny_mix_df, tiny_cache):
    return evaluate_condition_b(
        n_folds=2, seed=0, cache_root=tiny_cache, baseline_labels=[BASELINE_LABEL], mix_df=tiny_mix_df
    )


class TestBuildConditionBFoldBasis:
    def test_fold_basis_matches_sdr_sweep_convention(self, tiny_mix_df):
        """Must be computed the exact same way sdr_sweep.build_sdr_sweep()
        computes it -- same function, same substrate, same seed -- or
        Condition B's weight-sharing guarantee silently breaks."""
        expected = assign_folds(mix_df=tiny_mix_df, n_folds=2, seed=0)
        mix_df, _hs_df = build_condition_b_fold_basis(n_folds=2, seed=0, mix_df=tiny_mix_df)
        pd.testing.assert_series_equal(
            mix_df.set_index("Mixed Sound ID")["fold"], expected.set_index("Mixed Sound ID")["fold"]
        )

    def test_classifier_never_trains_on_a_recording_in_its_own_held_out_fold(self, tiny_mix_df):
        mix_df, hs_df = build_condition_b_fold_basis(n_folds=2, seed=0, mix_df=tiny_mix_df)
        for k in sorted(mix_df["fold"].unique()):
            held_out_heart_hashes = set(mix_df.loc[mix_df["fold"] == k, "heart_hash"])
            train_hashes = set(hs_df.loc[hs_df["fold"] != k, "heart_hash"])
            assert held_out_heart_hashes.isdisjoint(train_hashes)


class TestEvaluateConditionB:
    def test_schema_and_row_count(self, tiny_predictions, tiny_mix_df):
        # One "Isolated" prediction + one per tested baseline (1 here) per row.
        assert len(tiny_predictions) == len(tiny_mix_df) * 2
        assert set(tiny_predictions["baseline"]) == {ISOLATED_LABEL, BASELINE_LABEL}

    def test_isolated_rows_have_no_sdr(self, tiny_predictions):
        isolated = tiny_predictions[tiny_predictions["baseline"] == ISOLATED_LABEL]
        assert isolated["achieved_sdr"].isna().all()

    def test_separated_rows_have_a_finite_sdr(self, tiny_predictions):
        separated = tiny_predictions[tiny_predictions["baseline"] == BASELINE_LABEL]
        assert np.isfinite(separated["achieved_sdr"]).all()

    def test_same_classifier_used_for_isolated_and_separated_in_same_fold(self, tiny_mix_df, tiny_cache):
        """The load-bearing weight-sharing property: for a given row, the
        isolated prediction must come from literally the same fitted
        classifier object Condition B used for that row's fold."""
        mix_df, hs_df = build_condition_b_fold_basis(n_folds=2, seed=0, mix_df=tiny_mix_df)
        classifiers = train_fold_classifiers(hs_df, n_folds=2)
        predictions_df = evaluate_condition_b(
            n_folds=2, seed=0, cache_root=tiny_cache, baseline_labels=[BASELINE_LABEL], mix_df=tiny_mix_df
        )

        for _, row in mix_df.iterrows():
            expected_clf = classifiers[row["fold"]]
            heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
            expected_pred = expected_clf.predict(extract_features(heart_ref, sr).reshape(1, -1))[0]
            actual_pred = predictions_df.loc[
                (predictions_df["baseline"] == ISOLATED_LABEL) & (predictions_df["mixed_id"] == row["Mixed Sound ID"]),
                "pred",
            ].iloc[0]
            assert actual_pred == expected_pred


class TestSummaries:
    def test_summarize_by_baseline_includes_isolated_and_tested_baseline(self, tiny_predictions):
        summary = summarize_by_baseline(tiny_predictions)
        assert summary.loc[ISOLATED_LABEL, "n"] > 0
        assert summary.loc[BASELINE_LABEL, "n"] > 0
        assert summary["accuracy"].dropna().between(0, 1).all()

    def test_summarize_by_fold_and_paired_delta_shapes(self, tiny_predictions):
        fold_summary = summarize_by_fold(tiny_predictions)
        assert set(fold_summary["baseline"].unique()) == {ISOLATED_LABEL, BASELINE_LABEL}

        paired = paired_delta_vs_isolated(fold_summary)
        assert BASELINE_LABEL in paired.index
        assert paired.loc[BASELINE_LABEL, "n_folds"] <= 2


class _FakeBackend:
    """Minimal backend stub (S7-06's `backend` contract): always predicts a
    fixed label regardless of audio content, and 'trains' by just
    recording the fold index -- exists purely to prove evaluate_condition_b/
    evaluate_no_separation actually call the *passed* backend's
    train_fold_classifiers/predict_one, not heart_classifier's, rather than
    silently ignoring the parameter."""

    FIXED_PREDICTION = "Rhythm Disorder"  # deliberately not what heart_classifier would predict

    @staticmethod
    def train_fold_classifiers(hs_df, n_folds):
        return {k: f"fake-classifier-fold-{k}" for k in range(n_folds)}

    @staticmethod
    def predict_one(clf, y, sr):
        assert clf.startswith("fake-classifier-fold-")  # proves *this* backend's classifier was used
        return _FakeBackend.FIXED_PREDICTION


class TestBackendParameter:
    def test_evaluate_condition_b_uses_the_passed_backend(self, tiny_mix_df, tiny_cache):
        predictions_df = evaluate_condition_b(
            n_folds=2, seed=0, cache_root=tiny_cache, baseline_labels=[BASELINE_LABEL],
            mix_df=tiny_mix_df, backend=_FakeBackend,
        )
        assert (predictions_df["pred"] == _FakeBackend.FIXED_PREDICTION).all()

    def test_evaluate_no_separation_uses_the_passed_backend(self, tiny_mix_df):
        predictions_df = evaluate_no_separation(n_folds=2, seed=0, mix_df=tiny_mix_df, backend=_FakeBackend)
        assert (predictions_df["pred"] == _FakeBackend.FIXED_PREDICTION).all()
        assert (predictions_df["baseline"] == NO_SEPARATION_LABEL).all()

    def test_default_backend_is_heart_classifier(self, tiny_mix_df, tiny_cache):
        """Not passing `backend` at all must still work exactly as before
        (Architecture 1) -- a real, non-fake run, unlike the two tests above."""
        predictions_df = evaluate_condition_b(
            n_folds=2, seed=0, cache_root=tiny_cache, baseline_labels=[BASELINE_LABEL], mix_df=tiny_mix_df,
        )
        assert predictions_df["pred"].isin(["Normal", "Murmur", "Extra Sound", "Rhythm Disorder"]).all()
