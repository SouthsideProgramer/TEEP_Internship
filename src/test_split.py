"""
Unit tests for split.py's leakage-safe fold assignment, and eval_harness.py's
wiring of it into cross-validated evaluation.

The property that actually matters here: no recording that appears in a
fold's held-out mixtures may appear in that same fold's dictionary-fitting
pool. Everything else (balance, reproducibility, coverage) is secondary.
"""
import numpy as np
import pandas as pd
import pytest

from eval_harness import aggregate_across_folds, aggregate_by_fold, cross_validate
from load_dataset import load_hs, load_ls, load_mix
from split import _content_hash, assign_folds, dictionary_pool


@pytest.fixture(scope="module")
def mix_df():
    return assign_folds(n_folds=5, seed=0)


@pytest.fixture(scope="module")
def hs_ls():
    return load_hs(), load_ls()


class TestFoldAssignment:
    def test_every_row_gets_exactly_one_fold(self, mix_df):
        assert mix_df["fold"].notna().all()
        assert set(mix_df["fold"].unique()) == set(range(5))
        assert len(mix_df) == len(load_mix())

    def test_leak_groups_never_split_across_folds(self, mix_df):
        """The whole point of grouping: every row sharing a leak group must land in the same fold."""
        folds_per_group = mix_df.groupby("leak_group")["fold"].nunique()
        assert (folds_per_group == 1).all(), (
            f"leak groups spanning multiple folds: {folds_per_group[folds_per_group > 1].to_dict()}"
        )

    def test_rows_sharing_a_recording_share_a_fold(self, mix_df):
        """Direct check without going through leak_group: same heart_hash or lung_hash -> same fold."""
        for col in ("heart_hash", "lung_hash"):
            folds_per_hash = mix_df.groupby(col)["fold"].nunique()
            assert (folds_per_hash == 1).all()

    def test_folds_are_reasonably_balanced(self, mix_df):
        sizes = mix_df.groupby("fold").size()
        assert sizes.min() >= len(mix_df) // 5 - 10  # loose bound; largest leak group is 32/145 rows

    def test_reproducible_with_same_seed(self):
        a = assign_folds(n_folds=5, seed=42)
        b = assign_folds(n_folds=5, seed=42)
        pd.testing.assert_series_equal(a["fold"], b["fold"])


class TestDictionaryPoolNoLeakage:
    @pytest.mark.parametrize("fold", range(5))
    def test_dictionary_pool_excludes_all_held_out_recordings(self, mix_df, hs_ls, fold):
        """
        The critical property: hash every recording in the fold's dictionary
        pool and confirm none of them match a recording used anywhere in
        that fold's held-out evaluation mixtures.
        """
        hs_df, ls_df = hs_ls
        held_out = mix_df[mix_df["fold"] == fold]
        excluded_heart = set(held_out["heart_hash"])
        excluded_lung = set(held_out["lung_hash"])

        hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=fold)

        hs_allowed_hashes = hs_allowed["audio_path"].map(_content_hash)
        ls_allowed_hashes = ls_allowed["audio_path"].map(_content_hash)

        assert not set(hs_allowed_hashes) & excluded_heart
        assert not set(ls_allowed_hashes) & excluded_lung

    def test_dictionary_pool_only_removes_whats_necessary(self, mix_df, hs_ls):
        """Recordings not reused in the held-out fold's mixtures should stay in the pool."""
        hs_df, ls_df = hs_ls
        hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=0)
        # some rows are always excluded (fold 0 reuses some HS/LS recordings), but not all
        assert 0 < len(hs_allowed) < len(hs_df)
        assert 0 < len(ls_allowed) < len(ls_df)


class TestCrossValidationHarness:
    def test_cross_validate_covers_every_mix_row_exactly_once(self):
        def passthrough(hs_allowed, ls_allowed):
            return lambda mixed, sr: (mixed, mixed)

        results_df, fold_summary, cv_summary = cross_validate(passthrough, n_folds=5, seed=0)

        assert set(results_df["Mixed Sound ID"]) == set(load_mix()["Mixed Sound ID"])
        assert len(results_df) == len(load_mix()) * 2  # heart + lung per row
        # each (Mixed Sound ID, source) pair appears exactly once, in exactly one fold
        counts = results_df.groupby(["Mixed Sound ID", "source"])["fold"].nunique()
        assert (counts == 1).all()

    def test_dictionary_pool_actually_reaches_the_fit_function(self):
        """The harness must hand each fold its own restricted pool, not the full HS.csv/LS.csv every time."""
        seen_pool_sizes = []

        def spy(hs_allowed, ls_allowed):
            seen_pool_sizes.append((len(hs_allowed), len(ls_allowed)))
            return lambda mixed, sr: (mixed, mixed)

        cross_validate(spy, n_folds=5, seed=0)

        hs_total, ls_total = len(load_hs()), len(load_ls())
        assert len(seen_pool_sizes) == 5
        # every fold's pool must be smaller than the full HS/LS set (some overlap always exists)
        assert all(n_hs < hs_total and n_ls < ls_total for n_hs, n_ls in seen_pool_sizes)

    def test_aggregate_by_fold_matches_manual_groupby(self):
        def passthrough(hs_allowed, ls_allowed):
            return lambda mixed, sr: (mixed, mixed)

        results_df, fold_summary, _ = cross_validate(passthrough, n_folds=5, seed=0)
        expected = results_df.groupby(["fold", "source"])[["sdr", "sir", "sar"]].mean().reset_index()
        pd.testing.assert_frame_equal(
            fold_summary.sort_values(["fold", "source"]).reset_index(drop=True),
            expected.sort_values(["fold", "source"]).reset_index(drop=True),
        )

    def test_aggregate_across_folds_reports_mean_and_std_per_source(self):
        def passthrough(hs_allowed, ls_allowed):
            return lambda mixed, sr: (mixed, mixed)

        _, fold_summary, cv_summary = cross_validate(passthrough, n_folds=5, seed=0)

        assert list(cv_summary.index) == ["heart", "lung"]
        for metric in ("sdr", "sir", "sar"):
            assert (metric, "mean") in cv_summary.columns
            assert (metric, "std") in cv_summary.columns
            assert np.isfinite(cv_summary[(metric, "mean")]).all()
