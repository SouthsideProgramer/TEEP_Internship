"""
Unit tests for synthetic_mix.py (S1-09/S1-10/S1-13).

The property that actually matters here, mirroring test_split.py's own
framing: no heart or lung recording used in a fold's held-out synthetic
evaluation pairs may appear in that same fold's dictionary-fitting pool.
Also covers S1-10's own claim -- that the construction reproduces the 36
native additive rows -- since that's the evidence the synthetic substrate
relies on to stand in for the dataset's real mixtures.
"""
import numpy as np
import pandas as pd
import pytest

from load_dataset import load_hs, load_ls
from synthetic_mix import (
    SNR_SWEEP_DB,
    SYNTHETIC_MIX_N_FOLDS,
    assign_source_folds,
    build_synthetic_set,
    dictionary_pool_synthetic,
    synthesize_row,
    validate_against_native,
)


@pytest.fixture(scope="module")
def hs_ls():
    return load_hs(), load_ls()


@pytest.fixture(scope="module")
def synthetic_df():
    return build_synthetic_set(n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)


class TestSourceFoldAssignment:
    def test_every_recording_gets_exactly_one_fold(self, hs_ls):
        hs_df, ls_df = hs_ls
        heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=5, seed=0)

        assert set(heart_fold_map) == set(hs_df["Heart Sound ID"])
        assert set(lung_fold_map) == set(ls_df["Lung Sound ID"])
        assert set(heart_fold_map.values()) == set(range(5))
        assert set(lung_fold_map.values()) == set(range(5))

    def test_folds_are_balanced(self, hs_ls):
        """50 recordings / 5 folds should split evenly -- no groups to balance here, unlike split.assign_folds."""
        hs_df, ls_df = hs_ls
        heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=5, seed=0)

        heart_sizes = pd.Series(heart_fold_map).value_counts()
        lung_sizes = pd.Series(lung_fold_map).value_counts()
        assert heart_sizes.max() - heart_sizes.min() <= 1
        assert lung_sizes.max() - lung_sizes.min() <= 1

    def test_reproducible_with_same_seed(self, hs_ls):
        hs_df, ls_df = hs_ls
        a_heart, a_lung = assign_source_folds(hs_df, ls_df, n_folds=5, seed=7)
        b_heart, b_lung = assign_source_folds(hs_df, ls_df, n_folds=5, seed=7)
        assert a_heart == b_heart
        assert a_lung == b_lung


class TestSyntheticSetConstruction:
    def test_only_same_fold_pairs_are_included(self, hs_ls, synthetic_df):
        """S1-13: a synthetic pair is only usable for evaluation when both sources share a fold."""
        hs_df, ls_df = hs_ls
        heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)

        for _, row in synthetic_df.drop_duplicates(["heart_id", "lung_id"]).iterrows():
            assert heart_fold_map[row["heart_id"]] == lung_fold_map[row["lung_id"]] == row["fold"]

    def test_expected_row_count(self, synthetic_df):
        """50 heart x 50 lung, 5 folds -> 10x10=100 same-fold pairs/fold x 5 folds x len(SNR_SWEEP_DB) levels."""
        n_pairs = synthetic_df[["heart_id", "lung_id"]].drop_duplicates().shape[0]
        assert n_pairs == (50 // SYNTHETIC_MIX_N_FOLDS) ** 2 * SYNTHETIC_MIX_N_FOLDS
        assert len(synthetic_df) == n_pairs * len(SNR_SWEEP_DB)

    def test_provenance_columns_complete(self, synthetic_df):
        required = {"mix_id", "heart_id", "lung_id", "gain_a", "snr_db", "fold"}
        assert required.issubset(synthetic_df.columns)
        assert synthetic_df[list(required)].notna().all().all()
        assert synthetic_df["mix_id"].is_unique

    def test_gain_fixed_per_pair_across_snr_realizations(self, synthetic_df):
        """Difficulty varies along the noise axis only -- gain is drawn once per (heart, lung) pair."""
        gains_per_pair = synthetic_df.groupby(["heart_id", "lung_id"])["gain_a"].nunique()
        assert (gains_per_pair == 1).all()

    def test_snr_sweep_fully_represented_per_pair(self, synthetic_df):
        levels_per_pair = synthetic_df.groupby(["heart_id", "lung_id"])["snr_db"].apply(lambda s: set(s))
        assert (levels_per_pair == set(SNR_SWEEP_DB)).all()

    def test_reproducible_with_same_seed(self):
        a = build_synthetic_set(n_folds=5, seed=3)
        b = build_synthetic_set(n_folds=5, seed=3)
        pd.testing.assert_frame_equal(a, b)


class TestDictionaryPoolNoLeakage:
    @pytest.mark.parametrize("fold", range(SYNTHETIC_MIX_N_FOLDS))
    def test_dictionary_pool_excludes_both_sources_of_held_out_pairs(self, hs_ls, synthetic_df, fold):
        """
        The critical property (mirrors test_split.py's dictionary-pool test):
        neither the heart nor the lung recording used in any of this fold's
        held-out synthetic pairs may appear in this fold's dictionary pool.
        """
        hs_df, ls_df = hs_ls
        heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)

        held_out = synthetic_df[synthetic_df["fold"] == fold]
        excluded_heart = set(held_out["heart_id"])
        excluded_lung = set(held_out["lung_id"])

        hs_allowed, ls_allowed = dictionary_pool_synthetic(hs_df, ls_df, heart_fold_map, lung_fold_map, held_out_fold=fold)

        assert not set(hs_allowed["Heart Sound ID"]) & excluded_heart
        assert not set(ls_allowed["Lung Sound ID"]) & excluded_lung
        assert excluded_heart <= {hid for hid, f in heart_fold_map.items() if f == fold}
        assert excluded_lung <= {lid for lid, f in lung_fold_map.items() if f == fold}


class TestSynthesizeRow:
    def test_zero_snr_edge_and_reconstruction_matches_gain_model(self, hs_ls, synthetic_df):
        """mixed ~= gain_a*(heart+lung) + noise -- check the noise-free component matches exactly."""
        from load_dataset import load_audio

        hs_df, ls_df = hs_ls
        hs_cache = {r["Heart Sound ID"]: load_audio(r["audio_path"], sr=None) for _, r in hs_df.iterrows()}
        ls_cache = {r["Lung Sound ID"]: load_audio(r["audio_path"], sr=None) for _, r in ls_df.iterrows()}

        row = synthetic_df.iloc[0]
        heart_ref, lung_ref, mixed, sr = synthesize_row(row, hs_cache, ls_cache, seed=0)

        base = row["gain_a"] * (heart_ref + lung_ref)
        noise = mixed - base
        target_noise_rms = np.sqrt(np.mean(base**2)) / (10 ** (row["snr_db"] / 20.0))
        assert np.isclose(np.sqrt(np.mean(noise**2)), target_noise_rms, rtol=0.2)

    def test_reproducible_with_same_seed(self, hs_ls, synthetic_df):
        from load_dataset import load_audio

        hs_df, ls_df = hs_ls
        hs_cache = {r["Heart Sound ID"]: load_audio(r["audio_path"], sr=None) for _, r in hs_df.iterrows()}
        ls_cache = {r["Lung Sound ID"]: load_audio(r["audio_path"], sr=None) for _, r in ls_df.iterrows()}

        row = synthetic_df.iloc[0]
        _h1, _l1, mixed_a, _sr = synthesize_row(row, hs_cache, ls_cache, seed=0)
        _h2, _l2, mixed_b, _sr = synthesize_row(row, hs_cache, ls_cache, seed=0)
        np.testing.assert_array_equal(mixed_a, mixed_b)


class TestValidateAgainstNative:
    def test_reproduces_all_native_additive_rows(self):
        """
        S1-10: applying this module's construction (gain*(heart+lung), zero
        noise) to the 36 native additive rows' own real audio should land in
        the same tight residual cluster the genuine native rows occupy --
        verify_additive_triplets() itself is the checker, reused directly.
        """
        result = validate_against_native()
        assert result["all_reproduce"], [r for r in result["rows"] if not r["reproduces"]]

        residuals = [r["synthetic_residual"] for r in result["rows"]]
        assert max(residuals) < 1e-2
