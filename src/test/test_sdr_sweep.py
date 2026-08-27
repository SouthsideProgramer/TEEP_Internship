"""
Unit tests for sdr_sweep.py (S6-02, the generated controlled SDR sweep
dataset). Deliberately cheap: uses a 4-row subset of the native additive
rows and only Baseline 1 (bandpass, zero-training) rather than the full
6-baseline x 36-row sweep __main__ generates -- the properties checked
here (schema, the alpha/clamping relationship, and that
synthesize_sweep_row() faithfully reconstructs what build_sdr_sweep()
recorded) don't depend on which baseline or how many rows.
"""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from baseline.baseline1 import fit_bandpass_baseline
from load_dataset import load_audio, load_mix, verify_additive_triplets
from metrics import SOURCE_LABELS, evaluate_heart_lung
from sdr_sweep import TARGET_SDR_GRID_DB, build_provenance_from_cache, build_sdr_sweep, synthesize_sweep_row
from split import assign_folds


@pytest.fixture(scope="module")
def tiny_mix_df():
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    return valid.iloc[:4].reset_index(drop=True)


@pytest.fixture(scope="module")
def tiny_sweep(tiny_mix_df, tmp_path_factory):
    cache_root = tmp_path_factory.mktemp("sdr_sweep_cache")
    sweep_df = build_sdr_sweep(
        n_folds=2,
        seed=0,
        mix_df=tiny_mix_df,
        baseline_specs=[("Baseline 1 (bandpass)", fit_bandpass_baseline)],
        cache_root=cache_root,
    )
    return sweep_df, cache_root


class TestBuildSdrSweep:
    def test_schema_and_row_count(self, tiny_sweep, tiny_mix_df):
        sweep_df, _cache_root = tiny_sweep
        expected_n = len(tiny_mix_df) * len(SOURCE_LABELS) * len(TARGET_SDR_GRID_DB)
        assert len(sweep_df) == expected_n
        assert set(sweep_df["mixed_id"]) == set(tiny_mix_df["Mixed Sound ID"])
        assert set(sweep_df["source"]) == set(SOURCE_LABELS)
        assert set(sweep_df["target_sdr"]) == set(TARGET_SDR_GRID_DB)

    def test_alpha_in_valid_range(self, tiny_sweep):
        sweep_df, _ = tiny_sweep
        assert sweep_df["alpha"].between(0, 1).all()

    def test_clamped_flag_matches_alpha_at_one(self, tiny_sweep):
        sweep_df, _ = tiny_sweep
        assert ((sweep_df["alpha"] >= 1.0 - 1e-9) == sweep_df["clamped_to_baseline_floor"]).all()

    def test_cached_estimate_wavs_exist(self, tiny_sweep, tiny_mix_df):
        _sweep_df, cache_root = tiny_sweep
        for mixed_id in tiny_mix_df["Mixed Sound ID"]:
            for source in SOURCE_LABELS:
                assert (cache_root / "Baseline1" / f"{mixed_id}_{source}.wav").is_file()

    def test_lower_targets_are_closer_to_or_at_the_baseline_floor(self, tiny_sweep):
        """A lower target_sdr should never need a *smaller* alpha than a higher
        target for the same row -- monotonicity of the underlying scheme,
        surfaced through the generated table rather than degradation.py's own
        unit tests."""
        sweep_df, _ = tiny_sweep
        for (_baseline, mixed_id, source), group in sweep_df.groupby(["baseline", "mixed_id", "source"]):
            ordered = group.sort_values("target_sdr", ascending=False)
            assert ordered["alpha"].is_monotonic_increasing


class TestSynthesizeSweepRow:
    def test_reconstruction_matches_recorded_sdr(self, tiny_sweep, tiny_mix_df):
        sweep_df, cache_root = tiny_sweep
        row = sweep_df.iloc[0]
        degraded, _sr = synthesize_sweep_row(row, mix_df=tiny_mix_df, cache_root=cache_root)

        mix_row = tiny_mix_df[tiny_mix_df["Mixed Sound ID"] == row["mixed_id"]].iloc[0]
        heart_ref, _ = load_audio(mix_row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(mix_row["lung_audio_path"], sr=None)

        heart_arg = degraded if row["source"] == "heart" else heart_ref
        lung_arg = degraded if row["source"] == "lung" else lung_ref
        remeasured = evaluate_heart_lung(heart_ref, lung_ref, heart_arg, lung_arg)[row["source"]]["sdr"]
        # Loose tolerance: the cached estimate round-trips through 16-bit PCM
        # (soundfile's WAV default), same quantization every other cached
        # .wav in this project already tolerates (e.g. verify_additive_
        # triplets' own 1e-3 relative-residual threshold).
        assert remeasured == pytest.approx(row["achieved_sdr"], abs=0.1)

    def test_alpha_zero_reconstructs_ground_truth_exactly(self, tiny_mix_df, tmp_path):
        heart_ref, sr = load_audio(tiny_mix_df.iloc[0]["heart_audio_path"], sr=None)
        (tmp_path / "Baseline1").mkdir()
        mixed_id = tiny_mix_df.iloc[0]["Mixed Sound ID"]
        # The cached "estimate" is arbitrary here -- alpha=0 should ignore it entirely.
        sf.write(tmp_path / "Baseline1" / f"{mixed_id}_heart.wav", heart_ref * 0.1, sr)

        row = pd.Series({"mixed_id": mixed_id, "source": "heart", "baseline": "Baseline 1 (bandpass)", "alpha": 0.0})
        degraded, _sr = synthesize_sweep_row(row, mix_df=tiny_mix_df, cache_root=tmp_path)
        np.testing.assert_allclose(degraded, heart_ref, atol=1e-3)


class TestBuildProvenanceFromCache:
    def test_matches_build_sdr_sweep_exactly_on_the_same_cache(self, tiny_mix_df, tmp_path):
        """The whole point of this function: reading back an already-cached
        estimate and recomputing the target grid must reproduce
        build_sdr_sweep()'s own provenance rows exactly, not approximately
        -- it's the same computation, just skipping the (here, already-done)
        separation step."""
        cache_root = tmp_path
        original = build_sdr_sweep(
            n_folds=2, seed=0, mix_df=tiny_mix_df,
            baseline_specs=[("Baseline 1 (bandpass)", fit_bandpass_baseline)],
            cache_root=cache_root,
        )

        mix_df_with_folds = assign_folds(mix_df=tiny_mix_df, n_folds=2, seed=0)
        rebuilt = build_provenance_from_cache(
            mix_df_with_folds, ["Baseline 1 (bandpass)"], cache_root=cache_root, n_folds=2, seed=0
        )

        key_cols = ["baseline", "mixed_id", "source", "target_sdr"]
        original_sorted = original.sort_values(key_cols).reset_index(drop=True)
        rebuilt_sorted = rebuilt.sort_values(key_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(original_sorted, rebuilt_sorted, check_like=True)

    def test_assigns_folds_when_missing_from_mix_df(self, tiny_mix_df, tmp_path):
        cache_root = tmp_path
        build_sdr_sweep(
            n_folds=2, seed=0, mix_df=tiny_mix_df,
            baseline_specs=[("Baseline 1 (bandpass)", fit_bandpass_baseline)],
            cache_root=cache_root,
        )
        # Pass mix_df *without* a 'fold' column -- should compute it internally.
        rebuilt = build_provenance_from_cache(
            tiny_mix_df, ["Baseline 1 (bandpass)"], cache_root=cache_root, n_folds=2, seed=0
        )
        assert set(rebuilt["mixed_id"]) == set(tiny_mix_df["Mixed Sound ID"])
