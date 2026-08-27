"""
Unit tests for degradation.py (S6-01, the controlled separation-degradation
scheme). The property that matters most: degrade_toward_ground_truth()'s
alpha sweep is actually monotonic in measured SDR on real data -- everything
else (endpoints, source isolation, the root-finder) is secondary but still
checked directly rather than assumed.
"""
import numpy as np
import pytest

from baseline.baseline1 import fit_bandpass_baseline
from degradation import (
    degrade_row,
    degrade_toward_ground_truth,
    find_alpha_for_target_sdr,
    find_alphas_for_target_sdrs,
    sdr_sweep,
)
from load_dataset import load_audio, load_mix, verify_additive_triplets
from metrics import evaluate_heart_lung


@pytest.fixture(scope="module")
def native_additive_row():
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    return mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].iloc[0]


@pytest.fixture(scope="module")
def real_audio(native_additive_row):
    heart_ref, sr = load_audio(native_additive_row["heart_audio_path"], sr=None)
    lung_ref, _ = load_audio(native_additive_row["lung_audio_path"], sr=None)
    mixed, _ = load_audio(native_additive_row["mixed_audio_path"], sr=None)
    heart_est, lung_est = fit_bandpass_baseline(None, None)(mixed, sr)
    return heart_ref, lung_ref, heart_est, lung_est


class TestDegradeTowardGroundTruth:
    def test_alpha_zero_is_ground_truth(self):
        g = np.array([1.0, 2.0, 3.0])
        s = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(degrade_toward_ground_truth(g, s, alpha=0.0), g)

    def test_alpha_one_is_estimate(self):
        g = np.array([1.0, 2.0, 3.0])
        s = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(degrade_toward_ground_truth(g, s, alpha=1.0), s)

    def test_alpha_half_is_the_midpoint(self):
        g = np.zeros(4)
        s = np.ones(4) * 2
        np.testing.assert_allclose(degrade_toward_ground_truth(g, s, alpha=0.5), np.ones(4))

    def test_truncates_to_shorter_length(self):
        g = np.ones(10)
        s = np.zeros(7)
        result = degrade_toward_ground_truth(g, s, alpha=1.0)
        assert len(result) == 7


class TestDegradeRow:
    def test_degrading_heart_leaves_lung_estimate_unchanged(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        degraded = degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha=0.5, source="heart")
        undisturbed = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        assert degraded["lung"]["sdr"] == pytest.approx(undisturbed["lung"]["sdr"])

    def test_invalid_source_raises(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        with pytest.raises(ValueError, match="source must be"):
            degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha=0.5, source="not_a_source")


class TestSdrSweepIsMonotonic:
    """The load-bearing property: real separation error injected at increasing
    alpha must never *increase* measured SDR, on real audio."""

    def test_monotonic_non_increasing_on_real_data(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        sweep = sdr_sweep(heart_ref, lung_ref, heart_est, lung_est, alphas=np.linspace(0, 1, 11), source="heart")
        sdrs = np.array([row["sdr"] for row in sweep])
        assert np.all(np.diff(sdrs) <= 1e-6)

    def test_alpha_zero_is_near_clean_and_alpha_one_matches_measured_sdr(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        sweep = sdr_sweep(heart_ref, lung_ref, heart_est, lung_est, alphas=(0.0, 1.0), source="heart")
        real_metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        assert sweep[0]["sdr"] > sweep[1]["sdr"]  # clean is far better than the real separated output
        assert sweep[1]["sdr"] == pytest.approx(real_metrics["heart"]["sdr"])


class TestFindAlphaForTargetSdr:
    def test_finds_alpha_achieving_target_within_tolerance(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        real_metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        achievable_lo = real_metrics["heart"]["sdr"]
        target = achievable_lo + 5.0  # comfortably inside the achievable range
        alpha = find_alpha_for_target_sdr(heart_ref, lung_ref, heart_est, lung_est, target_sdr_db=target, source="heart")
        achieved = degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha, "heart")["heart"]["sdr"]
        assert achieved == pytest.approx(target, abs=0.5)
        assert 0.0 <= alpha <= 1.0

    def test_clamps_to_alpha_one_when_target_below_achievable_range(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        real_metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        very_low_target = real_metrics["heart"]["sdr"] - 50.0
        alpha = find_alpha_for_target_sdr(heart_ref, lung_ref, heart_est, lung_est, target_sdr_db=very_low_target, source="heart")
        assert alpha == 1.0

    def test_clamps_to_alpha_zero_when_target_above_achievable_range(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        alpha = find_alpha_for_target_sdr(heart_ref, lung_ref, heart_est, lung_est, target_sdr_db=1000.0, source="heart")
        assert alpha == 0.0


class TestFindAlphasForTargetSdrsBatched:
    """The batched version (sdr_sweep.py's fast path) must agree with the
    single-target function it replaces -- it's a performance optimization,
    not a different definition."""

    def test_agrees_with_single_target_search(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        real_metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        floor = real_metrics["heart"]["sdr"]
        targets = [floor + 3.0, floor + 8.0, floor + 15.0]

        batched = find_alphas_for_target_sdrs(heart_ref, lung_ref, heart_est, lung_est, targets, source="heart")
        for target in targets:
            individual = find_alpha_for_target_sdr(heart_ref, lung_ref, heart_est, lung_est, target_sdr_db=target, source="heart")
            achieved_batched = degrade_row(heart_ref, lung_ref, heart_est, lung_est, batched[target], "heart")["heart"]["sdr"]
            achieved_individual = degrade_row(heart_ref, lung_ref, heart_est, lung_est, individual, "heart")["heart"]["sdr"]
            # Compare achieved SDR, not the raw alphas -- both methods can land
            # on slightly different alphas that hit the same target equally well.
            assert achieved_batched == pytest.approx(achieved_individual, abs=0.3)

    def test_clamps_consistently_with_single_target_search(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        batched = find_alphas_for_target_sdrs(heart_ref, lung_ref, heart_est, lung_est, [1000.0, -1000.0], source="heart")
        assert batched[1000.0] == 0.0
        assert batched[-1000.0] == 1.0

    def test_returns_one_alpha_per_target(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        targets = [20.0, 10.0, 5.0, 0.0]
        result = find_alphas_for_target_sdrs(heart_ref, lung_ref, heart_est, lung_est, targets, source="heart")
        assert set(result.keys()) == set(targets)
        assert all(0.0 <= a <= 1.0 for a in result.values())
