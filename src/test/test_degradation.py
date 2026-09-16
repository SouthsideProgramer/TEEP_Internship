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
        assert sweep[0]["sdr"] > sweep[1]["sdr"]
        assert sweep[1]["sdr"] == pytest.approx(real_metrics["heart"]["sdr"])


class TestFindAlphaForTargetSdr:
    def test_finds_alpha_achieving_target_within_tolerance(self, real_audio):
        heart_ref, lung_ref, heart_est, lung_est = real_audio
        real_metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        achievable_lo = real_metrics["heart"]["sdr"]
        target = achievable_lo + 5.0
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


class TestGridFinder:
    """find_alphas_for_target_sdrs_grid (2026-09-13): exact SDR(alpha) and the
    pre-stated smallest-root rule."""

    def test_closed_form_matches_direct_evaluation(self, real_audio):
        import numpy as np
        from degradation import sdr_curve_closed_form
        from metrics import evaluate_heart_lung

        heart_ref, lung_ref, heart_est, lung_est = real_audio
        refs = np.stack([heart_ref, lung_ref])
        for alpha in (0.05, 0.3, 0.7, 1.0):
            closed = sdr_curve_closed_form(refs, heart_ref, heart_est, 0, np.array([alpha]))[0]
            d = (1 - alpha) * heart_ref + alpha * heart_est
            direct = evaluate_heart_lung(heart_ref, lung_ref, d, lung_est)["heart"]["sdr"]
            assert abs(closed - direct) < 1e-4, (alpha, closed, direct)

    def test_grid_finder_hits_targets_and_flags_unattainable(self, real_audio):
        from degradation import degrade_row, find_alphas_for_target_sdrs_grid

        heart_ref, lung_ref, heart_est, lung_est = real_audio
        real = degrade_row(heart_ref, lung_ref, heart_est, lung_est, 1.0, "heart")["heart"]["sdr"]
        targets = [real + 5.0, real + 15.0, real - 100.0, 1000.0]
        found = find_alphas_for_target_sdrs_grid(heart_ref, lung_ref, heart_est, lung_est, targets, source="heart")
        assert found["_curve"]["monotone_decreasing"] in (True, False)
        for t in targets[:2]:
            hit = found[float(t)]
            assert hit["attainable"] and 0.0 < hit["alpha"] < 1.0
            got = degrade_row(heart_ref, lung_ref, heart_est, lung_est, hit["alpha"], "heart")["heart"]["sdr"]
            assert abs(got - t) < 0.1, (t, got)
        assert not found[float(targets[2])]["attainable"]
        assert not found[1000.0]["attainable"]

    def test_smallest_root_rule_on_a_synthetic_dip(self):
        import numpy as np
        from degradation import grid_roots

        alphas = np.linspace(0, 1, 11)
        sdr = np.array([30, 20, 10, 0, -10, -5, 0, 5, 10, 12, 12], dtype=float)  # dips then recovers
        roots = grid_roots(alphas, sdr, 5.0)
        assert len(roots) == 2
        assert min(roots) < 0.3  # descending branch from the clean end
