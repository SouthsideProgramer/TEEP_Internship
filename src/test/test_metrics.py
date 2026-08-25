"""
Unit tests for metrics.py.

Two kinds of checks:
  1. Cross-checks against mir_eval: metrics.py is a thin wrapper around
     mir_eval.separation.bss_eval_sources, so its outputs must agree with
     calling mir_eval directly on the same arrays -- both on synthetic
     signals and on real dataset audio.
  2. Synthetic mixtures with known ground truth: signals are built so the
     "correct" SDR/SIR/SAR are computable in closed form (additive
     independent noise -> known SNR; additive cross-source leakage -> known
     SIR), and the metrics must land within a small tolerance of that
     theoretical value.

Run with:
    python -m pytest test_metrics.py -v
"""
import warnings

import numpy as np
import pytest
from mir_eval.separation import bss_eval_sources

from load_dataset import load_audio, load_mix
from metrics import bss_eval, evaluate_dataset, evaluate_heart_lung, summarize_by_class

N = 32000  # synthetic signal length; >> the 512-tap filter bss_eval fits, so stats converge tightly
SEED = 0


def _white_sources(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal(n), rng


def _scale_to_power(x, target_power):
    return x * np.sqrt(target_power / np.mean(x**2))


def _raw_mir_eval(reference_sources, estimated_sources, compute_permutation=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sdr, sir, sar, perm = bss_eval_sources(
            np.atleast_2d(reference_sources), np.atleast_2d(estimated_sources), compute_permutation
        )
    return sdr, sir, sar, perm


# ---------------------------------------------------------------------------
# 1. Cross-checks against mir_eval
# ---------------------------------------------------------------------------

class TestCrossCheckAgainstMirEval:
    def test_bss_eval_matches_raw_mir_eval_call(self):
        """bss_eval() must return exactly what a direct mir_eval call returns."""
        s1, s2, rng = _white_sources()
        est1 = s1 + 0.2 * rng.standard_normal(N) + 0.1 * s2
        est2 = s2 + 0.2 * rng.standard_normal(N) + 0.1 * s1
        reference = np.stack([s1, s2])
        estimated = np.stack([est1, est2])

        wrapped = bss_eval(reference, estimated)
        raw_sdr, raw_sir, raw_sar, raw_perm = _raw_mir_eval(reference, estimated)

        np.testing.assert_allclose(wrapped["sdr"], raw_sdr)
        np.testing.assert_allclose(wrapped["sir"], raw_sir)
        np.testing.assert_allclose(wrapped["sar"], raw_sar)
        np.testing.assert_array_equal(wrapped["perm"], raw_perm)

    def test_evaluate_heart_lung_matches_raw_mir_eval_call(self):
        """evaluate_heart_lung()'s per-source dict must match a manual stack + raw mir_eval call."""
        s1, s2, rng = _white_sources()
        heart_est = s1 + 0.15 * rng.standard_normal(N)
        lung_est = s2 + 0.15 * rng.standard_normal(N)

        result = evaluate_heart_lung(s1, s2, heart_est, lung_est)
        raw_sdr, raw_sir, raw_sar, raw_perm = _raw_mir_eval(
            np.stack([s1, s2]), np.stack([heart_est, lung_est])
        )

        # with independent per-source noise (not swapped), mir_eval should find the identity permutation
        assert list(raw_perm) == [0, 1]
        assert result["heart"]["sdr"] == pytest.approx(raw_sdr[0])
        assert result["heart"]["sir"] == pytest.approx(raw_sir[0])
        assert result["heart"]["sar"] == pytest.approx(raw_sar[0])
        assert result["lung"]["sdr"] == pytest.approx(raw_sdr[1])
        assert result["lung"]["sir"] == pytest.approx(raw_sir[1])
        assert result["lung"]["sar"] == pytest.approx(raw_sar[1])

    def test_evaluate_mix_row_matches_raw_mir_eval_on_real_dataset_audio(self):
        """Same cross-check, but against real HLS-CMDS audio instead of synthetic noise."""
        row = load_mix().iloc[0]
        heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
        rng = np.random.default_rng(1)
        heart_est = heart_ref + 0.01 * rng.standard_normal(len(heart_ref))
        lung_est = lung_ref + 0.01 * rng.standard_normal(len(lung_ref))

        result = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        raw_sdr, raw_sir, raw_sar, raw_perm = _raw_mir_eval(
            np.stack([heart_ref, lung_ref]), np.stack([heart_est, lung_est])
        )

        assert list(raw_perm) == [0, 1]
        assert result["heart"]["sdr"] == pytest.approx(raw_sdr[0])
        assert result["lung"]["sdr"] == pytest.approx(raw_sdr[1])


# ---------------------------------------------------------------------------
# 2. Synthetic mixtures with known ground truth
# ---------------------------------------------------------------------------

class TestSyntheticGroundTruth:
    def test_identity_separation_is_near_perfect(self):
        """est == ref exactly -> SDR/SIR/SAR should all be very high (bounded only by float precision)."""
        s1, s2, _ = _white_sources()
        result = evaluate_heart_lung(s1, s2, s1, s2)
        for source in ("heart", "lung"):
            assert result[source]["sdr"] > 100
            assert result[source]["sir"] > 100
            assert result[source]["sar"] > 100

    @pytest.mark.parametrize("snr_db", [20.0, 6.0])
    def test_artifact_only_approaches_theoretical_snr(self, snr_db):
        """
        est = ref + independent noise at a known power ratio, no cross-source leakage.
        With no interference term, SDR and SAR should both converge to the injected SNR,
        and SIR (unaffected by the independent noise) should sit well above it.
        """
        s1, s2, rng = _white_sources()
        n1 = _scale_to_power(rng.standard_normal(N), np.mean(s1**2) / 10 ** (snr_db / 10))
        n2 = _scale_to_power(rng.standard_normal(N), np.mean(s2**2) / 10 ** (snr_db / 10))

        result = evaluate_heart_lung(s1, s2, s1 + n1, s2 + n2)

        for source in ("heart", "lung"):
            assert result[source]["sdr"] == pytest.approx(snr_db, abs=1.0)
            assert result[source]["sar"] == pytest.approx(snr_db, abs=1.0)
            assert result[source]["sir"] > snr_db + 10

    @pytest.mark.parametrize("alpha", [0.1, 0.3])
    def test_interference_only_approaches_theoretical_sir(self, alpha):
        """
        est_j = s_j + alpha * s_k (leakage of the other source, no independent noise).
        With no artifact term, SDR and SIR should both converge to the theoretical SIR
        10*log10(power(s_j) / power(alpha*s_k)), and SAR should be very high.
        """
        s1, s2, _ = _white_sources()
        theoretical_sir_db = -20 * np.log10(alpha)  # power(s1) ~= power(s2) for white noise of equal length

        result = evaluate_heart_lung(s1, s2, s1 + alpha * s2, s2 + alpha * s1)

        for source in ("heart", "lung"):
            assert result[source]["sir"] == pytest.approx(theoretical_sir_db, abs=1.0)
            assert result[source]["sdr"] == pytest.approx(theoretical_sir_db, abs=1.5)
            assert result[source]["sar"] > 100

    def test_length_mismatch_trims_instead_of_raising(self):
        """A caller whose estimate is a few samples short/long shouldn't crash the evaluation."""
        s1, s2, rng = _white_sources()
        heart_est = (s1 + 0.05 * rng.standard_normal(N))[:-7]
        lung_est = (s2 + 0.05 * rng.standard_normal(N))[:-3]

        result = evaluate_heart_lung(s1, s2, heart_est, lung_est)

        for source in ("heart", "lung"):
            assert np.isfinite(result[source]["sdr"])
            assert result[source]["sdr"] > 10  # still recognizably a good, near-identity estimate


# ---------------------------------------------------------------------------
# Batch API integration (real dataset, trivial separation function)
# ---------------------------------------------------------------------------

class TestBatchEvaluation:
    def test_evaluate_dataset_and_summarize_by_class_on_real_rows(self):
        mix_df = load_mix().head(4)

        def passthrough(mixed, sr):
            return mixed, mixed  # trivial "no separation" baseline

        results = evaluate_dataset(passthrough, mix_df=mix_df)
        assert set(results["Mixed Sound ID"]) == set(mix_df["Mixed Sound ID"])
        assert set(results["source"]) == {"heart", "lung"}
        assert {"sdr", "sir", "sar"}.issubset(results.columns)

        summary = summarize_by_class(results, "heart", "Heart Sound Type")
        assert len(summary) > 0
