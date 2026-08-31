"""
Unit tests for compute_cost.py (MACs/parameter counts per method). Checks
the *exact* rows' formulas directly against small hand-computable cases,
the *estimate* rows' formulas against their own stated inputs, and cross-
validates real measurements (Conv-TasNet's param count, the classifier's
support-vector count) against independent recomputation rather than
trusting the function's own internal arithmetic circularly.
"""
import numpy as np
import pytest

from compute_cost import (
    bandpass_macs,
    classifier_params_and_macs,
    convtasnet_params_and_macs,
    evmd_macs,
    mssa_macs,
    standard_nmf_macs,
    supervised_nmf_macs,
)


class TestBandpassMacs:
    def test_formula_on_a_small_case(self):
        result = bandpass_macs(n_samples=100)
        assert result["macs_per_inference"] == 100 * 5 * 2 * 2 * 2
        assert result["params"] == 0
        assert result["precision"] == "exact"

    def test_scales_linearly_with_n_samples(self):
        small = bandpass_macs(n_samples=1000)["macs_per_inference"]
        large = bandpass_macs(n_samples=2000)["macs_per_inference"]
        assert large == 2 * small


class TestNmfMacs:
    def test_supervised_params_match_dictionary_shape(self):
        from baseline.baseline2 import K_HEART, K_LUNG, N_FFT

        result = supervised_nmf_macs()
        f = N_FFT // 2 + 1
        assert result["params"] == f * (K_HEART + K_LUNG)

    def test_standard_nmf_has_no_persisted_params(self):
        result = standard_nmf_macs()
        assert result["params"] == 0

    def test_standard_nmf_costs_more_per_mixture_than_supervised(self):
        """The real, expected asymmetry: Baseline 3 is fully transductive
        (more iterations, both W and H update every iteration) while
        Baseline 2 only solves activations against an already-frozen
        dictionary -- Baseline 3 must cost more per mixture."""
        supervised = supervised_nmf_macs()["macs_per_inference"]
        standard = standard_nmf_macs()["macs_per_inference"]
        assert standard > supervised

    def test_supervised_macs_scale_with_activation_iters(self):
        from baseline.baseline2 import ACTIVATION_ITERS, HOP_LENGTH, K_HEART, K_LUNG, N_FFT
        from compute_cost import _nmf_iteration_macs, _stft_frame_count

        f = N_FFT // 2 + 1
        t = _stft_frame_count(60000, N_FFT, HOP_LENGTH)
        k = K_HEART + K_LUNG
        expected_activation = ACTIVATION_ITERS * _nmf_iteration_macs(f, k, t, supervised=True)
        expected_reconstruction = f * k * t
        result = supervised_nmf_macs()
        assert result["macs_per_inference"] == expected_activation + expected_reconstruction


class TestMssaMacs:
    def test_formula_matches_window_length_and_trajectory_size(self):
        from baseline.baseline4 import SSA_WINDOW_LENGTH

        n_samples = 60000
        L = SSA_WINDOW_LENGTH
        K = n_samples - L + 1
        expected_one_svd = 4 * L * L * K + 8 * L**3
        result = mssa_macs(n_samples=n_samples)
        assert result["macs_per_inference"] == 2 * expected_one_svd
        assert result["precision"] == "estimate"

    def test_scales_roughly_linearly_with_n_samples_for_large_n(self):
        small = mssa_macs(n_samples=60000)["macs_per_inference"]
        large = mssa_macs(n_samples=120000)["macs_per_inference"]
        assert large == pytest.approx(2 * small, rel=0.05)


class TestEvmdMacs:
    def test_sums_over_the_full_k_sweep(self):
        from baseline.baseline5 import EVMD_K_MAX, EVMD_K_MIN, EVMD_MAX_ITER

        n_samples = 60000
        half = n_samples // 2
        t_ext = half + n_samples + half
        admm_total = sum(EVMD_MAX_ITER * k * t_ext * 8 for k in range(EVMD_K_MIN, EVMD_K_MAX + 1))
        result = evmd_macs(n_samples=n_samples)
        assert result["macs_per_inference"] >= admm_total
        assert result["macs_per_inference"] == pytest.approx(admm_total, rel=0.05)

    def test_is_the_most_expensive_classical_baseline(self):
        """Matches this project's own measured wall-clock finding (BACKLOG.md:
        EVMD ~15s/mixture, the slowest of baselines 1/3/4/5) -- the MAC
        estimate should reproduce that same relative ordering."""
        from compute_cost import bandpass_macs, mssa_macs, standard_nmf_macs

        evmd = evmd_macs()["macs_per_inference"]
        others = [bandpass_macs()["macs_per_inference"], standard_nmf_macs()["macs_per_inference"], mssa_macs()["macs_per_inference"]]
        assert evmd > max(others)


class TestConvTasNetMacsAndParams:
    def test_param_count_matches_direct_sum(self):
        import torch

        from convtasnet import ConvTasNetLite

        model = ConvTasNetLite()
        expected = sum(p.numel() for p in model.parameters())
        result = convtasnet_params_and_macs()
        assert result["params"] == expected

    def test_param_count_is_in_the_previously_measured_ballpark(self):
        result = convtasnet_params_and_macs()
        assert 300_000 <= result["params"] <= 350_000

    def test_macs_are_positive_and_precision_is_exact(self):
        result = convtasnet_params_and_macs()
        assert result["macs_per_inference"] > 0
        assert result["precision"] == "exact"


class TestClassifierParamsAndMacs:
    def test_support_vector_count_matches_the_fitted_model(self):
        from heart_classifier import assign_classifier_folds, train_fold_classifiers

        hs_df = assign_classifier_folds(n_folds=5, seed=0)
        classifiers = train_fold_classifiers(hs_df, n_folds=5)
        expected_n_support = classifiers[0].named_steps["svm"].support_vectors_.shape[0]

        result = classifier_params_and_macs()
        assert result["params"] == expected_n_support
        assert result["params"] > 0
        assert result["macs_per_inference"] > 0
