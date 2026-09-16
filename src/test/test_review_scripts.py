"""
Tests for the 2026-09-13 review scripts: additive_audit_sensitivity,
condition_b_stats and sdr_alpha_audit helpers. Synthetic inputs only.
"""
import numpy as np
import pandas as pd

from additive_audit_sensitivity import _apply_shift, _best_shift, audit_row
from condition_b_stats import cluster_bootstrap, mcnemar_midp, permutation_p
from sdr_alpha_audit import audit_curve, select_root


class TestAdditiveSensitivity:
    def test_shift_roundtrip(self):
        x = np.arange(10, dtype=float)
        assert np.allclose(_apply_shift(_apply_shift(x, 3), -3)[:7], x[:7])  # the last 3 samples are lost to padding

    def test_recovers_lag_and_gain_on_synthetic_mixture(self):
        rng = np.random.default_rng(0)
        h, l = rng.normal(size=8000), rng.normal(size=8000)
        mixed = 0.7 * _apply_shift(h + l, 5) + 0.02
        assert _best_shift(mixed, h + l, 100) == 5
        r = audit_row(mixed, h, l)
        assert r["strict"]["residual"] > 0.5            # zero-lag fit fails
        assert r["dc+shift"]["residual"] < 1e-2          # DC + lag fit passes
        assert r["dc+shift"]["lag"] == 5

    def test_unrelated_signals_fail_every_model(self):
        rng = np.random.default_rng(1)
        h, l, m = rng.normal(size=4000), rng.normal(size=4000), rng.normal(size=4000)
        r = audit_row(m, h, l)
        assert all(v["residual"] > 0.9 for v in r.values())


class TestConditionBStats:
    def _mat(self):
        ids = [f"M{i}" for i in range(12)]
        groups = pd.Series(["g0"] * 4 + ["g1"] * 4 + [f"s{i}" for i in range(4)], index=ids)
        mat = pd.DataFrame({"iso": [1] * 9 + [0] * 3, "sep": [1] * 3 + [0] * 9}, index=ids, dtype=float)
        return mat, groups

    def test_cluster_bootstrap_shapes_and_range(self):
        mat, groups = self._mat()
        b = cluster_bootstrap(mat, groups, 200, 0)
        assert set(b) == {"iso", "sep"} and len(b["iso"]) == 200
        assert 0 <= b["sep"].min() and b["iso"].max() <= 1

    def test_permutation_detects_large_effect_and_not_a_null(self):
        ids = [f"M{i}" for i in range(10)]
        groups = pd.Series([f"s{i}" for i in range(10)], index=ids)
        effect = pd.Series([1.0] * 10, index=ids)           # every row: isolated right, separated wrong
        null = pd.Series([1.0, -1.0] * 5, index=ids)         # balanced, mean 0
        assert permutation_p(effect, groups, 4000, 0) < 0.01  # sign-flip: 2/1024 attainable
        assert permutation_p(null, groups, 4000, 0) > 0.5

    def test_mcnemar_counts(self):
        a = pd.Series([1, 1, 0, 0, 1], dtype=float)
        b = pd.Series([1, 0, 0, 1, 0], dtype=float)
        n_ab, n_ba, p = mcnemar_midp(a, b)
        assert (n_ab, n_ba) == (2, 1) and 0 < p <= 1


class TestAlphaAudit:
    def test_monotone_and_non_monotone_curves(self):
        alphas = np.linspace(0, 1, 11)
        mono = np.linspace(30, 5, 11)
        dip = np.array([30, 20, 10, 0, -10, -5, 0, 5, 10, 12, 12], dtype=float)
        assert audit_curve(alphas, mono)["monotone_decreasing"]
        a = audit_curve(alphas, dip)
        assert not a["monotone_decreasing"] and a["alpha_at_min"] == 0.4

    def test_select_root_rule(self):
        alphas = np.linspace(0, 1, 11)
        dip = np.array([30, 20, 10, 0, -10, -5, 0, 5, 10, 12, 12], dtype=float)
        alpha, n_roots, attainable = select_root(alphas, dip, 5.0)
        assert attainable and n_roots == 2 and alpha < 0.3
        assert select_root(alphas, dip, 40.0)[2] is False
        assert select_root(alphas, dip, -20.0)[2] is False
