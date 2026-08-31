"""
Unit tests for sdr_knee_point.py (S6-04's knee-point detection). Uses
synthetic accuracy curves, not real audio -- the crossing logic is pure
arithmetic over a small DataFrame, and its correctness doesn't depend on
which baseline or dataset produced the numbers.
"""
import numpy as np
import pandas as pd
import pytest

from sdr_knee_point import compare_knee_points, find_knee_point, find_knee_points


def _curve(sdrs, accuracies):
    return pd.DataFrame({"mean_achieved_sdr": sdrs, "accuracy": accuracies})


class TestFindKneePoint:
    def test_clean_crossing_is_interpolated(self):
        curve = _curve([20, 15, 10, 5, 0], [0.9, 0.8, 0.6, 0.4, 0.2])
        result = find_knee_point(curve, no_separation_accuracy=0.5)
        assert result["status"] == "crossed"
        assert result["knee_sdr"] == pytest.approx(7.5)
        assert result["n_points"] == 5

    def test_exact_equality_at_a_point_still_resolves(self):
        curve = _curve([10, 5], [0.5, 0.5])
        result = find_knee_point(curve, no_separation_accuracy=0.5)
        assert result["status"] == "always_above"

    def test_always_above_when_separation_never_loses(self):
        curve = _curve([20, 15, 10, 5, 0], [0.9, 0.85, 0.8, 0.75, 0.7])
        result = find_knee_point(curve, no_separation_accuracy=0.5)
        assert result == {"knee_sdr": None, "status": "always_above", "n_points": 5}

    def test_always_below_when_separation_never_wins(self):
        curve = _curve([20, 15, 10, 5, 0], [0.3, 0.25, 0.2, 0.15, 0.1])
        result = find_knee_point(curve, no_separation_accuracy=0.5)
        assert result == {"knee_sdr": None, "status": "always_below", "n_points": 5}

    def test_noisy_crossing_detected_distinctly(self):
        curve = _curve([20, 15, 10, 5], [0.3, 0.3, 0.9, 0.9])
        result = find_knee_point(curve, no_separation_accuracy=0.5)
        assert result["status"] == "noisy_crossing"
        assert result["knee_sdr"] == 15

    def test_order_of_input_rows_does_not_matter(self):
        curve_sorted = _curve([20, 15, 10, 5, 0], [0.9, 0.8, 0.6, 0.4, 0.2])
        curve_shuffled = curve_sorted.sample(frac=1, random_state=0).reset_index(drop=True)
        assert find_knee_point(curve_sorted, 0.5) == find_knee_point(curve_shuffled, 0.5)


class TestFindKneePoints:
    def test_one_row_per_baseline(self):
        curve_df = pd.concat([
            _curve([20, 10, 0], [0.9, 0.6, 0.2]).assign(baseline="A"),
            _curve([20, 10, 0], [0.3, 0.25, 0.1]).assign(baseline="B"),
        ])
        result = find_knee_points(curve_df, no_separation_accuracy=0.5)
        assert set(result.index) == {"A", "B"}
        assert result.loc["A", "status"] == "crossed"
        assert result.loc["B", "status"] == "always_below"


def _knee_df(rows: dict) -> pd.DataFrame:
    """rows: {baseline: (knee_sdr, status)} -> a find_knee_points()-shaped DataFrame."""
    return pd.DataFrame(
        [{"baseline": b, "knee_sdr": sdr, "status": status, "n_points": 7} for b, (sdr, status) in rows.items()]
    ).set_index("baseline")


class TestCompareKneePoints:
    def test_agrees_when_both_crossed_within_tolerance(self):
        arch1 = _knee_df({"Baseline 1": (10.0, "crossed")})
        arch2 = _knee_df({"Baseline 1": (11.5, "crossed")})
        result = compare_knee_points({"arch1": arch1, "arch2": arch2}, tolerance_db=3.0)
        assert result.loc["Baseline 1", "agrees"] == True
        assert result.loc["Baseline 1", "knee_sdr_spread_db"] == pytest.approx(1.5)

    def test_disagrees_when_crossed_but_far_apart(self):
        arch1 = _knee_df({"Baseline 1": (10.0, "crossed")})
        arch2 = _knee_df({"Baseline 1": (2.0, "crossed")})
        result = compare_knee_points({"arch1": arch1, "arch2": arch2}, tolerance_db=3.0)
        assert result.loc["Baseline 1", "agrees"] == False
        assert result.loc["Baseline 1", "knee_sdr_spread_db"] == pytest.approx(8.0)

    def test_inconclusive_when_one_side_has_no_clean_crossing(self):
        arch1 = _knee_df({"Baseline 1": (10.0, "crossed")})
        arch2 = _knee_df({"Baseline 1": (None, "always_above")})
        result = compare_knee_points({"arch1": arch1, "arch2": arch2}, tolerance_db=3.0)
        assert result.loc["Baseline 1", "agrees"] is None
        assert pd.isna(result.loc["Baseline 1", "knee_sdr_spread_db"])

    def test_only_compares_baselines_present_in_every_backend(self):
        arch1 = _knee_df({"Baseline 1": (10.0, "crossed"), "Baseline 2": (5.0, "crossed")})
        arch2 = _knee_df({"Baseline 1": (11.0, "crossed")})
        result = compare_knee_points({"arch1": arch1, "arch2": arch2})
        assert set(result.index) == {"Baseline 1"}

    def test_requires_at_least_two_backends(self):
        arch1 = _knee_df({"Baseline 1": (10.0, "crossed")})
        with pytest.raises(ValueError, match="at least two"):
            compare_knee_points({"arch1": arch1})
