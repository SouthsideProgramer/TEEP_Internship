"""
Unit tests for sdr_compute_plane.py. is_pareto_dominated() is pure
arithmetic over small dicts -- tested with synthetic values, not real
measurements, so correctness doesn't depend on which numbers this
project happens to have measured.
"""
from sdr_compute_plane import is_pareto_dominated


class TestIsParetoDominated:
    def test_strictly_worse_on_both_axes_is_dominated(self):
        sdr = {"A": 5.0, "B": 10.0}
        compute = {"A": 100.0, "B": 50.0}
        assert is_pareto_dominated("A", sdr, compute) is True
        assert is_pareto_dominated("B", sdr, compute) is False

    def test_better_sdr_worse_compute_is_not_dominated(self):
        """A trade-off (better on one axis, worse on the other) is never
        dominance -- both methods remain legitimate choices depending on
        what the user weights more."""
        sdr = {"A": 10.0, "B": 5.0}
        compute = {"A": 100.0, "B": 50.0}
        assert is_pareto_dominated("A", sdr, compute) is False
        assert is_pareto_dominated("B", sdr, compute) is False

    def test_identical_values_are_not_dominated(self):
        """Equal on both axes: neither is strictly better, so neither
        dominates the other (the definition requires at least one strict
        inequality)."""
        sdr = {"A": 5.0, "B": 5.0}
        compute = {"A": 100.0, "B": 100.0}
        assert is_pareto_dominated("A", sdr, compute) is False
        assert is_pareto_dominated("B", sdr, compute) is False

    def test_equal_sdr_strictly_better_compute_dominates(self):
        sdr = {"A": 5.0, "B": 5.0}
        compute = {"A": 100.0, "B": 50.0}
        assert is_pareto_dominated("A", sdr, compute) is True
        assert is_pareto_dominated("B", sdr, compute) is False

    def test_three_methods_middle_one_dominated_by_either(self):
        sdr = {"A": 10.0, "B": 5.0, "C": 1.0}
        compute = {"A": 10.0, "B": 20.0, "C": 30.0}
        # A dominates both B and C (better SDR, better compute than each).
        assert is_pareto_dominated("A", sdr, compute) is False
        assert is_pareto_dominated("B", sdr, compute) is True
        assert is_pareto_dominated("C", sdr, compute) is True

    def test_dominance_ignores_the_method_itself(self):
        sdr = {"A": 5.0}
        compute = {"A": 100.0}
        assert is_pareto_dominated("A", sdr, compute) is False
