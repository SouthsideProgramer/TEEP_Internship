"""
Unit tests for sdr_knee_bootstrap.py -- synthetic per-row tables only.
"""
import numpy as np
import pandas as pd

from sdr_knee_bootstrap import curve_and_knee_on_rows
from sdr_knee_point import find_knee_point


def _per_row(ids, targets, acc_by_target):
    rows = []
    rng = np.random.default_rng(0)
    for t, acc in zip(targets, acc_by_target):
        correct = np.zeros(len(ids), dtype=bool)
        correct[: int(round(acc * len(ids)))] = True
        rng.shuffle(correct)
        for i, mid in enumerate(ids):
            rows.append({"baseline": "B", "mixed_id": mid, "target_sdr": t, "achieved_sdr": t + 0.1 * i, "correct": correct[i]})
    return pd.DataFrame(rows)


def test_full_draw_reproduces_headline_knee():
    ids = np.array([f"M{i:03d}" for i in range(12)])
    targets = [-5, 0, 5, 10, 15, 20, 25]
    per_row = _per_row(ids, targets, [0.25, 0.25, 0.33, 0.42, 0.5, 0.58, 0.67])
    no_sep = pd.Series([True] * 5 + [False] * 7, index=ids)
    got = curve_and_knee_on_rows(per_row, no_sep, ids, "B")
    curve = per_row.groupby("target_sdr").agg(mean_achieved_sdr=("achieved_sdr", "mean"), accuracy=("correct", "mean")).reset_index()
    want = find_knee_point(curve, float(no_sep.mean()))
    assert got["status"] == want["status"] == "crossed"
    assert abs(got["knee_sdr"] - want["knee_sdr"]) < 1e-9


def test_duplicate_draw_weights_rows():
    ids = np.array(["A", "B"])
    per_row = pd.DataFrame([
        {"baseline": "B", "mixed_id": "A", "target_sdr": 0, "achieved_sdr": 0.0, "correct": True},
        {"baseline": "B", "mixed_id": "B", "target_sdr": 0, "achieved_sdr": 10.0, "correct": False},
        {"baseline": "B", "mixed_id": "A", "target_sdr": 25, "achieved_sdr": 25.0, "correct": True},
        {"baseline": "B", "mixed_id": "B", "target_sdr": 25, "achieved_sdr": 25.0, "correct": True},
    ])
    no_sep = pd.Series([False, True], index=ids)
    # draw A three times, B once: reference = 1/4, low-SDR accuracy = 3/4 -> always above
    got = curve_and_knee_on_rows(per_row, no_sep, np.array(["A", "A", "A", "B"]), "B")
    assert got["status"] == "always_above"
    # draw B three times, A once: reference = 3/4, low-SDR accuracy = 1/4, high = 1 -> crossed
    got = curve_and_knee_on_rows(per_row, no_sep, np.array(["B", "B", "B", "A"]), "B")
    assert got["status"] == "crossed"
