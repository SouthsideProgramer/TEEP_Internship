"""
Unit tests for sdr_accuracy_curve.py (S6-03, measuring downstream accuracy
at every point in the SDR sweep). Deliberately cheap: builds a real but
tiny sweep (4 rows, Baseline 1 only) via sdr_sweep.build_sdr_sweep() rather
than depending on the full S6-02 production run.

The properties that matter most: only heart-source rows get measured, the
resume/checkpoint behavior actually skips completed points on a re-run
(the "runs unattended, safe to interrupt" requirement), and the fold
consistency assertion actually catches a real mismatch rather than
silently trusting mismatched inputs.
"""
import numpy as np
import pandas as pd
import pytest

from baseline.baseline1 import fit_bandpass_baseline
from load_dataset import load_mix, verify_additive_triplets
from sdr_accuracy_curve import measure_accuracy_at_each_sdr_point, output_path, summarize_accuracy_curve
from sdr_sweep import TARGET_SDR_GRID_DB, build_sdr_sweep

BASELINE_LABEL = "Baseline 1 (bandpass)"


@pytest.fixture(scope="module")
def tiny_mix_df():
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    return valid.iloc[:4].reset_index(drop=True)


@pytest.fixture(scope="module")
def tiny_sweep(tiny_mix_df, tmp_path_factory):
    cache_root = tmp_path_factory.mktemp("sdr_accuracy_cache")
    sweep_df = build_sdr_sweep(
        n_folds=2, seed=0, mix_df=tiny_mix_df, baseline_specs=[(BASELINE_LABEL, fit_bandpass_baseline)],
        cache_root=cache_root,
    )
    return sweep_df, cache_root


class TestMeasureAccuracyAtEachSdrPoint:
    def test_only_heart_rows_are_measured(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep
        assert (sweep_df["source"] == "lung").any()  # sanity: the sweep does contain lung rows

        out_path = tmp_path / "results.csv"
        results_df = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=out_path,
            resume=False, mix_df=tiny_mix_df,
        )
        expected_n = len(tiny_mix_df) * len(TARGET_SDR_GRID_DB)  # heart-source rows only
        assert len(results_df) == expected_n
        assert results_df["correct"].isin([True, False]).all()

    def test_output_is_written_and_reloadable(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep
        out_path = tmp_path / "results.csv"
        measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=out_path,
            resume=False, mix_df=tiny_mix_df,
        )
        assert out_path.is_file()
        reloaded = pd.read_csv(out_path)
        assert len(reloaded) == len(tiny_mix_df) * len(TARGET_SDR_GRID_DB)


class TestResumeBehavior:
    def test_rerun_skips_already_measured_points(self, tiny_sweep, tiny_mix_df, tmp_path, monkeypatch):
        sweep_df, cache_root = tiny_sweep
        out_path = tmp_path / "results.csv"

        full = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=out_path,
            resume=False, mix_df=tiny_mix_df,
        )

        # Simulate an interrupted run: keep only the first half of the checkpoint.
        partial = full.iloc[: len(full) // 2]
        partial.to_csv(out_path, index=False)

        calls = []
        import sdr_accuracy_curve as mod

        original = mod.synthesize_sweep_row

        def counting_synthesize(row, *args, **kwargs):
            calls.append((row["baseline"], row["mixed_id"], row["target_sdr"]))
            return original(row, *args, **kwargs)

        monkeypatch.setattr(mod, "synthesize_sweep_row", counting_synthesize)

        resumed = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=out_path,
            resume=True, mix_df=tiny_mix_df,
        )

        assert len(resumed) == len(full)
        # Only the missing (second) half should have triggered new work.
        assert len(calls) == len(full) - len(partial)

    def test_resumed_results_match_a_fresh_full_run(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep

        fresh_path = tmp_path / "fresh.csv"
        fresh = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=fresh_path,
            resume=False, mix_df=tiny_mix_df,
        )

        resumed_path = tmp_path / "resumed.csv"
        partial = fresh.iloc[:2]
        partial.to_csv(resumed_path, index=False)
        resumed = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=resumed_path,
            resume=True, mix_df=tiny_mix_df,
        )

        key_cols = ["baseline", "mixed_id", "target_sdr"]
        fresh_sorted = fresh.sort_values(key_cols).reset_index(drop=True)
        resumed_sorted = resumed.sort_values(key_cols).reset_index(drop=True)
        pd.testing.assert_series_equal(fresh_sorted["pred"], resumed_sorted["pred"])
        pd.testing.assert_series_equal(fresh_sorted["correct"], resumed_sorted["correct"])


class TestFoldConsistencyCheck:
    def test_raises_on_a_fold_mismatch(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep
        tampered = sweep_df.copy()
        # Corrupt one row's recorded fold so it disagrees with the
        # independently recomputed classifier fold basis.
        tampered.loc[tampered.index[0], "fold"] = -999

        with pytest.raises(AssertionError, match="fold mismatch"):
            measure_accuracy_at_each_sdr_point(
                sweep_df=tampered, n_folds=2, seed=0, cache_root=cache_root, out_path=tmp_path / "out.csv",
                resume=False, mix_df=tiny_mix_df,
            )


class TestSummarizeAccuracyCurve:
    def test_summary_shape_and_accuracy_matches_grouping(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep
        results_df = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=tmp_path / "out.csv",
            resume=False, mix_df=tiny_mix_df,
        )
        curve_df = summarize_accuracy_curve(results_df)
        assert len(curve_df) == len(TARGET_SDR_GRID_DB)  # one baseline x len(grid) target points
        assert curve_df["accuracy"].between(0, 1).all()

        for _, row in curve_df.iterrows():
            group = results_df[
                (results_df["baseline"] == row["baseline"]) & (results_df["target_sdr"] == row["target_sdr"])
            ]
            assert row["accuracy"] == pytest.approx(group["correct"].mean())


class _FakeBackend:
    """Backend stub (S7-06's contract) -- see test_condition_b.py's own
    _FakeBackend for the rationale: proves the `backend` parameter is
    actually used, not silently ignored."""

    FIXED_PREDICTION = "Rhythm Disorder"

    @staticmethod
    def train_fold_classifiers(hs_df, n_folds):
        return {k: f"fake-classifier-fold-{k}" for k in range(n_folds)}

    @staticmethod
    def predict_one(clf, y, sr):
        assert clf.startswith("fake-classifier-fold-")
        return _FakeBackend.FIXED_PREDICTION


class TestBackendParameter:
    def test_measure_accuracy_uses_the_passed_backend(self, tiny_sweep, tiny_mix_df, tmp_path):
        sweep_df, cache_root = tiny_sweep
        results_df = measure_accuracy_at_each_sdr_point(
            sweep_df=sweep_df, n_folds=2, seed=0, cache_root=cache_root, out_path=tmp_path / "out.csv",
            resume=False, mix_df=tiny_mix_df, backend=_FakeBackend,
        )
        assert (results_df["pred"] == _FakeBackend.FIXED_PREDICTION).all()

    def test_output_path_is_backend_specific(self):
        import heart_classifier

        default_path = output_path()
        explicit_default_path = output_path(heart_classifier)
        fake_path = output_path(_FakeBackend)

        assert default_path == explicit_default_path
        assert fake_path != default_path
        assert "_FakeBackend" in fake_path.name
