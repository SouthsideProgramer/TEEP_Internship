"""
Unit tests for oracle_mask_ceiling.py. Synthetic only -- the properties
checked (an oracle mask on an additive mixture recovers the sources far
above passthrough; the summary/headroom tables have the expected shape and
arithmetic) do not need the dataset.
"""
import numpy as np
import pandas as pd
import pytest

from metrics import SOURCE_LABELS, evaluate_heart_lung
from oracle_mask_ceiling import (
    ORACLE_LABELS,
    REFERENCE_LABELS,
    apply_mask,
    headroom_table,
    oracle_masks,
    oracle_separate,
    summarize,
)

SR = 4000


@pytest.fixture(scope="module")
def two_band_sources():
    """Heart-like low tone burst and lung-like high band noise, spectrally
    disjoint, so an ideal mask can separate them almost perfectly."""
    rng = np.random.default_rng(0)
    t = np.arange(4 * SR) / SR
    heart = 0.5 * np.sin(2 * np.pi * 60 * t) * (1 + np.sign(np.sin(2 * np.pi * 1.2 * t)))
    noise = rng.normal(size=t.size)
    from scipy.signal import butter, sosfiltfilt

    lung = 0.3 * sosfiltfilt(butter(4, [400, 1200], btype="band", fs=SR, output="sos"), noise)
    return heart, lung, heart + lung


class TestOracleMasks:
    def test_mask_shapes_and_ranges(self, two_band_sources):
        heart, lung, _ = two_band_sources
        masks = oracle_masks(heart, lung)
        assert set(masks) == set(ORACLE_LABELS)
        shapes = {m.shape for m in masks.values()}
        assert len(shapes) == 1
        for m in masks.values():
            assert m.min() >= 0.0 and m.max() <= 1.0

    def test_oracle_beats_passthrough_by_a_wide_margin(self, two_band_sources):
        heart, lung, mixed = two_band_sources
        passthrough = evaluate_heart_lung(heart, lung, mixed, mixed)
        for kind in ORACLE_LABELS:
            h_est, l_est = oracle_separate(mixed, heart, lung, kind=kind)
            assert len(h_est) == len(mixed) and len(l_est) == len(mixed)
            oracle = evaluate_heart_lung(heart, lung, h_est, l_est)
            for source in SOURCE_LABELS:
                assert oracle[source]["sdr"] > passthrough[source]["sdr"] + 10.0, (kind, source)

    def test_all_ones_mask_is_passthrough(self, two_band_sources):
        _, _, mixed = two_band_sources
        ones = np.ones_like(oracle_masks(mixed, mixed)["IRM"])
        h_est, _ = apply_mask(mixed, ones)
        np.testing.assert_allclose(h_est, mixed, atol=1e-6)


class TestTables:
    @pytest.fixture
    def fake_results(self):
        rows = []
        for mid in ["M1", "M2"]:
            for method, sdr in [("B0", 0.0), ("B1", 5.0), ("IRM", 15.0), ("IWF", 16.0), ("IBM", 14.0)]:
                for source in SOURCE_LABELS:
                    rows.append({"mixed_id": mid, "method": method, "source": source, "sdr": sdr, "sir": sdr, "sar": sdr})
        return pd.DataFrame(rows)

    def test_summary_shape_and_order(self, fake_results):
        summary = summarize(fake_results)
        assert list(summary.index.get_level_values("method").unique()) == list(REFERENCE_LABELS) + list(ORACLE_LABELS)
        assert (summary["n"] == 2).all()
        assert summary.loc[("IRM", "heart"), "sdr_mean"] == 15.0

    def test_headroom_arithmetic(self, fake_results):
        summary = summarize(fake_results)
        headroom = headroom_table(summary, {"published": {"heart": 3.0, "lung": 1.0}})
        assert headroom.loc[REFERENCE_LABELS["B0"], "headroom_heart_db"] == pytest.approx(15.0)
        assert headroom.loc[REFERENCE_LABELS["B1"], "headroom_lung_db"] == pytest.approx(10.0)
        assert headroom.loc["published", "headroom_heart_db"] == pytest.approx(12.0)
        assert headroom.loc["published", "headroom_lung_db"] == pytest.approx(14.0)
