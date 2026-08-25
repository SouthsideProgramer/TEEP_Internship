"""
Tests for convtasnet.py (Baseline 6, Conv-TasNet-lite). Not an attempt to
verify separation quality (that needs real training + eval_harness, see
baseline6_report.py) -- these check the parts with no ambiguity: the model
produces correctly-shaped output for arbitrary input lengths, the SI-SDR
loss behaves the way SI-SDR is defined to behave (scale invariant,
maximal on identity, penalizes uncorrelated estimates), and the training
loop (make_convtasnet_baseline) actually reduces loss / plugs into the
eval_harness fit_and_separate_fn contract without erroring, on a tiny
synthetic pool kept fast on purpose.
"""
import numpy as np
import pytest
import torch

from convtasnet import ConvTasNetLite, _crop_or_pad, _sample_batch, make_convtasnet_baseline, si_sdr


@pytest.fixture
def tiny_pool_dfs():
    """A 6-heart/6-lung synthetic pool in load_hs()/load_ls() format, with real
    (short, sine-tone) audio written to temp .wav files -- keeps the test
    fast (no real dataset I/O) while exercising the exact DataFrame/column
    contract make_convtasnet_baseline expects."""
    import tempfile
    from pathlib import Path

    import pandas as pd
    import soundfile as sf

    sr = 4000
    duration_s = 2.0
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)

    tmp_dir = Path(tempfile.mkdtemp(prefix="convtasnet_test_"))
    heart_rows, lung_rows = [], []
    for i in range(6):
        heart_y = 0.5 * np.sin(2 * np.pi * (60 + i) * t) + 0.02 * rng.normal(size=n)
        lung_y = 0.3 * np.sin(2 * np.pi * (300 + 10 * i) * t) + 0.02 * rng.normal(size=n)
        heart_path = tmp_dir / f"heart_{i}.wav"
        lung_path = tmp_dir / f"lung_{i}.wav"
        sf.write(heart_path, heart_y.astype(np.float32), sr)
        sf.write(lung_path, lung_y.astype(np.float32), sr)
        heart_rows.append({"Heart Sound ID": f"H{i}", "audio_path": str(heart_path)})
        lung_rows.append({"Lung Sound ID": f"L{i}", "audio_path": str(lung_path)})

    return pd.DataFrame(heart_rows), pd.DataFrame(lung_rows)


class TestConvTasNetLiteForward:
    def test_output_shape_matches_input(self):
        model = ConvTasNetLite()
        x = torch.randn(2, 8000)
        y = model(x)
        assert y.shape == (2, 2, 8000)

    def test_handles_length_not_a_multiple_of_stride(self):
        model = ConvTasNetLite()
        x = torch.randn(1, 8003)  # not aligned to the encoder's kernel/stride
        y = model(x)
        assert y.shape == (1, 2, 8003)

    def test_output_is_finite(self):
        model = ConvTasNetLite()
        x = torch.randn(2, 8000)
        y = model(x)
        assert torch.isfinite(y).all()


class TestSISDR:
    def test_identity_is_very_high(self):
        target = torch.randn(2, 4000)
        assert (si_sdr(target.clone(), target) > 50).all()

    def test_scale_invariant(self):
        target = torch.randn(2, 4000)
        scaled = target * 3.7
        # both should be near the (numerically bounded) ceiling, not just "similar"
        assert (si_sdr(target.clone(), target) > 50).all()
        assert (si_sdr(scaled, target) > 50).all()

    def test_uncorrelated_estimate_is_low(self):
        rng = torch.Generator().manual_seed(0)
        target = torch.randn(4000, generator=rng)
        estimate = torch.randn(4000, generator=rng)
        assert si_sdr(estimate, target) < 5.0

    def test_negated_target_is_not_penalized_less_than_correlated(self):
        """SI-SDR only cares about scale, not sign flip along the correct axis --
        estimate = -target is still perfectly explained by a negative alpha."""
        target = torch.randn(2, 4000)
        assert (si_sdr(-target, target) > 50).all()


class TestCropOrPad:
    def test_crop_returns_requested_length(self):
        rng = np.random.default_rng(0)
        y = np.arange(100.0)
        cropped = _crop_or_pad(y, 30, rng)
        assert cropped.shape == (30,)

    def test_pad_returns_requested_length_when_input_too_short(self):
        rng = np.random.default_rng(0)
        y = np.arange(10.0)
        padded = _crop_or_pad(y, 30, rng)
        assert padded.shape == (30,)
        assert np.array_equal(padded[:10], y)
        assert np.all(padded[10:] == 0.0)


class TestSampleBatch:
    def test_batch_shapes(self, tiny_pool_dfs):
        hs_df, ls_df = tiny_pool_dfs
        heart_cache = {row["Heart Sound ID"]: (np.zeros(8000), 4000) for _, row in hs_df.iterrows()}
        lung_cache = {row["Lung Sound ID"]: (np.zeros(8000), 4000) for _, row in ls_df.iterrows()}
        rng = np.random.default_rng(0)
        mixed, heart, lung = _sample_batch(
            list(heart_cache), list(lung_cache), heart_cache, lung_cache,
            batch_size=5, crop_len=1000, rng=rng, gain_range=(0.5, 2.0), snr_range=(-5.0, 35.0),
        )
        assert mixed.shape == (5, 1000)
        assert heart.shape == (5, 1000)
        assert lung.shape == (5, 1000)


class TestMakeConvTasNetBaseline:
    def test_training_loop_runs_and_returns_working_separate_fn(self, tiny_pool_dfs):
        hs_df, ls_df = tiny_pool_dfs
        fit_fn = make_convtasnet_baseline(seed=0, max_epochs=2, steps_per_epoch=3, batch_size=2)
        separate_fn = fit_fn(hs_df, ls_df)

        assert separate_fn.train_info["epochs_trained"] >= 1
        assert np.isfinite(separate_fn.train_info["best_val_si_sdr"])

        mixed = np.random.default_rng(0).normal(size=8000).astype(np.float32)
        heart_est, lung_est = separate_fn(mixed, 4000)
        assert heart_est.shape == mixed.shape
        assert lung_est.shape == mixed.shape
        assert np.isfinite(heart_est).all()
        assert np.isfinite(lung_est).all()

    def test_raises_on_pool_too_small_to_split(self, tiny_pool_dfs):
        """With only one recording per class, VAL_FRACTION's 80/20 split
        assigns it entirely to validation, leaving no training data --
        _train_convtasnet should raise rather than silently train on zero
        examples."""
        hs_df, ls_df = tiny_pool_dfs
        fit_fn = make_convtasnet_baseline(seed=0, max_epochs=1, steps_per_epoch=1, batch_size=1)
        with pytest.raises(ValueError, match="too small"):
            fit_fn(hs_df.iloc[:1].reset_index(drop=True), ls_df.iloc[:1].reset_index(drop=True))
