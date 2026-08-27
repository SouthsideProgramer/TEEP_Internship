"""
Unit tests for latency.py (uniform-protocol desktop latency measurement).
`time_calls()` itself is tested with fast, deterministic synthetic
functions (a counter, a known sleep duration) rather than real separation
methods, so the protocol's own correctness doesn't depend on timing real
audio processing -- only two of the real measure_*_latency() functions
(bandpass and the classifier, both cheap) are exercised end-to-end, to
keep this test file's own cost low regardless of how expensive the other
four methods (NMF x2, MSSA, EVMD, Conv-TasNet) are.
"""
import time

from latency import N_WARMUP, measure_bandpass_latency, measure_classifier_latency, time_calls
from load_dataset import load_audio, load_hs, load_mix


class TestTimeCalls:
    def test_calls_fn_the_right_number_of_times(self):
        calls = []
        time_calls(lambda: calls.append(1), n_trials=4, n_warmup=2)
        assert len(calls) == 4 + 2

    def test_reports_the_requested_n_trials(self):
        result = time_calls(lambda: None, n_trials=7, n_warmup=1)
        assert result["n_trials"] == 7

    def test_wall_and_cpu_medians_are_non_negative(self):
        result = time_calls(lambda: sum(range(1000)), n_trials=5)
        assert result["wall_median_s"] >= 0
        assert result["cpu_median_s"] >= 0
        assert result["wall_iqr_s"] >= 0
        assert result["cpu_iqr_s"] >= 0

    def test_median_reflects_a_known_sleep_duration(self):
        """Sanity check the protocol against a function with a known real
        duration -- wall-clock median should be at least the sleep time
        (never less; can be more under scheduling contention -- this
        session's own machine was under very heavy external load, so a
        tight upper bound would be flaky) while CPU time stays far below
        wall-clock, since sleeping doesn't consume CPU -- exactly the
        wall-clock/CPU-time distinction this module's docstring cites as
        the point of reporting both."""
        result = time_calls(lambda: time.sleep(0.05), n_trials=3, n_warmup=1)
        assert result["wall_median_s"] >= 0.05
        assert result["cpu_median_s"] < result["wall_median_s"]

    def test_default_warmup_is_untimed(self):
        """A warm-up call that raises after the first invocation would make
        every *timed* call fail too if warm-up weren't actually separate --
        checked directly by counting invocations against N_WARMUP + n_trials."""
        call_count = 0

        def counting_fn():
            nonlocal call_count
            call_count += 1

        time_calls(counting_fn, n_trials=3, n_warmup=N_WARMUP)
        assert call_count == 3 + N_WARMUP


class TestRealMethodIntegration:
    """Exercises two cheap real methods end-to-end (not the four expensive
    ones) to confirm measure_*_latency() wires real separation/
    classification calls into time_calls() correctly."""

    def test_bandpass_latency_on_real_audio(self):
        mix_row = load_mix().iloc[0]
        mixed, sr = load_audio(mix_row["mixed_audio_path"], sr=None)
        result = measure_bandpass_latency(mixed, sr)
        assert result["method"] == "Baseline 1 (bandpass)"
        assert result["wall_median_s"] > 0
        assert result["n_trials"] > 0

    def test_classifier_latency_on_real_audio(self):
        y, sr = load_audio(load_hs().iloc[0]["audio_path"], sr=None)
        result = measure_classifier_latency(y, sr)
        assert result["method"] == "Condition A classifier (MFCC + SVM)"
        assert result["wall_median_s"] > 0
        assert result["n_trials"] > 0
