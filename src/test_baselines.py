"""
Minimal correctness tests for baselines.py's new machinery (Baseline 5,
EVMD). Not an attempt to re-verify the EVMD paper's own K-selection
semantics -- those are already flagged in code_description.md as this
project's interpretation of a genuinely thin source description, not
re-litigated here. This just checks the VMD implementation itself (the
part with no ambiguity: it's a standard, fully-specified algorithm) does
what VMD is supposed to do.
"""
import numpy as np
import pytest

from baselines import _energy_loss_coefficient, _normalized_permutation_entropy, _vmd


@pytest.fixture(scope="module")
def two_tone_signal():
    sr = 4000
    t = np.arange(0, 3, 1 / sr)  # short (3s, not this dataset's 15s) -- keeps the test fast
    rng = np.random.default_rng(0)
    low = np.sin(2 * np.pi * 80 * t)
    high = 0.5 * np.sin(2 * np.pi * 400 * t)
    noise = 0.02 * rng.normal(size=len(t))
    return low + high + noise, sr


class TestVMD:
    def test_modes_reconstruct_the_input_signal(self, two_tone_signal):
        signal, _sr = two_tone_signal
        modes = _vmd(signal, alpha=2000, K=2, max_iter=100)
        assert modes.shape == (2, len(signal))
        assert _energy_loss_coefficient(signal, modes) < 0.05

    def test_more_modes_reconstruct_at_least_as_well(self, two_tone_signal):
        """More degrees of freedom should never make reconstruction meaningfully worse."""
        signal, _sr = two_tone_signal
        loss_k2 = _energy_loss_coefficient(signal, _vmd(signal, alpha=2000, K=2, max_iter=100))
        loss_k5 = _energy_loss_coefficient(signal, _vmd(signal, alpha=2000, K=5, max_iter=100))
        assert loss_k5 <= loss_k2 + 1e-6

    def test_modes_are_real_valued_output(self, two_tone_signal):
        signal, _sr = two_tone_signal
        modes = _vmd(signal, alpha=2000, K=3, max_iter=50)
        assert np.isrealobj(modes)
        assert np.isfinite(modes).all()


class TestNormalizedPermutationEntropy:
    def test_pure_tone_has_low_entropy(self):
        t = np.arange(0, 3, 1 / 4000)
        pure_tone = np.sin(2 * np.pi * 100 * t)
        npe = _normalized_permutation_entropy(pure_tone)
        assert 0.0 <= npe < 0.3  # highly regular/periodic -> few distinct ordinal patterns

    def test_white_noise_has_high_entropy(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=12000)
        npe = _normalized_permutation_entropy(noise)
        assert npe > 0.9  # near-maximal -- every ordinal pattern roughly equally likely

    def test_bounded_in_unit_interval(self, two_tone_signal):
        signal, _sr = two_tone_signal
        assert 0.0 <= _normalized_permutation_entropy(signal) <= 1.0
