"""
Shared helpers for Baselines 0-5 (baseline1.py .. baseline5.py) -- the
handful of pieces literally reused by more than one baseline: the two
frequency bands, the zero-phase Butterworth bandpass, and Welch-PSD
peak-frequency picking. Anything used by only one baseline stays defined
in that baseline's own module, even where two baselines share an
algorithm family (e.g. Baseline 3 imports Baseline 2's NMF machinery
directly, since Baseline 3 is explicitly an ablation of Baseline 2 --
see baseline3.py -- rather than a second copy of shared code).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

HEART_BAND = (20.0, 200.0)   # Hz
LUNG_BAND = (150.0, 1000.0)  # Hz


def _bandpass(y: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (sosfiltfilt avoids the phase distortion filtfilt-less IIR would add)."""
    nyquist = sr / 2
    low_norm = max(low / nyquist, 1e-6)
    high_norm = min(high / nyquist, 1 - 1e-6)
    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y)


def _peak_frequency(y: np.ndarray, sr: int) -> float:
    """Dominant frequency of y via Welch PSD -- shared by Baseline 4 (MSSA) and Baseline 5 (EVMD)."""
    freqs, psd = welch(y, fs=sr)
    return freqs[np.argmax(psd)]
