"""
Baseline 5: EVMD (S4-01) -- reproducing the separation stage (Sec. II.B) of
Puneet, Shankar, Koluguri & Srivastava, "Edge-Enabled Portable Classifier
for Lung Sounds Using Convolutional Neural Networks," IEEE BioCAS 2025,
doi: 10.1109/BioCAS67066.2025.00016
(`papers/Edge-Enabled_Portable_Classifier_for_Lung_Sounds_Using_
Convolutional_Neural_Networks.pdf`; `[edgelung]` in PROTOCOL.md). That
paper runs EVMD-based lung isolation directly on HLS-CMDS mixtures but
reports no separation metric at all -- this baseline computing SDR/SIR/SAR
for it is a direct instance of the gap this project exists to close.

See code_description.md for the full writeup, including every place this
reproduction has to fill in what the paper's own description leaves
ambiguous.

Usage:
    from baseline.baseline5 import fit_evmd_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_evmd_baseline, n_folds=5)
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from baseline.common import HEART_BAND, LUNG_BAND, _peak_frequency

EVMD_ALPHA = 2000.0
EVMD_K_MIN, EVMD_K_MAX = 2, 10
EVMD_MU1 = 0.01
EVMD_MU2 = 0.4
EVMD_MU3 = 0.3
EVMD_MU4 = 0.05
EVMD_HEART_LOWPASS_HZ = 150.0
EVMD_NPE_EMBED_DIM = 5
EVMD_MAX_ITER = 100
EVMD_TOL = 1e-6


def _lowpass(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass (companion to common._bandpass, needed for EVMD's single-cutoff heart isolation)."""
    nyquist = sr / 2
    norm = min(cutoff / nyquist, 1 - 1e-6)
    sos = butter(order, norm, btype="lowpass", output="sos")
    return sosfiltfilt(sos, y)


def _vmd(signal: np.ndarray, alpha: float, K: int, max_iter: int = EVMD_MAX_ITER, tol: float = EVMD_TOL) -> np.ndarray:
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso, IEEE Trans.
    Signal Processing 2014), solved by ADMM in the frequency domain --
    implemented directly (no VMD package is installed or in
    requirements.txt), the same way Baseline 4's SSA was implemented from
    scratch rather than pulled from a library.

    Mirrors `signal` at both ends before transforming (standard VMD
    practice, suppresses boundary artifacts in the mode estimates), then
    crops the reconstruction back to the original length.

    Returns modes, shape (K, len(signal)).
    """
    T = len(signal)
    half = T // 2
    f_mirror = np.concatenate([signal[:half][::-1], signal, signal[-half:][::-1]])
    T_ext = len(f_mirror)

    freqs = np.fft.fftfreq(T_ext)
    f_hat = np.fft.fft(f_mirror)

    omega = 0.5 * np.arange(K) / K
    u_hat = np.zeros((K, T_ext), dtype=complex)
    lambda_hat = np.zeros(T_ext, dtype=complex)

    pos = freqs > 0
    for _ in range(max_iter):
        u_hat_prev = u_hat.copy()
        sum_uk = u_hat.sum(axis=0)
        for k in range(K):
            sum_uk -= u_hat[k]
            u_hat[k] = (f_hat - sum_uk + lambda_hat / 2) / (1 + alpha * (freqs - omega[k]) ** 2)
            sum_uk += u_hat[k]

            power = np.abs(u_hat[k][pos]) ** 2
            denom = power.sum()
            if denom > 0:
                omega[k] = (freqs[pos] * power).sum() / denom

        lambda_hat = lambda_hat + (f_hat - u_hat.sum(axis=0))

        change = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (np.sum(np.abs(u_hat_prev) ** 2) + 1e-12)
        if change < tol:
            break

    modes_ext = np.real(np.fft.ifft(u_hat, axis=1))
    return modes_ext[:, half : half + T]


def _normalized_permutation_entropy(x: np.ndarray, m: int = EVMD_NPE_EMBED_DIM) -> float:
    """
    Bandt-Pompe permutation entropy of `x`, normalized to [0, 1] by log(m!)
    -- vectorized (sliding_window_view + argsort, no per-sample Python loop)
    so it stays cheap across the K-sweep's many mode evaluations.
    """
    from math import factorial

    if len(x) < m:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    order = np.argsort(windows, axis=1)
    codes = (order * (m ** np.arange(m))).sum(axis=1)
    counts = np.bincount(codes, minlength=m**m)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(factorial(m)))


def _energy_loss_coefficient(signal: np.ndarray, modes: np.ndarray) -> float:
    """||signal - sum(modes)||^2 / ||signal||^2 -- Sec. II.B's "energy loss coefficient" for a given K."""
    reconstructed = modes.sum(axis=0)
    return float(np.sum((signal - reconstructed) ** 2) / np.sum(signal**2))


def _kurtosis_index(mode: np.ndarray) -> float:
    """
    Normalized excess-kurtosis index in (0, 1], used as EVMD's Kurtosis
    Index fallback check. INTERPRETATION: mu4=0.05 is far too small to be a
    threshold on raw Fisher excess kurtosis (unbounded, ~0 for Gaussian,
    routinely >>1 for physiological transients), so this project reads mu4
    as applying to a normalized index instead -- 1/(1+|excess kurtosis|),
    near 1 for Gaussian-like modes, near 0 for strongly impulsive ones. Not
    stated in the paper; flagged the same way _evmd_select_k's cascade is.
    """
    from scipy.stats import kurtosis

    excess = float(kurtosis(mode, fisher=True, bias=False))
    return 1.0 / (1.0 + abs(excess))


def _evmd_select_k(signal: np.ndarray, sr: int):
    """
    Sec. II.B's K-selection sweep: try K=2..10, alpha=2000, stopping at the
    first K where every mode passes the per-mode acceptance check below;
    energy loss must also clear mu1 at that K.

    INTERPRETATION (the paper's own text is thin here -- flagged per this
    project's convention for filling gaps in a source paper, see Baseline 2/
    4's docstrings): a mode with NPE <= mu2 passes outright (low-complexity,
    clearly structured). A mode with NPE > mu2 additionally passes if BOTH
    (a) its dominant frequency falls inside a recognized heart or lung band
    (this project's own HEART_BAND/LUNG_BAND, from Baseline 1's PSD survey
    -- read as the paper's unspecified "frequency domain signature" check)
    AND (b) its NPE ratio (this mode's NPE over the highest NPE among the K
    modes at this candidate K) is <= mu3. Failing that, a low
    (<= mu4) normalized Kurtosis Index (_kurtosis_index) is treated as a
    fallback pass. If no K in [2, 10] gets every mode through, this falls
    back to K=10 and reports non-convergence (`converged=False`) rather than
    forcing a pass -- a genuine, documented result, not a bug to paper over.

    Returns (K, modes, converged: bool).
    """
    modes = None
    for K in range(EVMD_K_MIN, EVMD_K_MAX + 1):
        modes = _vmd(signal, EVMD_ALPHA, K)
        if _energy_loss_coefficient(signal, modes) >= EVMD_MU1:
            continue

        npes = np.array([_normalized_permutation_entropy(mode) for mode in modes])
        max_npe = npes.max() if len(npes) else 0.0

        if _all_modes_pass(modes, npes, max_npe, sr):
            return K, modes, True

    return EVMD_K_MAX, modes, False


def _all_modes_pass(modes: np.ndarray, npes: np.ndarray, max_npe: float, sr: int) -> bool:
    for mode, npe in zip(modes, npes):
        if npe <= EVMD_MU2:
            continue
        freq = _peak_frequency(mode, sr)
        in_band = (HEART_BAND[0] <= freq <= HEART_BAND[1]) or (LUNG_BAND[0] <= freq <= LUNG_BAND[1])
        npe_ratio = npe / max_npe if max_npe > 0 else 0.0
        if in_band and npe_ratio <= EVMD_MU3:
            continue
        if _kurtosis_index(mode) <= EVMD_MU4:
            continue
        return False
    return True


def evmd_separate(mixed: np.ndarray, sr: int):
    """
    separate_fn(mixed, sr) -> (heart_est, lung_est). EVMD-decompose `mixed`
    (K selected per _evmd_select_k), isolate the mode "corresponding to the
    cardiac frequency band" as the one with the lowest Welch-PSD peak
    frequency (paper: singular "the mode," not a summed cardiac set --
    distinct from Baseline 4's MSSA, which sums all cardiac-band RCs),
    lowpass it at 150 Hz (the paper's own cutoff) for heart_est, then
    subtract from the original mixture for lung_est (paper: "the sound
    component of the heart is subtracted from the original signal, leaving
    behind the residual lung sound").
    """
    _K, modes, _converged = _evmd_select_k(mixed, sr)
    freqs = np.array([_peak_frequency(mode, sr) for mode in modes])
    cardiac_mode = modes[int(np.argmin(freqs))]
    heart_est = _lowpass(cardiac_mode, sr, EVMD_HEART_LOWPASS_HZ)
    lung_est = mixed - heart_est
    return heart_est, lung_est


def fit_evmd_baseline(_hs_allowed, _ls_allowed):
    """
    fit_and_separate_fn for eval_harness.cross_validate. EVMD has no learned
    parameters -- zero-training, same as fit_bandpass_baseline/
    fit_ssa_baseline -- so the fold's allowed dictionary pool is unused here.
    """
    return evmd_separate
