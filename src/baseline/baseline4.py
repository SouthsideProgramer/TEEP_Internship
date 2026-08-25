"""
Baseline 4: multi-stage SSA (MSSA), reproducing Han & Quan, "Cardiorespiratory
Sound Separation Using Singular Spectrum Analysis," ICSPS 2025
(papers/Cardiorespiratory_Sound_Separation_Using_Singular_Spectrum_Analysis.pdf).

See code_description.md for the full writeup.

Usage:
    from baseline.baseline4 import fit_ssa_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_ssa_baseline, n_folds=5)
"""
import numpy as np

from baseline.common import _peak_frequency

SSA_WINDOW_LENGTH = 50           # L, Sec. II.A
SSA_CARDIAC_SPLIT_HZ = 250.0     # Sec. II.B
SSA_EIGENVALUE_THRESHOLD_PCT = 100.0 / SSA_WINDOW_LENGTH  # 2% at L=50, Sec. II.B
SSA_CORRELATION_THRESHOLD = 0.50  # Sec. II.B


def _diagonal_average_weights(L: int, K: int) -> np.ndarray:
    """Number of trajectory-matrix entries [r, c] with r+c=n contributing to
    Hankelized index n, for an L x K trajectory matrix (output length L+K-1)."""
    T = L + K - 1
    n = np.arange(T)
    return np.minimum.reduce([n + 1, np.full(T, L), np.full(T, K), T - n])


def _ssa_decompose(y: np.ndarray, L: int):
    """
    Basic SSA step (Sec. II.A): trajectory-matrix embedding, SVD, and
    diagonal-averaging reconstruction of each eigentriple into a length-T
    reconstructed component (RC).

    Each elementary matrix s_i * outer(U[:,i], Vt[i,:]) is rank-1, so its
    Hankelization (averaging entries along each anti-diagonal) equals
    s_i * convolve(U[:,i], Vt[i,:]) divided by the per-diagonal entry count
    -- avoids ever materializing the L x K elementary matrices explicitly.

    Returns (RCs, eigenvalues): RCs shape (d, T), eigenvalues shape (d,),
    d = min(L, K).
    """
    T = len(y)
    K = T - L + 1
    X = np.lib.stride_tricks.sliding_window_view(y, K)[:L]  # L x K trajectory matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d = len(S)
    weights = _diagonal_average_weights(L, K)

    RCs = np.empty((d, T))
    for i in range(d):
        RCs[i] = S[i] * np.convolve(U[:, i], Vt[i, :], mode="full") / weights

    return RCs, S**2


def mssa_separate(mixed: np.ndarray, sr: int):
    """separate_fn(mixed, sr) -> (heart_est, lung_est), the MSSA two-stage algorithm."""
    # Stage 1: cardiac extraction from the raw mixture.
    rcs1, _ = _ssa_decompose(mixed, SSA_WINDOW_LENGTH)
    peak_freqs = np.array([_peak_frequency(rc, sr) for rc in rcs1])
    is_cardiac = peak_freqs <= SSA_CARDIAC_SPLIT_HZ

    heart_est = rcs1[is_cardiac].sum(axis=0)
    residual = rcs1[~is_cardiac].sum(axis=0)

    # Stage 2: respiratory refinement from the stage-1 residual.
    rcs2, eigenvalues2 = _ssa_decompose(residual, SSA_WINDOW_LENGTH)
    relative_variance = eigenvalues2 / eigenvalues2.sum() * 100.0
    is_selected = relative_variance >= SSA_EIGENVALUE_THRESHOLD_PCT

    breath_candidate = rcs2[is_selected].sum(axis=0)
    for j in np.flatnonzero(~is_selected):
        correlation = np.corrcoef(rcs2[j], breath_candidate)[0, 1]
        if correlation > SSA_CORRELATION_THRESHOLD:
            is_selected[j] = True

    lung_est = rcs2[is_selected].sum(axis=0)
    return heart_est, lung_est


def fit_ssa_baseline(_hs_allowed, _ls_allowed):
    """
    fit_and_separate_fn for eval_harness.cross_validate. MSSA has no learned
    parameters -- it's a fixed two-stage decomposition applied identically
    to every mixture -- so the fold's allowed dictionary pool is unused
    here, same as fit_bandpass_baseline.
    """
    return mssa_separate
