"""
Baseline separation methods for the heart/lung mix set -- the "does the real
method even beat a filter" sanity floor referenced in PROTOCOL.md 5.2/8.

See code_description.md for the full writeup of each baseline (bandpass,
supervised NMF, standard NMF, SSA, EVMD): papers, parameter choices, the
ablation vs. leakage-trap reasoning, and the interpretation calls flagged as
this project's own extensions of the source papers.

Usage:
    from baselines import fit_bandpass_baseline, make_supervised_nmf_baseline, make_standard_nmf_baseline, fit_ssa_baseline, fit_evmd_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(make_standard_nmf_baseline(seed=0), n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(fit_ssa_baseline, n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(fit_evmd_baseline, n_folds=5)
"""
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from metrics import SOURCE_LABELS

HEART_BAND = (20.0, 200.0)   # Hz
LUNG_BAND = (150.0, 1000.0)  # Hz


def _bandpass(y: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (sosfiltfilt avoids the phase distortion filtfilt-less IIR would add)."""
    nyquist = sr / 2
    low_norm = max(low / nyquist, 1e-6)
    high_norm = min(high / nyquist, 1 - 1e-6)
    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y)


def raw_mixture_separate(mixed: np.ndarray, _sr: int):
    """
    separate_fn(mixed, sr) -> (heart_est, lung_est) that performs no
    separation at all -- both "estimates" are just the mixture. This is the
    reference point every other baseline's SDR must be read against: SDR
    already reflects a mixture-vs-two-references orthogonal projection even
    with zero separation, so e.g. "Baseline 1 gets 5.46 dB SDR" is only
    meaningful once you know what passthrough alone already scores on the
    same rows. Also the fastest way to sanity-check the additivity axis
    itself (see build_synthetic_mixes/verify_additive_triplets): on rows
    where mixed != a*(heart+lung), a large chunk of the mixture's own energy
    isn't in either reference's span, so *every* method -- including this
    zero-op one -- inherits negative SAR from that non-additive residual.
    """
    return mixed, mixed


def fit_raw_mixture_baseline(_hs_allowed, _ls_allowed):
    """fit_and_separate_fn for eval_harness.cross_validate. No learned parameters, no dictionary pool needed."""
    return raw_mixture_separate


def bandpass_separate(mixed: np.ndarray, sr: int, heart_band=HEART_BAND, lung_band=LUNG_BAND):
    """separate_fn(mixed, sr) -> (heart_est, lung_est), per the metrics.py/eval_harness.py contract."""
    heart_est = _bandpass(mixed, sr, *heart_band)
    lung_est = _bandpass(mixed, sr, *lung_band)
    return heart_est, lung_est


def fit_bandpass_baseline(_hs_allowed, _ls_allowed):
    """
    fit_and_separate_fn for eval_harness.cross_validate. Bandpass filtering
    has no learned parameters, so the fold's allowed dictionary pool is
    unused here (the whole point of a zero-training baseline) -- included
    only to match the interface every fold-aware separation method needs to
    satisfy.
    """
    return bandpass_separate


# --- Baseline 2: supervised NMF -------------------------------------------

K_LUNG = 20    # Kr in the paper
K_HEART = 10   # Ki in the paper
DICT_ITERS = 100        # MU iterations to fit each fixed dictionary
ACTIVATION_ITERS = 60   # MU iterations to solve activations on a held-out mixture

N_FFT = 512       # 128 ms window at this dataset's 4000 Hz sample rate
HOP_LENGTH = 256  # 64 ms hop -- Sec. 3.2.3 of the paper: "512-sample analysis window,
                  # 256-sample hop size, and a 512-point FFT" (confirmed from the
                  # primary source; this project previously used 128 unverified)
DENOISE_BAND = (50.0, 1800.0)  # Hz -- Sec. 3.2.1: "a fourth-order Butterworth bandpass
                                # filter with cut-off frequencies of 50 Hz and 1800 Hz,"
                                # applied to every snippet before STFT/NMF, to suppress
                                # baseline drift and high-frequency acquisition noise.
                                # This was missing from the initial reproduction here --
                                # added to match the paper's actual pipeline order
                                # (denoise -> STFT -> NMF), not just its NMF math.
_EPS = 1e-12      # matches the paper's Eq. 3 (V = |S| + epsilon, epsilon = 1e-12)
_H_INIT_FLOOR = 1e-3  # Sec. 3.2.4: "H was initialized with random non-negative
                      # values lower bounded by 10^-3" -- for the activation-solving
                      # (frozen-dictionary) phase specifically


def _nmf_kl(V: np.ndarray, k: int, iters: int, rng: np.random.Generator, W_init: np.ndarray | None = None):
    """
    KL-divergence multiplicative-update NMF (Lee & Seung 2001) on a
    magnitude spectrogram V (freq x time), rank k.

    If W_init is given, W is held fixed and only H is updated each
    iteration -- the "solve activations against a frozen dictionary" half
    of supervised NMF. Otherwise both W and H are updated, and W's columns
    are re-normalized to sum to 1 after every step (standard practice to
    keep the W/H scale split from drifting arbitrarily during dictionary
    learning); with W fixed there is nothing to re-normalize.
    """
    f, t = V.shape
    supervised = W_init is not None
    W = W_init.copy() if supervised else rng.random((f, k)) + _EPS
    H = rng.random((k, t)) + (_H_INIT_FLOOR if supervised else _EPS)

    for _ in range(iters):
        WH = W @ H + _EPS
        H *= (W.T @ (V / WH)) / (W.sum(axis=0, keepdims=True).T + _EPS)
        if not supervised:
            WH = W @ H + _EPS
            W *= ((V / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + _EPS)
            W /= W.sum(axis=0, keepdims=True) + _EPS

    return W, H


def _fit_dictionary(recordings_df: pd.DataFrame, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Concatenate every allowed recording's magnitude spectrogram along time and learn one fixed-rank dictionary."""
    specs = []
    for path in recordings_df["audio_path"]:
        y, sr = librosa.load(path, sr=None)
        y = _bandpass(y, sr, *DENOISE_BAND)
        specs.append(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)))
    V = np.concatenate(specs, axis=1)
    W, _H = _nmf_kl(V, rank, DICT_ITERS, rng)
    return W


def make_supervised_nmf_baseline(seed: int = 0):
    """
    fit_and_separate_fn factory for eval_harness.cross_validate. Fits the
    heart (Ki=10) and lung (Kr=20) dictionaries once per fold from that
    fold's leakage-safe hs_allowed/ls_allowed pool, then returns a
    separate_fn that solves per-mixture activations against those frozen
    dictionaries and reconstructs each source via a soft (Wiener-style)
    mask on the mixture's complex STFT.
    """

    def fit_and_separate(hs_allowed: pd.DataFrame, ls_allowed: pd.DataFrame):
        rng = np.random.default_rng(seed)
        W_heart = _fit_dictionary(hs_allowed, K_HEART, rng)
        W_lung = _fit_dictionary(ls_allowed, K_LUNG, rng)
        W = np.concatenate([W_heart, W_lung], axis=1)
        k_heart = W_heart.shape[1]

        def separate(mixed: np.ndarray, sr: int):
            denoised = _bandpass(mixed, sr, *DENOISE_BAND)
            S_mix = librosa.stft(denoised, n_fft=N_FFT, hop_length=HOP_LENGTH)
            V_mix = np.abs(S_mix)
            _W, H = _nmf_kl(V_mix, W.shape[1], ACTIVATION_ITERS, rng, W_init=W)

            heart_mag = W[:, :k_heart] @ H[:k_heart]
            lung_mag = W[:, k_heart:] @ H[k_heart:]
            mask_heart = heart_mag / (heart_mag + lung_mag + _EPS)

            heart_est = librosa.istft(mask_heart * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
            lung_est = librosa.istft((1.0 - mask_heart) * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
            return heart_est, lung_est

        return separate

    return fit_and_separate


# --- Baseline 3: standard NMF, no learned dictionary -----------------------

STANDARD_NMF_ITERS = DICT_ITERS + ACTIVATION_ITERS  # same total MU-update budget as Baseline 2's two phases combined


def make_standard_nmf_baseline(seed: int = 0):
    """
    fit_and_separate_fn factory for eval_harness.cross_validate. Unlike
    make_supervised_nmf_baseline, there is no per-fold dictionary-fitting
    step -- hs_allowed/ls_allowed are ignored -- because this baseline's
    whole point is what happens without one: W and H are both factorized
    fresh out of each mixture's own spectrogram (rank Ki+Kr=30 total, same
    as Baseline 2), then the resulting components are split into a
    heart-like and lung-like group by spectral centroid (heart components
    are the K_HEART lowest-centroid ones) before reconstructing via the
    same soft-mask recipe as Baseline 2.
    """

    def fit_and_separate(_hs_allowed, _ls_allowed):
        rng = np.random.default_rng(seed)
        k_total = K_HEART + K_LUNG

        def separate(mixed: np.ndarray, sr: int):
            denoised = _bandpass(mixed, sr, *DENOISE_BAND)
            S_mix = librosa.stft(denoised, n_fft=N_FFT, hop_length=HOP_LENGTH)
            V_mix = np.abs(S_mix)
            W, H = _nmf_kl(V_mix, k_total, STANDARD_NMF_ITERS, rng)

            freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
            centroids = (W * freqs[:, None]).sum(axis=0) / (W.sum(axis=0) + _EPS)
            is_heart = np.zeros(k_total, dtype=bool)
            is_heart[np.argsort(centroids)[:K_HEART]] = True

            heart_mag = W[:, is_heart] @ H[is_heart]
            lung_mag = W[:, ~is_heart] @ H[~is_heart]
            mask_heart = heart_mag / (heart_mag + lung_mag + _EPS)

            heart_est = librosa.istft(mask_heart * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
            lung_est = librosa.istft((1.0 - mask_heart) * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
            return heart_est, lung_est

        return separate

    return fit_and_separate


# --- Baseline 4: multi-stage SSA (MSSA) -------------------------------------

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


def _peak_frequency(y: np.ndarray, sr: int) -> float:
    """Dominant frequency of y via Welch PSD (Sec. II.B, Eq. 3)."""
    freqs, psd = welch(y, fs=sr)
    return freqs[np.argmax(psd)]


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


# --- Baseline 5: EVMD (S4-01) -----------------------------------------------
#
# Reproducing the separation stage (Sec. II.B) of Puneet, Shankar, Koluguri &
# Srivastava, "Edge-Enabled Portable Classifier for Lung Sounds Using
# Convolutional Neural Networks," IEEE BioCAS 2025, doi:
# 10.1109/BioCAS67066.2025.00016 (`papers/Edge-Enabled_Portable_Classifier_
# for_Lung_Sounds_Using_Convolutional_Neural_Networks.pdf`; `[edgelung]` in
# PROTOCOL.md). That paper runs EVMD-based lung isolation directly on
# HLS-CMDS mixtures but reports no separation metric at all -- this baseline
# computing SDR/SIR/SAR for it is a direct instance of the gap this project
# exists to close. See code_description.md for the full writeup, including
# every place this reproduction has to fill in what the paper's own
# description leaves ambiguous.

EVMD_ALPHA = 2000.0             # Sec. II.B: "a balancing parameter alpha = 2000"
EVMD_K_MIN, EVMD_K_MAX = 2, 10  # Sec. II.B: "begins with K=2 and incrementally increases up to K=10"
EVMD_MU1 = 0.01                 # Energy Loss Coefficient threshold
EVMD_MU2 = 0.4                  # Normalised Permutation Entropy threshold
EVMD_MU3 = 0.3                  # Normalised Permutation Entropy Ratio threshold
EVMD_MU4 = 0.05                 # Kurtosis Index threshold
EVMD_HEART_LOWPASS_HZ = 150.0   # Sec. II.B: heart-isolating lowpass cutoff
EVMD_NPE_EMBED_DIM = 5          # not stated in the paper -- a common default
                                 # embedding dimension for permutation entropy
EVMD_MAX_ITER = 100             # this project's own choice, not stated in the
                                 # paper -- VMD literature commonly converges
                                 # within 100-200 iterations at tol=1e-6, and
                                 # the full K=2..10 sweep run over thousands of
                                 # synthetic mixtures (synthetic_mix.py) makes
                                 # runtime a real constraint
EVMD_TOL = 1e-6


def _lowpass(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass (companion to _bandpass, needed for EVMD's single-cutoff heart isolation)."""
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

    freqs = np.fft.fftfreq(T_ext)  # cycles/sample, in [-0.5, 0.5)
    f_hat = np.fft.fft(f_mirror)

    omega = 0.5 * np.arange(K) / K  # uniform init over [0, 0.5) cycles/sample
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

    return EVMD_K_MAX, modes, False  # `modes` is already the K=EVMD_K_MAX decomposition from the loop's last pass


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


# --- SSA-paper-style synthetic evaluation set -------------------------------

SYNTHETIC_N_HEART = 10           # Sec. III: "10 random select cardiac"
SYNTHETIC_N_LUNG = 5             # Sec. III: "5 random select respiratory"
SYNTHETIC_NOISE_RMS_FRAC = 0.02  # Sec. III: "2% of the RMS value of the combined signal"
SYNTHETIC_SEED = 0               # this project's own choice -- the paper doesn't
                                  # say which 10/5 recordings it picked, so this
                                  # set is reproducible but not literally theirs


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def build_synthetic_mixes(seed: int = SYNTHETIC_SEED) -> list[dict]:
    """
    Reproduces the SSA paper's own evaluation-set recipe (Sec. III), not
    HLS_CMDS's own Mix.csv: pick SYNTHETIC_N_HEART heart + SYNTHETIC_N_LUNG
    lung recordings from this project's own HS.csv/LS.csv, form every
    combinatorial pair (10x5=50), sum each pair, then add Gaussian noise at
    SYNTHETIC_NOISE_RMS_FRAC of the combined signal's RMS amplitude. This
    gives this project a like-for-like comparison against the paper's Table I
    (`src/baselines.py`'s own real-mixture numbers, from `eval_harness`, are
    not directly comparable -- see PROTOCOL.md's `[ssa]` row in Sec. 2).
    """
    from load_dataset import load_audio, load_hs, load_ls

    hs_df, ls_df = load_hs(), load_ls()
    rng = np.random.default_rng(seed)
    heart_rows = hs_df.iloc[rng.choice(len(hs_df), size=SYNTHETIC_N_HEART, replace=False)]
    lung_rows = ls_df.iloc[rng.choice(len(ls_df), size=SYNTHETIC_N_LUNG, replace=False)]

    pairs = []
    for _, heart_row in heart_rows.iterrows():
        heart_y, sr = load_audio(heart_row["audio_path"], sr=None)
        for _, lung_row in lung_rows.iterrows():
            lung_y, _ = load_audio(lung_row["audio_path"], sr=None)
            n = min(len(heart_y), len(lung_y))
            heart_y_n, lung_y_n = heart_y[:n], lung_y[:n]
            combined = heart_y_n + lung_y_n
            noise = rng.normal(0.0, SYNTHETIC_NOISE_RMS_FRAC * _rms(combined), size=n)
            pairs.append(
                {
                    "heart": heart_y_n,
                    "lung": lung_y_n,
                    "mixed": combined + noise,
                    "sr": sr,
                    "heart_id": heart_row["Heart Sound ID"],
                    "lung_id": lung_row["Lung Sound ID"],
                }
            )
    return pairs


def _run_synthetic_report(label: str, separate_fn, pairs: list[dict]) -> str:
    """
    Evaluate a plain separate_fn(mixed, sr) -> (heart_est, lung_est) over the
    synthetic pairs and report mean SDR + Pearson correlation per source, to
    compare directly against the SSA paper's Table I (SDR/Correlation
    columns; no STOI here -- `pystoi` isn't in requirements.txt/`.venv` and
    this project didn't want to add a new dependency unasked-for, so STOI is
    a known gap relative to the paper's third metric).

    Prints only a short progress line; returns an HTML <section> for the
    combined results/baselines_report.html (see report_utils.py).
    """
    from metrics import evaluate_heart_lung
    from report_utils import df_to_html, section

    print(f"Running {label} on synthetic set ({len(pairs)} pairs)...")
    rows = []
    for pair in pairs:
        heart_est, lung_est = separate_fn(pair["mixed"], pair["sr"])
        m = evaluate_heart_lung(pair["heart"], pair["lung"], heart_est, lung_est)
        nh = min(len(pair["heart"]), len(heart_est))
        nl = min(len(pair["lung"]), len(lung_est))
        rows.append(
            {
                "sdr_heart": m["heart"]["sdr"],
                "sdr_lung": m["lung"]["sdr"],
                "corr_heart": float(np.corrcoef(pair["heart"][:nh], heart_est[:nh])[0, 1]),
                "corr_lung": float(np.corrcoef(pair["lung"][:nl], lung_est[:nl])[0, 1]),
            }
        )

    df = pd.DataFrame(rows)
    mean_df = df.mean().to_frame(name="mean").T
    return section(f"{label} (synthetic set)", f"{len(pairs)} pairs, mean over all pairs", df_to_html(mean_df, index_label=""))


def _run_and_report(label: str, fit_fn, full_mix_df: pd.DataFrame, valid_mix_df: pd.DataFrame, seed: int = 0) -> dict:
    """
    Run fit_fn's cross-validation on both the full Mix.csv and the
    additive-only subset (load_dataset.verify_additive_triplets()). Per
    TEEP2026_Sprint0_Review: the full-145-row headline is known to be
    diluted by 109 rows whose "mixed" file is acoustically unrelated to its
    named heart/lung sources on the current (GitHub, not yet Mendeley)
    dataset copy -- see README.md's Dataset section. Both fold-level
    (mean+/-std across folds) and row-level (mean/median pooled across every
    evaluated row) numbers are reported and explicitly labeled, since they
    are not interchangeable -- averaging within folds before taking std
    understates row-to-row spread (the review's statistics note).

    Prints only short progress lines; returns {"html": <section> for the
    combined results/baselines_report.html, "cv_summary_full"/
    "cv_summary_valid": the raw across-fold summaries, so __main__ can build
    a ΔSDR-vs-baseline table without re-running cross_validate}.
    """
    from eval_harness import cross_validate, summarize_pooled
    from report_utils import df_to_html, section

    print(f"Running {label}...")

    print(f"  full {len(full_mix_df)} rows...")
    results_full, fold_summary_full, cv_summary_full = cross_validate(fit_fn, n_folds=5, seed=seed, mix_df=full_mix_df)
    full_body = "\n".join([
        "<h4>Per-fold means</h4>",
        df_to_html(fold_summary_full.set_index(["fold", "source"]), index_label="fold / source"),
        "<h4>Across-fold mean &plusmn; std (fold-level dispersion)</h4>",
        df_to_html(cv_summary_full, index_label="source"),
        "<h4>Pooled mean/median/std (row-level dispersion, every evaluated row)</h4>",
        df_to_html(summarize_pooled(results_full), index_label="source"),
    ])

    print(f"  additive-only {len(valid_mix_df)} rows...")
    results_valid, fold_summary_valid, cv_summary_valid = cross_validate(
        fit_fn, n_folds=5, seed=seed, mix_df=valid_mix_df
    )
    valid_body = "\n".join([
        "<h4>Per-fold means</h4>",
        df_to_html(fold_summary_valid.set_index(["fold", "source"]), index_label="fold / source"),
        "<h4>Across-fold mean &plusmn; std (fold-level dispersion)</h4>",
        df_to_html(cv_summary_valid, index_label="source"),
        "<h4>Pooled mean/median/std (row-level dispersion, every evaluated row)</h4>",
        df_to_html(summarize_pooled(results_valid), index_label="source"),
    ])

    body = (
        f'<h3>Full {len(full_mix_df)} rows</h3>\n{full_body}\n'
        f'<h3>Additive-only {len(valid_mix_df)} rows (mixed &asymp; a&middot;(heart+lung))</h3>\n{valid_body}'
    )
    html = section(label, f"full {len(full_mix_df)} / additive-only {len(valid_mix_df)} rows", body)
    return {
        "html": html,
        "cv_summary_full": cv_summary_full,
        "cv_summary_valid": cv_summary_valid,
    }


def _delta_vs_raw_mixture_section(baseline_results: list, raw_cv_summary_valid: pd.DataFrame) -> str:
    """
    Delta-SDR table (additive-only subset only -- the full-145-row axis
    isn't interpretable, see the report's dek): each baseline's across-fold
    mean SDR minus Baseline 0's (raw mixture, no separation) mean SDR, per
    source. This is the number that actually answers "did separation help
    at all", since a baseline's raw SDR is meaningless without knowing what
    passthrough already scores on the same rows.
    """
    from report_utils import df_to_html, section

    rows = []
    for label, cv_summary_valid in baseline_results:
        row = {"baseline": label}
        for src in SOURCE_LABELS:
            row[f"{src}_sdr"] = cv_summary_valid.loc[src, ("sdr", "mean")]
            row[f"delta_{src}_sdr"] = cv_summary_valid.loc[src, ("sdr", "mean")] - raw_cv_summary_valid.loc[src, ("sdr", "mean")]
        rows.append(row)
    delta_df = pd.DataFrame(rows).set_index("baseline")

    return section(
        "ΔSDR vs. raw mixture (additive-only subset)",
        "across-fold mean SDR minus Baseline 0's (no separation)",
        df_to_html(delta_df, index_label="baseline", float_fmt="{:.2f}"),
    )


if __name__ == "__main__":
    from load_dataset import load_mix, verify_additive_triplets
    from report_utils import report_shell, results_dir, stat_tile, write_report

    print("Loading Mix.csv and checking additivity (mixed ~= a*(heart+lung))...")
    full_mix_df = load_mix()
    additivity = verify_additive_triplets(full_mix_df)
    valid_mix_df = full_mix_df[full_mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(f"Dataset: {len(full_mix_df)} rows total, {len(valid_mix_df)} pass the additivity check")

    baseline_specs = [
        ("Baseline 0 (raw mixture, no separation)", fit_raw_mixture_baseline),
        (f"Baseline 1 (bandpass): heart={HEART_BAND} Hz, lung={LUNG_BAND} Hz", fit_bandpass_baseline),
        (
            f"Baseline 2 (supervised NMF): Ki(heart)={K_HEART}, Kr(lung)={K_LUNG}, "
            f"dict_iters={DICT_ITERS}, activation_iters={ACTIVATION_ITERS}",
            make_supervised_nmf_baseline(seed=0),
        ),
        (
            f"Baseline 3 (standard NMF, no learned dictionary; ablation against Baseline 2): "
            f"k_total={K_HEART + K_LUNG}, iters={STANDARD_NMF_ITERS}",
            make_standard_nmf_baseline(seed=0),
        ),
        (
            f"Baseline 4 (MSSA, reproducing Han & Quan ICSPS 2025): L={SSA_WINDOW_LENGTH}, "
            f"split={SSA_CARDIAC_SPLIT_HZ} Hz, eigenvalue_threshold={SSA_EIGENVALUE_THRESHOLD_PCT}%, "
            f"correlation_threshold={SSA_CORRELATION_THRESHOLD}",
            fit_ssa_baseline,
        ),
    ]

    baseline_run_results = [
        (label, _run_and_report(label, fit_fn, full_mix_df, valid_mix_df)) for label, fit_fn in baseline_specs
    ]
    sections = [r["html"] for _label, r in baseline_run_results]

    raw_cv_summary_valid = baseline_run_results[0][1]["cv_summary_valid"]
    sections.append(
        _delta_vs_raw_mixture_section(
            [(label, r["cv_summary_valid"]) for label, r in baseline_run_results[1:]],
            raw_cv_summary_valid,
        )
    )

    print("Building SSA-paper-style synthetic evaluation set...")
    synthetic_pairs = build_synthetic_mixes(seed=SYNTHETIC_SEED)
    print(
        f"Synthetic set: {len(synthetic_pairs)} pairs "
        f"({SYNTHETIC_N_HEART} heart x {SYNTHETIC_N_LUNG} lung, all combinations), "
        f"{SYNTHETIC_NOISE_RMS_FRAC * 100:.0f}% RMS Gaussian noise added"
    )
    sections.append(_run_synthetic_report("Baseline 0 (raw mixture, no separation)", raw_mixture_separate, synthetic_pairs))
    sections.append(_run_synthetic_report("Baseline 1 (bandpass)", bandpass_separate, synthetic_pairs))
    sections.append(_run_synthetic_report("Baseline 4 (MSSA)", mssa_separate, synthetic_pairs))

    stat_tiles = "\n".join([
        stat_tile("Mix rows", str(len(full_mix_df)), "total"),
        stat_tile("Additive rows", str(len(valid_mix_df)), f"{100 * len(valid_mix_df) / len(full_mix_df):.0f}%"),
        stat_tile("Baselines run", "5", "0-4, + 3 synthetic"),
    ])

    html = report_shell(
        title="Baseline Separation Results",
        eyebrow="HLS-CMDS · separation baselines",
        heading="Baselines 0&ndash;4 + SSA-paper synthetic set",
        dek=(
            "Cross-validated SDR/SIR/SAR for each baseline on the full Mix.csv and the "
            "additive-only subset, plus a synthetic-set comparison against the SSA "
            "paper's own Table I (MSSA: cardiac SDR 26.4 dB / corr 99.2%, respiratory "
            "SDR 5.3 dB / corr 80.5%; Butterworth baseline: cardiac SDR 5.7 dB, "
            "respiratory SDR -5.7 dB). <strong>The full-145-row numbers are not "
            "interpretable as separation quality</strong> -- 109 rows have mixed != "
            "a*(heart+lung), so a large share of the mixture's own energy sits outside "
            "the span of both references and every method (including Baseline 0) "
            "inherits negative SAR from that non-additive residual. Only the "
            "additive-only 36-row numbers, and the ΔSDR-vs-Baseline-0 table below, "
            "should be read as separation quality."
        ),
        stat_tiles=stat_tiles,
        body="\n\n".join(sections),
        footer=(
            "<p><strong>Method.</strong> See <code>baselines.py</code> / "
            "<code>code_description.md</code> for each baseline's parameters and "
            "interpretation notes. Baseline 0 added specifically so every other "
            "baseline's SDR can be read as a delta over doing nothing, not an "
            "absolute number.</p>"
        ),
    )

    report_path = write_report(results_dir() / "baselines_report.html", html)
    print(f"\nReport written to {report_path}")
