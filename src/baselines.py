"""
Baseline separation methods for the heart/lung mix set -- the "does the real
method even beat a filter" sanity floor referenced in PROTOCOL.md 5.2/8.

Baseline 1: simple bandpass filtering. Zero-training, non-adaptive: a fixed
Butterworth bandpass per source, applied identically to every mixture. Any
learned method (NMF, SSA, ...) should comfortably beat this; if it doesn't,
that's a sign of a bug, not a hard separation problem.

Cutoffs were chosen from the average Welch PSD of HS.csv/LS.csv's isolated
recordings (not just literature defaults): heart energy dominates below
~150-200 Hz, lung takes over from ~200 Hz up to ~700-800 Hz, and both fall
into the noise floor above ~1 kHz at this dataset's 4000 Hz sample rate. The
heart/lung bands below therefore overlap on purpose in the 150-200 Hz
region -- that overlap is real (both sources have genuine energy there) and
is exactly the "spectral overlap" failure mode this baseline is meant to
demonstrate, not an implementation bug.

Baseline 2: supervised NMF, adapting Han, Quan, Matuszewski & Corbett,
"Respiratory Disease Classification Using NMF-Enhanced Log-Mel Spectrograms
and Convolutional Recurrent Neural Networks," Sensors 2026, 26(13):4268,
doi:10.3390/s26134268 (papers/Respiratory_Disease_Classification_...pdf --
now actually cross-checked against the primary source, Sec. 3.2.3/3.2.4,
resolving PROTOCOL.md 5.2's earlier hedge). Two fixed dictionaries -- one per
source -- are learned from isolated H/L recordings via KL-divergence
multiplicative-update NMF, then frozen and used to solve for per-mixture
activations, per the classic supervised-NMF separation recipe (Smaragdis
2007). Paper notation, confirmed: Kr = respiratory (lung) dictionary rank =
20, Ki = interference (heart) dictionary rank = 10; 100 MU iterations to fit
each dictionary, 60 MU iterations to solve activations against a held-out
mixture with the dictionaries frozen. The paper's own auxiliary dictionary
corpus *is* HLS-CMDS's isolated heart/lung recordings (Sec. 3.2.3: "HLS-CMDS
recordings were used exclusively for dictionary learning") -- this also
resolves PROTOCOL.md 2's "conflicting" `[nmfcrnn]` dataset row: HLS-CMDS is
the auxiliary dictionary source, while ICBHI 2017 + Fraiwan CWLS (Sec. 4.1)
is the *classification* dataset the paper's accuracy numbers are reported
on. Not a conflict -- two different datasets for two different roles in the
same paper.

ADAPTATION, not literal reproduction: the paper's own pipeline is
single-sided -- it only ever reconstructs the *respiratory* (lung) signal
via M_r = V-hat_r / (V-hat_r + V-hat_i) and discards the interference
(heart) component entirely (Sec. 3.2.4); it never reports a separated heart
signal or any separation-quality metric (SDR/SNR/etc.) for either source,
only downstream classification accuracy/Macro-F1 on the enhanced respiratory
output. This project needs both sources back out (its own research question
is heart+lung separation quality, not respiratory-only enhancement for
classification), so this implementation extends the paper's mask to a
symmetric two-source split: mask_heart = heart_mag / (heart_mag +
lung_mag), mask_lung = 1 - mask_heart. The dictionary-learning and
activation-solving math is faithful to the paper; the two-sided
reconstruction is this project's own extension of it.

PIPELINE ORDER (Sec. 3.2.1, initially missed): the paper denoises every
snippet -- both the isolated dictionary-training recordings and the mixture
being separated -- with a 4th-order Butterworth bandpass (50-1800 Hz) *before*
STFT/NMF, to strip baseline drift and high-frequency acquisition noise.
That's a single broadband pre-filter shared by both sources, distinct in
purpose from Baseline 1's per-source 20-200/150-1000 Hz bands (which *are*
the separation, not a pre-filter for one). Added here (`DENOISE_BAND`,
applied in `_fit_dictionary` and both `separate()` closures below) to match
the paper's actual pipeline order, not just its NMF math -- the initial
reproduction skipped this stage entirely.

LEAKAGE TRAP (see PROTOCOL.md 5.1/split.py): a third of HS.csv/LS.csv's
recordings are byte-identical to a heart/lung component of some Mix.csv
row -- the row's own ground truth. A naive reproduction that fits the
dictionaries on "all of HS.csv/LS.csv" therefore lets a mixture's own
ground-truth source into its separation dictionary. This implementation
only ever sees `hs_allowed`/`ls_allowed` -- eval_harness.cross_validate's
fold-safe pool, with every recording that appears in the held-out fold's
mixtures already excluded (same triplet-level split as S2-03) -- so its
score is expected to come in lower than a naive reproduction's. That lower
number is the correct one; a higher one would mean leakage, not a better
model.

Baseline 3: standard NMF, no learned dictionary (ablation against Baseline 2 /
S3-02; ref S3-03). Same total rank (Ki+Kr=30), STFT params, and DENOISE_BAND
pre-filter as Baseline 2, but W and H are both factorized directly out of
each held-out mixture's own (denoised) spectrogram -- no dictionary-learning
phase from isolated recordings, so hs_allowed/ls_allowed go unused, same as
Baseline 1. Keeping the denoising step identical to Baseline 2 matters here:
the point of this ablation is to isolate the effect of the *learned
dictionary* specifically, so every other stage of the pipeline (denoise,
STFT params, mask/reconstruction recipe) is held fixed between the two.
This isolates how much
of Baseline 2's score comes from the learned/frozen dictionary versus NMF
factorization alone. The resulting components are unlabeled by construction;
they're assigned to heart/lung post-hoc by spectral centroid (heart energy
concentrated below ~200 Hz, per Baseline 1's own PSD survey) -- a fixed
physical prior, not a source label, so the method stays genuinely blind.

Baseline 4: multi-stage Singular Spectrum Analysis (MSSA), reproducing Han &
Quan, "Cardiorespiratory Sound Separation Using Singular Spectrum Analysis,"
2025 17th IEEE Int'l Conf. on Signal Processing Systems (ICSPS), doi:
10.1109/ICSPS66615.2025.11347745 (`papers/Cardiorespiratory_Sound_Separation_
Using_Singular_Spectrum_Analysis.pdf`). Zero-training like Baseline 1 --
`hs_allowed`/`ls_allowed` go unused, no leakage trap applies (there is no
fitting step at all). All four hyperparameters are stated explicitly in the
paper and used here as-is:

- `SSA_WINDOW_LENGTH = 50` (Sec. II.A: "a window length of 50 was selected
  to achieve a balance between processing speed and signal separability").
- `SSA_CARDIAC_SPLIT_HZ = 250.0` (Sec. II.B: "components with dominant
  frequencies below or equal to 250 Hz are classified as cardiac-related
  signals... S1 primarily occupies 100-200 Hz, while S2 extends up to
  250 Hz"). This is a physiological frequency in absolute Hz, not a value
  normalized to the paper's own sample rate, so it's valid at any sample
  rate whose Nyquist clears it -- including this project's 4000 Hz HLS_CMDS
  copy (Nyquist 2000 Hz, 8x above the split). The paper's own dataset is
  sourced from the same Torabi et al. HLS-CMDS descriptor paper this project
  cites (their ref [19] = this project's own PROTOCOL.md Sec. 1 citation),
  so there is no cross-dataset sample-rate mismatch here either.
- `SSA_EIGENVALUE_THRESHOLD_PCT` (Sec. II.B: "the SSA decomposition produces
  50 RC layers, an average contribution per mode is calculated as 2%...
  a threshold of 2% is set"). Derived as `100/SSA_WINDOW_LENGTH` rather than
  hardcoded, so it stays principled if the window length ever changes; it
  equals exactly 2.0 for L=50, matching the paper.
- `SSA_CORRELATION_THRESHOLD = 0.50` (Sec. II.B: "modes showing a
  correlation above 50% are additionally included").

Two-stage algorithm (Sec. II.A "Basic Algorithm" for the SSA math, II.B
"Proposed Method" for the two stages and thresholds):

  Stage 1 (cardiac): SSA-decompose the raw mixture (L=50) into 50
  reconstructed components (RCs) via trajectory-matrix embedding + SVD +
  diagonal averaging (Hankelization). Each RC's dominant frequency (Welch
  PSD peak) sorts it into cardiac (<=250 Hz, summed directly into the final
  heart_est -- Fig. 1 routes this branch straight to output, no stage-2
  refinement) or into a residual pool (>250 Hz) that feeds stage 2.

  Stage 2 (respiratory): SSA-decompose the residual (L=50 again) into a
  fresh 50 RCs. RCs whose relative eigenvalue contribution clears the 2%
  threshold are the initial "high-energy" respiratory set; each remaining
  RC is then Pearson-correlated against that set's sum and added if the
  correlation exceeds 50%. lung_est is the sum of the final selected set.
  (Interpretation choice, since the paper doesn't fully spell out what "the
  remaining modes" are correlated against: correlating each leftover RC
  against the sum of the already-selected high-energy modes is the natural
  reading of "the identified high-energy modes vs. the remaining modes" --
  flagged here the same way Baseline 2's two-sided-mask extension is
  flagged as this project's own interpretation, not literal paper text.)

Usage:
    from baselines import fit_bandpass_baseline, make_supervised_nmf_baseline, make_standard_nmf_baseline, fit_ssa_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(make_standard_nmf_baseline(seed=0), n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(fit_ssa_baseline, n_folds=5)
"""
import librosa
import numpy as np
import pandas as pd
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


def _run_synthetic_report(label: str, separate_fn, pairs: list[dict]) -> pd.DataFrame:
    """
    Evaluate a plain separate_fn(mixed, sr) -> (heart_est, lung_est) over the
    synthetic pairs and report mean SDR + Pearson correlation per source, to
    compare directly against the SSA paper's Table I (SDR/Correlation
    columns; no STOI here -- `pystoi` isn't in requirements.txt/`.venv` and
    this project didn't want to add a new dependency unasked-for, so STOI is
    a known gap relative to the paper's third metric).
    """
    from metrics import evaluate_heart_lung

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
    print(f"=== {label} (synthetic, {len(pairs)} pairs) ===")
    print(df.mean().to_string())
    print()
    return df


def _run_and_report(label: str, fit_fn, full_mix_df: pd.DataFrame, valid_mix_df: pd.DataFrame, seed: int = 0):
    """
    Run fit_fn's cross-validation on both the full Mix.csv and the
    additive-only subset (load_dataset.verify_additive_triplets()), and
    print both. Per TEEP2026_Sprint0_Review: the full-145-row headline is
    known to be diluted by 109 rows whose "mixed" file is acoustically
    unrelated to its named heart/lung sources on the current (GitHub, not
    yet Mendeley) dataset copy -- see README.md's Dataset section. Both
    fold-level (mean+/-std across folds) and row-level (mean/median pooled
    across every evaluated row) numbers are printed and explicitly labeled,
    since they are not interchangeable -- averaging within folds before
    taking std understates row-to-row spread (the review's statistics note).
    """
    from eval_harness import cross_validate, summarize_pooled

    print(f"=== {label} ===\n")

    print(f"-- Full {len(full_mix_df)} rows --")
    results_full, fold_summary_full, cv_summary_full = cross_validate(fit_fn, n_folds=5, seed=seed, mix_df=full_mix_df)
    print("Per-fold means:")
    print(fold_summary_full.to_string(index=False))
    print("\nAcross-fold mean +/- std (fold-level dispersion):")
    print(cv_summary_full.to_string())
    print("\nPooled mean/median/std (row-level dispersion, every evaluated row):")
    print(summarize_pooled(results_full).to_string())

    print(f"\n-- Additive-only {len(valid_mix_df)} rows (mixed ~= a*(heart+lung)) --")
    results_valid, fold_summary_valid, cv_summary_valid = cross_validate(
        fit_fn, n_folds=5, seed=seed, mix_df=valid_mix_df
    )
    print("Per-fold means:")
    print(fold_summary_valid.to_string(index=False))
    print("\nAcross-fold mean +/- std (fold-level dispersion):")
    print(cv_summary_valid.to_string())
    print("\nPooled mean/median/std (row-level dispersion, every evaluated row):")
    print(summarize_pooled(results_valid).to_string())
    print()


if __name__ == "__main__":
    from load_dataset import load_mix, verify_additive_triplets

    full_mix_df = load_mix()
    additivity = verify_additive_triplets(full_mix_df)
    valid_mix_df = full_mix_df[full_mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(
        f"Dataset: {len(full_mix_df)} rows total, {len(valid_mix_df)} pass the additivity check "
        f"(mixed ~= a*(heart+lung)) -- see README.md's Dataset section before trusting either number.\n\n"
    )

    _run_and_report(
        f"Baseline 1 (bandpass): heart={HEART_BAND} Hz, lung={LUNG_BAND} Hz",
        fit_bandpass_baseline,
        full_mix_df,
        valid_mix_df,
    )

    _run_and_report(
        f"Baseline 2 (supervised NMF): Ki(heart)={K_HEART}, Kr(lung)={K_LUNG}, "
        f"dict_iters={DICT_ITERS}, activation_iters={ACTIVATION_ITERS}",
        make_supervised_nmf_baseline(seed=0),
        full_mix_df,
        valid_mix_df,
    )

    _run_and_report(
        f"Baseline 3 (standard NMF, no learned dictionary; ablation against Baseline 2): "
        f"k_total={K_HEART + K_LUNG}, iters={STANDARD_NMF_ITERS}",
        make_standard_nmf_baseline(seed=0),
        full_mix_df,
        valid_mix_df,
    )

    _run_and_report(
        f"Baseline 4 (MSSA, reproducing Han & Quan ICSPS 2025): L={SSA_WINDOW_LENGTH}, "
        f"split={SSA_CARDIAC_SPLIT_HZ} Hz, eigenvalue_threshold={SSA_EIGENVALUE_THRESHOLD_PCT}%, "
        f"correlation_threshold={SSA_CORRELATION_THRESHOLD}",
        fit_ssa_baseline,
        full_mix_df,
        valid_mix_df,
    )

    synthetic_pairs = build_synthetic_mixes(seed=SYNTHETIC_SEED)
    print(
        f"\nSSA-paper-style synthetic set: {len(synthetic_pairs)} pairs "
        f"({SYNTHETIC_N_HEART} heart x {SYNTHETIC_N_LUNG} lung, all combinations), "
        f"{SYNTHETIC_NOISE_RMS_FRAC * 100:.0f}% RMS Gaussian noise added -- see "
        f"paper Table I for the reference numbers (MSSA: cardiac SDR 26.4 dB / "
        f"corr 99.2%, respiratory SDR 5.3 dB / corr 80.5%; Butterworth baseline: "
        f"cardiac SDR 5.7 dB, respiratory SDR -5.7 dB).\n"
    )
    _run_synthetic_report("Baseline 1 (bandpass)", bandpass_separate, synthetic_pairs)
    _run_synthetic_report("Baseline 4 (MSSA)", mssa_separate, synthetic_pairs)
