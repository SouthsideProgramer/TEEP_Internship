"""
Baseline 2: supervised NMF, adapting Han, Quan, Matuszewski & Corbett,
"Respiratory Disease Classification Using NMF-Enhanced Log-Mel Spectrograms
and Convolutional Recurrent Neural Networks," Sensors 2026, 26(13):4268,
doi:10.3390/s26134268 (papers/Respiratory_Disease_Classification_...pdf --
cross-checked against the primary source, Sec. 3.2.3/3.2.4). Two fixed
dictionaries -- one per source -- are learned from isolated H/L recordings
via KL-divergence multiplicative-update NMF, then frozen and used to solve
for per-mixture activations, per the classic supervised-NMF separation
recipe (Smaragdis 2007). Paper notation: Kr = respiratory (lung) dictionary
rank = 20, Ki = interference (heart) dictionary rank = 10; 100 MU
iterations to fit each dictionary, 60 MU iterations to solve activations
against a held-out mixture with the dictionaries frozen.

See code_description.md for the full writeup (paper cross-check, parameter
provenance, interpretation notes).

Usage:
    from baseline.baseline2 import make_supervised_nmf_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5)
"""
import librosa
import numpy as np
import pandas as pd

from baseline.common import _bandpass

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
