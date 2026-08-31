"""
Baseline 3: standard NMF, no learned dictionary -- an ablation against
Baseline 2 (baseline2.py). Unlike Baseline 2, there is no per-fold
dictionary-fitting step -- hs_allowed/ls_allowed are ignored -- because
this baseline's whole point is what happens without one: W and H are both
factorized fresh out of each mixture's own spectrogram (rank Ki+Kr=30
total, same as Baseline 2), then the resulting components are split into a
heart-like and lung-like group by spectral centroid (heart components are
the K_HEART lowest-centroid ones) before reconstructing via the same soft-
mask recipe as Baseline 2.

Reuses Baseline 2's NMF machinery and constants directly (not duplicated
here) since this is explicitly the same algorithm family with the
dictionary-fitting step removed, not a separate reproduction.

See code_description.md for the full writeup.

Usage:
    from baseline.baseline3 import make_standard_nmf_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(make_standard_nmf_baseline(seed=0), n_folds=5)
"""
import librosa
import numpy as np

from baseline.baseline2 import ACTIVATION_ITERS, DENOISE_BAND, DICT_ITERS, HOP_LENGTH, K_HEART, K_LUNG, N_FFT, _EPS, _nmf_kl
from baseline.common import _bandpass

STANDARD_NMF_ITERS = DICT_ITERS + ACTIVATION_ITERS


def make_standard_nmf_baseline(seed: int = 0):
    """
    fit_and_separate_fn factory for eval_harness.cross_validate. There is
    no per-fold dictionary-fitting step -- hs_allowed/ls_allowed are
    unused, matching this baseline's ablation purpose.
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
