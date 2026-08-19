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

Baseline 2: supervised NMF (reproduce Sensors 2026). Two fixed dictionaries
-- one per source -- are learned from isolated H/L recordings via KL-
divergence multiplicative-update NMF, then frozen and used to solve for
per-mixture activations, per the classic supervised-NMF separation recipe
(Smaragdis 2007). Paper notation, confirmed against the paper directly:
Kr = lung dictionary rank = 20, Ki = heart dictionary rank = 10; 100 MU
iterations to fit each dictionary, 60 MU iterations to solve activations
against a held-out mixture with the dictionaries frozen.

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

Usage:
    from baselines import fit_bandpass_baseline, make_supervised_nmf_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5)
    results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5)
"""
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

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
HOP_LENGTH = 128  # 32 ms hop
_EPS = 1e-10


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
    H = rng.random((k, t)) + _EPS

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
        y, _sr = librosa.load(path, sr=None)
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
            S_mix = librosa.stft(mixed, n_fft=N_FFT, hop_length=HOP_LENGTH)
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
