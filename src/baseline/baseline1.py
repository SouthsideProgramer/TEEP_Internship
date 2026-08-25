"""
Baseline 1: simple bandpass filtering -- the "does the real method even
beat a filter" sanity floor referenced in PROTOCOL.md 5.2/8. Zero-training,
non-adaptive: a fixed Butterworth bandpass per source, applied identically
to every mixture. Any learned method (NMF, SSA, ...) should comfortably
beat this; if it doesn't, that's a sign of a bug, not a hard separation
problem.

Cutoffs were chosen from the average Welch PSD of HS.csv/LS.csv's isolated
recordings (not just literature defaults): heart energy dominates below
~150-200 Hz, lung takes over from ~200 Hz up to ~700-800 Hz, and both fall
into the noise floor above ~1 kHz at this dataset's 4000 Hz sample rate. The
heart/lung bands below therefore overlap on purpose in the 150-200 Hz
region -- that overlap is real (both sources have genuine energy there) and
is exactly the "spectral overlap" failure mode this baseline is meant to
demonstrate, not an implementation bug.

See code_description.md for the full writeup.

Usage:
    from baseline.baseline1 import fit_bandpass_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5)
"""
from baseline.common import HEART_BAND, LUNG_BAND, _bandpass


def bandpass_separate(mixed, sr, heart_band=HEART_BAND, lung_band=LUNG_BAND):
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
