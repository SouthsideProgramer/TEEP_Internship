"""
One row-level artifact from which every downstream table is derived.

Why. The report's separation-quality table for the native additive subset
and its Condition B / knee-point tables were produced by different
scripts on different days and, for two methods, from different runs:
Baseline 2's table entry predated the primary-source corrections to its
hop and pre-denoising, and Baseline 6's came from a separate training run
(seed-fixed but GPU-nondeterministic, 5.15 vs 5.02 dB). Bandpass's 5.61
vs 5.46 was the same run summarised two ways (fold mean of fold means vs
pooled row mean). A reviewer cannot tell those cases apart from the
tables. This module scores the ONE set of separated waveforms that the
classification experiments actually consumed -- results/sdr_sweep_cache/,
the S6-02 run of all six baselines -- and writes a single CSV with, per
(mixed_id, method, source): fold, SDR/SIR/SAR, and the Architecture 1 and
Architecture 2 predictions on the method's real output. Baseline 0 (the
raw mixture) and the isolated reference are included as methods so the
no-separation and isolated rows come from the same file.

results/canonical_rows.csv columns:
    mixed_id, fold, leak_group, method, source, sdr, sir, sar,
    true_class, pred_svm, pred_cnn, heart_type, lung_type
Predictions are on the heart estimate (the classifier is a heart-sound
classifier); for source == "lung" rows they are repeated for convenience.

Two summaries are printed and written (results/canonical_summary.csv):
    pooled   -- mean over the 36 rows, 95% t-CI over rows
    foldmean -- mean over folds of the per-fold mean, +/- std over folds
Both are legitimate; a table must say which it prints.

Usage:
    make canonical-rows
"""
import numpy as np
import pandas as pd
from scipy import stats

import heart_classifier
import heart_classifier_cnn
from baselines import raw_mixture_separate
from condition_b import build_condition_b_fold_basis
from heart_classifier import HEART_TYPE_TO_GROUP
from load_dataset import load_audio
from metrics import SOURCE_LABELS, evaluate_heart_lung
from report_utils import results_dir
from sdr_sweep import BASELINE_LABELS, cache_dir, estimate_path

METHOD_ORDER = ["Isolated (ground truth)", "Baseline 0 (raw mixture, no separation)", *BASELINE_LABELS]


def build_canonical_rows(n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    mix_df, hs_df = build_condition_b_fold_basis(n_folds=n_folds, seed=seed)
    clf_svm = heart_classifier.train_fold_classifiers(hs_df, n_folds=n_folds)
    clf_cnn = heart_classifier_cnn.train_fold_classifiers(hs_df, n_folds=n_folds)
    root = cache_dir()

    rows = []
    for _, row in mix_df.iterrows():
        mid, fold = row["Mixed Sound ID"], int(row["fold"])
        true = HEART_TYPE_TO_GROUP[row["Heart Sound Type"]]
        heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
        mixed, _ = load_audio(row["mixed_audio_path"], sr=None)

        estimates = {
            "Isolated (ground truth)": (heart_ref, lung_ref),
            "Baseline 0 (raw mixture, no separation)": raw_mixture_separate(mixed, sr),
        }
        for label in BASELINE_LABELS:
            he, _ = load_audio(str(estimate_path(root, label, mid, "heart")), sr=None)
            le, _ = load_audio(str(estimate_path(root, label, mid, "lung")), sr=None)
            estimates[label] = (he, le)

        for method, (he, le) in estimates.items():
            m = evaluate_heart_lung(heart_ref, lung_ref, he, le)
            pred_svm = heart_classifier.predict_one(clf_svm[fold], he, sr)
            pred_cnn = heart_classifier_cnn.predict_one(clf_cnn[fold], he, sr)
            for source in SOURCE_LABELS:
                rows.append({
                    "mixed_id": mid, "fold": fold, "leak_group": row["leak_group"], "method": method, "source": source,
                    "sdr": m[source]["sdr"], "sir": m[source]["sir"], "sar": m[source]["sar"],
                    "true_class": true, "pred_svm": pred_svm, "pred_cnn": pred_cnn,
                    "heart_type": row["Heart Sound Type"], "lung_type": row["Lung Sound Type"],
                })
        print(f"  {mid} done")
    return pd.DataFrame(rows)


def _ci95(x):
    x = np.asarray(x, dtype=float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Both conventions, side by side, for SDR/SIR/SAR and both accuracies."""
    out = []
    for (method, source), g in df.groupby(["method", "source"], sort=False):
        rec = {"method": method, "source": source, "n": len(g)}
        for metric in ("sdr", "sir", "sar"):
            rec[f"{metric}_pooled_mean"] = g[metric].mean()
            rec[f"{metric}_pooled_ci95"] = _ci95(g[metric])
            fm = g.groupby("fold")[metric].mean()
            rec[f"{metric}_foldmean"] = fm.mean()
            rec[f"{metric}_foldstd"] = fm.std(ddof=1)
        for arch in ("svm", "cnn"):
            correct = (g[f"pred_{arch}"] == g["true_class"])
            rec[f"acc_{arch}_pooled"] = correct.mean()
            rec[f"acc_{arch}_foldmean"] = correct.groupby(g["fold"]).mean().mean()
        out.append(rec)
    s = pd.DataFrame(out).set_index(["method", "source"])
    return s.reindex(pd.MultiIndex.from_product([METHOD_ORDER, SOURCE_LABELS], names=["method", "source"]))


if __name__ == "__main__":
    print("Scoring the cached separated waveforms and classifying their heart estimates with both architectures ...")
    df = build_canonical_rows()
    df.to_csv(results_dir() / "canonical_rows.csv", index=False)
    s = summarize(df)
    s.to_csv(results_dir() / "canonical_summary.csv")
    pd.set_option("display.width", 250)
    cols = ["n", "sdr_pooled_mean", "sdr_pooled_ci95", "sdr_foldmean", "sdr_foldstd", "acc_svm_pooled", "acc_cnn_pooled"]
    print(s[cols].round(3).to_string())
    print(f"\nWrote {results_dir() / 'canonical_rows.csv'} and canonical_summary.csv")
