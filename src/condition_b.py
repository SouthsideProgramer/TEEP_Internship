"""
Condition B of PROTOCOL.md Sec. 5.3 -- evaluating the Condition A
classifier on real separated `heart_est` audio from S4-03's six separation
baselines (via S6-02's cached outputs, sdr_sweep.py), instead of on the
isolated ground truth. This is the controlled-conditions reproduction of
Yaqub et al.'s 89%->41% collapse (PROTOCOL.md Sec. 3): the same
isolated-vs-separated comparison, but with six separation methods instead
of their one bandpass filter, on this project's own dataset split and
classifier.

Weight-sharing decision (the open item PROTOCOL.md Sec. 5.3 flagged,
resolved here): **the same trained weights per fold, not retrained on
separated audio.** This is exactly Yaqub et al.'s own methodology -- their
Experiment 3 model (retrained on clean HLS-CMDS) *is* their Experiment 4
model (evaluated on bandpass-separated audio), with no retraining between
the two. Reproducing their finding "under controlled conditions" means
matching their causal claim -- does clean-trained inference degrade on
separated input -- not the different question of whether a classifier can
*adapt* to separation artifacts (PROTOCOL.md's noted follow-up, still
open). Same-weights also isolates the separation method as the only
variable between Condition A and Condition B: both conditions differ
solely in what audio is fed to inference at test time, nothing else about
the classifier changes.

Fold basis: Condition B needs the classifier's fold assignment and
sdr_sweep.py's fold assignment to agree on what "fold k" means, so that a
given fold's classifier never trained on a recording that leaks into that
same fold's held-out separated audio. sdr_sweep.py builds its folds from
the 36-native-additive-row Mix.csv subset (not the full 145-row Mix.csv
heart_classifier.assign_classifier_folds() defaults to for Condition A's
own standalone report) -- so this module rebuilds that same fold
assignment explicitly and passes it into assign_classifier_folds() via its
mix_df_with_folds parameter, rather than assuming the two already agree.

"Isolated" here means each of the 36 rows' own ground-truth heart
recording (Mix.csv's own heart_audio_path -- the same audio used
throughout this project to score every baseline's SDR/SIR/SAR), not a
lookup back into HS.csv -- correct and simpler, since not every Mix.csv
heart component has a standalone HS.csv counterpart at all (~57/89 distinct
mix-heart recordings don't, per split.py's own docstring).

Backend parameter (S7-06): every function below that trains or calls a
classifier takes a `backend` module (default `heart_classifier`,
Architecture 1). Pass `heart_classifier_cnn` (Architecture 2, S7-06) to
run the identical isolated/separated/no-separation comparison against the
second, architecturally distinct classifier -- see sdr_knee_point.py for
the cross-architecture knee-point comparison this feeds into. A backend
module must expose `train_fold_classifiers(hs_df, n_folds)` and
`predict_one(clf, y, sr)`; both heart_classifier.py and
heart_classifier_cnn.py satisfy this contract.

Usage:
    from condition_b import evaluate_condition_b
    import heart_classifier, heart_classifier_cnn

    predictions_df = evaluate_condition_b()  # Architecture 1 (default)
    predictions_df_cnn = evaluate_condition_b(backend=heart_classifier_cnn)  # Architecture 2
"""
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

import heart_classifier
from heart_classifier import CLASS_GROUPS, HEART_TYPE_TO_GROUP, assign_classifier_folds
from load_dataset import load_audio
from sdr_sweep import BASELINE_LABELS, cache_dir, estimate_path, native_additive_rows
from split import assign_folds

ISOLATED_LABEL = "Isolated (ground truth)"
NO_SEPARATION_LABEL = "No separation (raw mixture)"


def build_condition_b_fold_basis(
    n_folds: int = 5, seed: int = 0, mix_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The one fold assignment both the classifier and sdr_sweep.py's cached
    separation outputs must agree on. Returns (mix_df_with_folds, hs_df)
    -- mix_df_with_folds is exactly what sdr_sweep.build_sdr_sweep() used
    (same function, same substrate, same seed), and hs_df is the
    classifier's own training data folded consistently with it. Fold
    assignment is architecture-agnostic (it only depends on audio content
    and class labels, not on which backend later trains against it), so
    this has no backend parameter -- both architectures share one fold
    basis, which is exactly what makes their knee points comparable.

    mix_df: pass a pre-filtered subset of native_additive_rows() (e.g. a
    tiny slice for a test) instead of the full 36 rows.
    """
    mix_df = mix_df if mix_df is not None else native_additive_rows()
    mix_df = assign_folds(mix_df=mix_df, n_folds=n_folds, seed=seed)
    hs_df = assign_classifier_folds(n_folds=n_folds, seed=seed, mix_df_with_folds=mix_df)
    return mix_df, hs_df


def evaluate_condition_b(
    n_folds: int = 5, seed: int = 0, cache_root=None, baseline_labels=BASELINE_LABELS,
    mix_df: pd.DataFrame | None = None, backend=heart_classifier,
) -> pd.DataFrame:
    """
    For each native additive row (defaults to all 36): classify (a) the
    isolated ground-truth heart recording once, and (b) each baseline's
    real, undegraded separated heart_est (S6-02's cache) -- both through
    the *same* fold-k classifier (never trained on this row's own
    recording, if it happened to also sit in HS.csv).

    Returns predictions_df: one row per (baseline_or_isolated, mixed_id)
    with true/pred class_group, fold, and (for separated rows) the
    baseline's own measured heart SDR/SIR/SAR for that row.
    """
    mix_df, hs_df = build_condition_b_fold_basis(n_folds=n_folds, seed=seed, mix_df=mix_df)
    classifiers = backend.train_fold_classifiers(hs_df, n_folds=n_folds)
    root = cache_root if cache_root is not None else cache_dir()

    from metrics import evaluate_heart_lung

    rows = []
    for _, row in mix_df.iterrows():
        fold = row["fold"]
        clf = classifiers[fold]
        true = HEART_TYPE_TO_GROUP[row["Heart Sound Type"]]
        heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)

        pred = backend.predict_one(clf, heart_ref, sr)
        rows.append({
            "baseline": ISOLATED_LABEL, "mixed_id": row["Mixed Sound ID"], "fold": int(fold),
            "true": true, "pred": pred, "achieved_sdr": float("nan"),
        })

        for label in baseline_labels:
            est_path = estimate_path(root, label, row["Mixed Sound ID"], "heart")
            heart_est, est_sr = load_audio(str(est_path), sr=None)
            lung_est_path = estimate_path(root, label, row["Mixed Sound ID"], "lung")
            lung_est, _ = load_audio(str(lung_est_path), sr=None)

            pred = backend.predict_one(clf, heart_est, est_sr)
            sdr = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)["heart"]["sdr"]
            rows.append({
                "baseline": label, "mixed_id": row["Mixed Sound ID"], "fold": int(fold),
                "true": true, "pred": pred, "achieved_sdr": sdr,
            })

    return pd.DataFrame(rows)


def evaluate_no_separation(
    n_folds: int = 5, seed: int = 0, mix_df: pd.DataFrame | None = None, backend=heart_classifier,
) -> pd.DataFrame:
    """
    The third reference point PROTOCOL.md Sec. 4's C2 framing needs
    verbatim -- "the SDR below which separation actively hurts
    classification accuracy, *relative to not separating at all*":
    classify each row's raw, unseparated mixed recording directly
    (Baseline 0's own convention, `metrics.py`/`baselines.py` -- est=mixed
    for both sources) with the same fold-appropriate classifier weights
    Condition A/B use. Its accuracy is a single flat number (it doesn't
    depend on any separation method or target SDR), and its own SDR
    (`mixed` scored against the true heart source) is what anchors the
    "no separation at all" end of the x-axis.
    """
    from metrics import evaluate_heart_lung

    mix_df, hs_df = build_condition_b_fold_basis(n_folds=n_folds, seed=seed, mix_df=mix_df)
    classifiers = backend.train_fold_classifiers(hs_df, n_folds=n_folds)

    rows = []
    for _, row in mix_df.iterrows():
        fold = row["fold"]
        true = HEART_TYPE_TO_GROUP[row["Heart Sound Type"]]
        heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
        mixed, _ = load_audio(row["mixed_audio_path"], sr=None)

        pred = backend.predict_one(classifiers[fold], mixed, sr)
        sdr = evaluate_heart_lung(heart_ref, lung_ref, mixed, mixed)["heart"]["sdr"]
        rows.append({
            "baseline": NO_SEPARATION_LABEL, "mixed_id": row["Mixed Sound ID"], "fold": int(fold),
            "true": true, "pred": pred, "achieved_sdr": sdr,
        })

    return pd.DataFrame(rows)


def summarize_by_baseline(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, Macro-F1, n, and mean achieved SDR per baseline (+ Isolated)."""
    rows = []
    for baseline, group in predictions_df.groupby("baseline", sort=False):
        rows.append({
            "baseline": baseline,
            "n": len(group),
            "accuracy": accuracy_score(group["true"], group["pred"]),
            "macro_f1": f1_score(group["true"], group["pred"], average="macro", labels=CLASS_GROUPS, zero_division=0),
            "mean_achieved_sdr": group["achieved_sdr"].mean(),
        })
    # NO_SEPARATION_LABEL first (the SDR axis's low-end anchor), then Isolated
    # (the high end), then each separation baseline -- reindex only keeps
    # rows actually present in predictions_df, so this works whether the
    # caller passed just one condition or all of them concatenated.
    order = [NO_SEPARATION_LABEL, ISOLATED_LABEL, *BASELINE_LABELS]
    summary = pd.DataFrame(rows).set_index("baseline")
    return summary.reindex([label for label in order if label in summary.index])


def summarize_by_fold(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (baseline, fold) -- the paired unit for the significance test below."""
    rows = []
    for (baseline, fold), group in predictions_df.groupby(["baseline", "fold"], sort=False):
        rows.append({
            "baseline": baseline, "fold": fold, "n": len(group),
            "accuracy": accuracy_score(group["true"], group["pred"]),
        })
    return pd.DataFrame(rows)


def paired_delta_vs_isolated(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """
    PROTOCOL.md Sec. 5.4's primary result: accuracy_isolated - accuracy_separated,
    paired by fold, plus a paired t-test per baseline. n=5 folds -- per
    Sec. 6's own caveat, treat as indicative, not decisive.
    """
    isolated = fold_summary[fold_summary["baseline"] == ISOLATED_LABEL].set_index("fold")["accuracy"]

    rows = []
    for baseline in BASELINE_LABELS:
        separated = fold_summary[fold_summary["baseline"] == baseline].set_index("fold")["accuracy"]
        paired = pd.concat([isolated, separated], axis=1, keys=["isolated", "separated"]).dropna()
        delta = paired["isolated"] - paired["separated"]
        if len(paired) > 1 and delta.std(ddof=1) > 0:
            t_stat, p_value = stats.ttest_rel(paired["isolated"], paired["separated"])
        else:
            t_stat, p_value = float("nan"), float("nan")
        rows.append({
            "baseline": baseline,
            "n_folds": len(paired),
            "mean_delta_accuracy": float(delta.mean()) if len(delta) else float("nan"),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    return pd.DataFrame(rows).set_index("baseline")


if __name__ == "__main__":
    from heart_classifier import confusion_counts, confusion_recall_pct, plot_confusion_heatmap, top_confusions
    from report_utils import df_to_html, image_figure, report_shell, results_dir, section, stat_tile, write_report

    print("Evaluating Condition A classifier on isolated ground truth + all 6 separation baselines' real output...")
    print("(requires results/sdr_sweep_cache/ from `make sdr-sweep` -- S6-02)")
    predictions_df = evaluate_condition_b(n_folds=5, seed=0)
    predictions_df.to_csv(results_dir() / "condition_b_predictions.csv", index=False)

    baseline_summary = summarize_by_baseline(predictions_df)
    fold_summary = summarize_by_fold(predictions_df)
    paired = paired_delta_vs_isolated(fold_summary)

    isolated_acc = baseline_summary.loc[ISOLATED_LABEL, "accuracy"]
    worst_baseline = baseline_summary.drop(ISOLATED_LABEL)["accuracy"].idxmin()
    worst_acc = baseline_summary.loc[worst_baseline, "accuracy"]

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    confusion_sections = []
    for baseline in [ISOLATED_LABEL, *BASELINE_LABELS]:
        subset = predictions_df[predictions_df["baseline"] == baseline]
        counts = confusion_counts(subset)
        recall_pct = confusion_recall_pct(counts)
        confusions = top_confusions(counts)

        slug = baseline.split(" (")[0].replace(" ", "")  # "Baseline 1 (bandpass)" -> "Baseline1", "Isolated (...)" -> "Isolated"
        png_name = f"condition_b_confusion_{slug}.png"
        plot_confusion_heatmap(counts, recall_pct, plots_dir / png_name, title=f"{baseline}: confusion matrix")

        confusion_sections.append(section(
            f"{baseline}",
            f"accuracy {baseline_summary.loc[baseline, 'accuracy']:.1%}, macro-F1 {baseline_summary.loc[baseline, 'macro_f1']:.2f}",
            image_figure(f"plots/{png_name}", f"{baseline} confusion matrix")
            + df_to_html(confusions, index_label="true", float_fmt="{:.1%}"),
        ))

    stat_tiles = "\n".join([
        stat_tile("Isolated accuracy", f"{isolated_acc:.1%}", "ground truth (paired Condition A)"),
        stat_tile("Worst separated accuracy", f"{worst_acc:.1%}", worst_baseline),
        stat_tile("Biggest drop", f"{(isolated_acc - worst_acc):.1%}", f"isolated → {worst_baseline}", ok=False),
    ])

    body = "\n\n".join([
        section(
            "Accuracy / Macro-F1 per condition",
            "isolated ground truth vs. each separation baseline's real output, pooled across all 36 rows",
            df_to_html(baseline_summary, index_label="baseline", float_fmt="{:.3f}"),
        ),
        section(
            "Paired delta vs. isolated (PROTOCOL.md Sec. 5.4)",
            "accuracy_isolated - accuracy_separated, paired by fold, with a paired t-test (n=5 folds -- indicative, not decisive)",
            df_to_html(paired, index_label="baseline", float_fmt="{:.3f}"),
        ),
        section("Failure mode analysis: confusion matrices per condition", "", "\n\n".join(confusion_sections)),
    ])

    html = report_shell(
        title="Condition B: Classifying Separated Heart Sounds",
        eyebrow="HLS-CMDS · PROTOCOL.md Sec. 5.3/5.4, Condition B",
        heading="Reproducing Yaqub's 89%→41% collapse, under controlled conditions",
        dek=(
            "Same trained classifier weights per fold as Condition A, evaluated on isolated ground "
            "truth vs. each of the six separation baselines' real output (S6-02's cached separation "
            "outputs) for the same 36 native additive rows -- the same isolated-vs-separated "
            "comparison Yaqub et al. ran with one bandpass filter, run here with six methods."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>condition_b.py</code>'s module docstring for the "
            "weight-sharing decision and fold-basis alignment with <code>sdr_sweep.py</code> (S6-02); "
            "<code>PROTOCOL.md</code> Sec. 5.3/5.4 for the full writeup.</p>"
        ),
    )

    report_path = write_report(results_dir() / "condition_b_report.html", html)
    print(f"\nReport written to {report_path}")
