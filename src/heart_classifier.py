"""
Condition A of PROTOCOL.md Sec. 5.3 -- the isolated/oracle-upper-bound
classification accuracy on HS.csv's own clean heart sound recordings,
evaluated on the same leak-group folds every separation baseline already
uses (assign_hs_folds() in split.py, new alongside this module). This is
the first *accuracy* number in the project (every prior result is SDR/SIR/
SAR, in dB) and the Condition A anchor the rest of the charter's C2 sweep
gets compared against, so it is reported with a per-fold table and a CI,
not just a single number.

Class-grouping decision (blocking, made before any classifier code, per
the task brief): full 10-class Heart Sound Type classification is out per
the charter. Under the leak-group 5-fold split, S4 (n=2 recordings total)
cannot appear in every fold's training set at all -- if both of its two
recordings land in the same held-out fold's leak group, three of five
folds train with zero S4 examples -- and AV Block/Tachycardia (n=3 each)
are barely better off. Two options were considered: binary Normal/
Abnormal, and a grouped multi-class scheme. Grouped classes were chosen,
using the exact four groups Yaqub et al. [spectrotemporal] use for this
same dataset (confirmed directly from the primary source, Secs. 5.1-5.2 --
Tables 6-9's class list: 'Normal', 'Murmur', 'Extra Sound', 'Rhythm
Disorder' -- the same class list their own 89%->41% Experiment 3/4
collapse, the motivating result this project exists to explain (PROTOCOL.md
Sec. 3), is measured on). Mapping HLS-CMDS's 10 Heart Sound Types onto that
scheme is unambiguous from clinical terminology and the type names
themselves: Normal -> Normal (n=9); the four murmur types (Mid/Late
Systolic, Late Diastolic, Early Systolic) -> Murmur (n=24); S3/S4 (extra
heart sounds beyond S1/S2) -> Extra Sound (n=7); Atrial Fibrillation/
Tachycardia/AV Block (rhythm and conduction disorders) -> Rhythm Disorder
(n=10). This solves the small-class problem (the smallest group, Extra
Sound at n=7, is comfortably large enough to appear in every fold's
training set, unlike 10-class's n=2/3 classes) and, unlike collapsing to
binary Normal/Abnormal, keeps Condition A's number directly comparable to
the specific accuracy figures (86-89%) this project's Yaqub-collapse
motivation is anchored to -- binary would solve the imbalance problem too,
but at the cost of measuring a different question than the one the
charter's motivating result is actually about.

Architecture (one architecture, deliberately -- per the task brief and
PROTOCOL.md Sec. 5.3's own recommendation, this is a measuring instrument,
not a contribution; see BACKLOG.md's S7-06 note for the planned
second-architecture robustness check): 13 MFCCs, mean+std pooled over time
into a 26-dim feature vector, fed to an RBF-kernel SVM with class-balanced
weights. Deliberately classical rather than a deep spectrogram model --
n=50 recordings (as few as ~35-40/fold in training) is far too little data
to train a CNN without the result being dominated by overfitting noise
rather than signal, and MFCC+SVM is PROTOCOL.md's own explicitly stated
fallback for exactly this small-data regime. No hyperparameter search was
run (first working configuration, same disclosed-scope pattern as
Baseline 6's first-neural-baseline note) -- C=1.0, gamma='scale' are
sklearn's own defaults, not tuned on this data.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_dataset import load_audio, load_hs
from split import assign_folds, assign_hs_folds

N_MFCC = 13
N_FFT = 512  # ~128 ms at this dataset's native 4000 Hz
HOP_LENGTH = 256  # ~64 ms hop
RANDOM_SEED = 0

# See this module's docstring for the citation and mapping rationale.
HEART_TYPE_TO_GROUP = {
    "Normal": "Normal",
    "Mid Systolic Murmur": "Murmur",
    "Late Diastolic Murmur": "Murmur",
    "Early Systolic Murmur": "Murmur",
    "Late Systolic Murmur": "Murmur",
    "S3": "Extra Sound",
    "S4": "Extra Sound",
    "Atrial Fibrillation": "Rhythm Disorder",
    "Tachycardia": "Rhythm Disorder",
    "AV Block": "Rhythm Disorder",
}
CLASS_GROUPS = ["Normal", "Murmur", "Extra Sound", "Rhythm Disorder"]


def add_class_group(hs_df: pd.DataFrame) -> pd.DataFrame:
    """HS.csv rows plus a 'class_group' column (see module docstring for the mapping)."""
    hs_df = hs_df.copy()
    hs_df["class_group"] = hs_df["Heart Sound Type"].map(HEART_TYPE_TO_GROUP)
    unmapped = hs_df["class_group"].isna()
    if unmapped.any():
        raise ValueError(f"Unmapped Heart Sound Type(s): {sorted(hs_df.loc[unmapped, 'Heart Sound Type'].unique())}")
    return hs_df


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """13 MFCCs -> per-coefficient mean and std over time -> 26-dim feature vector."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def make_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def features_for(df: pd.DataFrame, path_col: str = "audio_path") -> np.ndarray:
    feats = [extract_features(*load_audio(path, sr=None)) for path in df[path_col]]
    return np.stack(feats)


def predict_one(clf, y: np.ndarray, sr: int) -> str:
    """Classify one raw waveform. Part of this module's "backend" contract
    (alongside train_fold_classifiers) that condition_b.py/
    sdr_accuracy_curve.py/sdr_knee_point.py call generically -- see
    heart_classifier_cnn.py (S7-06) for the second architecture that
    implements the same two functions over a different feature/model
    pair, so those downstream modules run unchanged against either."""
    return clf.predict(extract_features(y, sr).reshape(1, -1))[0]


def assign_classifier_folds(
    n_folds: int = 5, seed: int = RANDOM_SEED, mix_df_with_folds: pd.DataFrame | None = None
) -> pd.DataFrame:
    """HS.csv + class_group + fold, on the same leak-group folds every separation
    baseline uses, stratified by class_group for the recordings never reused
    in any mixture (so no fold is starved of a minority group).

    mix_df_with_folds: pass a pre-computed, already-fold-assigned Mix.csv
    DataFrame (assign_folds()'s output) to align this classifier's own fold
    numbering with a *specific* substrate's fold assignment -- e.g.
    sdr_sweep.py builds its folds from the 36-native-additive-row subset,
    not the full 145-row Mix.csv this function defaults to, and Condition B
    (evaluating this classifier's fold-k weights on that substrate's
    separated audio) requires both to agree on what fold k means. Defaults
    to None (full Mix.csv), Condition A's own standalone fold basis.
    """
    hs_df = add_class_group(load_hs())
    mix_df = mix_df_with_folds if mix_df_with_folds is not None else assign_folds(n_folds=n_folds, seed=seed)
    return assign_hs_folds(hs_df, mix_df, n_folds=n_folds, seed=seed, stratify_col="class_group")


def train_fold_classifiers(hs_df: pd.DataFrame, n_folds: int = 5) -> dict[int, Pipeline]:
    """
    hs_df must already carry 'class_group' and 'fold' columns
    (assign_classifier_folds()). Trains one classifier per fold on that
    fold's training split (excludes fold k's own recordings), keyed by
    fold. Shared by cross_validate_classifier (Condition A: evaluate fold
    k's weights on fold k's own isolated holdout) and Condition B
    (evaluate the *same* fold k weights on fold k's held-out separated
    audio instead -- PROTOCOL.md Sec. 5.3's weight-sharing decision).
    """
    X = features_for(hs_df)
    y = hs_df["class_group"].to_numpy()
    folds = hs_df["fold"].to_numpy()

    classifiers = {}
    for k in range(n_folds):
        train_mask = folds != k
        if not train_mask.any():
            continue
        clf = make_classifier()
        clf.fit(X[train_mask], y[train_mask])
        classifiers[k] = clf
    return classifiers


def cross_validate_classifier(
    n_folds: int = 5, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leak-group 5-fold CV of the Condition A classifier (isolated HS.csv audio
    only, PROTOCOL.md Sec. 5.3).

    Returns:
        predictions_df: one row per (fold, Heart Sound ID) with true/predicted class_group.
        fold_summary: accuracy + macro-F1 per fold.
        cv_summary: mean +/- 95% CI across folds (n=5 folds -- indicative, not
                    decisive, per PROTOCOL.md Sec. 6's own caveat on small fold counts).
    """
    hs_df = assign_classifier_folds(n_folds=n_folds, seed=seed)
    classifiers = train_fold_classifiers(hs_df, n_folds=n_folds)

    X = features_for(hs_df)
    folds = hs_df["fold"].to_numpy()

    pred_rows = []
    for k, clf in classifiers.items():
        test_mask = folds == k
        y_pred = clf.predict(X[test_mask])
        test_df = hs_df[test_mask]
        for hid, true, pred in zip(test_df["Heart Sound ID"], test_df["class_group"], y_pred):
            pred_rows.append({"fold": k, "Heart Sound ID": hid, "true": true, "pred": pred})

    predictions_df = pd.DataFrame(pred_rows)

    fold_rows = []
    for k in range(n_folds):
        fold_df = predictions_df[predictions_df["fold"] == k]
        if len(fold_df) == 0:
            continue
        fold_rows.append({
            "fold": k,
            "n": len(fold_df),
            "accuracy": accuracy_score(fold_df["true"], fold_df["pred"]),
            "macro_f1": f1_score(
                fold_df["true"], fold_df["pred"], average="macro", labels=CLASS_GROUPS, zero_division=0
            ),
        })
    fold_summary = pd.DataFrame(fold_rows)
    cv_summary = aggregate_ci95(fold_summary)
    return predictions_df, fold_summary, cv_summary


def confusion_counts(predictions_df: pd.DataFrame, labels: list[str] = CLASS_GROUPS) -> pd.DataFrame:
    """
    Raw confusion matrix, pooled across all folds' held-out predictions
    (every recording is predicted exactly once, in its own held-out fold --
    see cross_validate_classifier -- so pooling across folds is just pooling
    every recording's single prediction, not double-counting). Rows are the
    true class_group, columns are the predicted class_group.
    """
    cm = confusion_matrix(predictions_df["true"], predictions_df["pred"], labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred"))


def confusion_recall_pct(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalized confusion matrix (each true class's row sums to 100%) --
    the same convention Yaqub et al.'s own HLS-CMDS confusion matrices use
    (Figs. 13-16), so this one reads directly comparably to theirs.
    """
    row_totals = counts_df.sum(axis=1)
    return counts_df.div(row_totals, axis=0) * 100


def top_confusions(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each true class_group, the single most common *wrong* prediction
    (excludes the correct diagonal cell) -- the specific failure mode behind
    a low recall number, not just the aggregate rate.
    """
    rows = []
    for true_label in counts_df.index:
        n_true = int(counts_df.loc[true_label].sum())
        n_correct = int(counts_df.loc[true_label, true_label])
        wrong = counts_df.loc[true_label].drop(true_label)
        worst = wrong.idxmax() if wrong.sum() > 0 else "—"
        worst_count = int(wrong[worst]) if wrong.sum() > 0 else 0
        rows.append({
            "true": true_label,
            "n": n_true,
            "recall": n_correct / n_true if n_true else float("nan"),
            "most_confused_with": worst,
            "confusion_count": worst_count,
            "confusion_pct": worst_count / n_true if n_true else float("nan"),
        })
    return pd.DataFrame(rows).set_index("true")


def plot_confusion_heatmap(
    counts_df: pd.DataFrame, recall_pct_df: pd.DataFrame, output_path,
    title: str = "Condition A confusion matrix (pooled, 5-fold CV)",
) -> None:
    """Row-normalized-color heatmap, each cell annotated with raw count + row %."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(counts_df.index)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(recall_pct_df.to_numpy(), cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(counts_df.iloc[i, j])
            pct = float(recall_pct_df.iloc[i, j])
            ax.text(j, i, f"{count}\n{pct:.0f}%", ha="center", va="center",
                     color="white" if pct > 55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="row-normalized % (recall)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def aggregate_ci95(fold_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("accuracy", "macro_f1"):
        vals = fold_summary[metric].to_numpy(dtype=float)
        n = len(vals)
        mean = float(vals.mean()) if n else float("nan")
        ci95 = 1.96 * float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"metric": metric, "mean": mean, "ci95": ci95, "n_folds": n})
    return pd.DataFrame(rows).set_index("metric")


if __name__ == "__main__":
    from report_utils import df_to_html, image_figure, report_shell, results_dir, section, stat_tile, write_report

    print("Assigning leak-group folds to HS.csv (Condition A: isolated heart sounds)...")
    hs_df = assign_classifier_folds(n_folds=5, seed=0)
    group_counts = hs_df["class_group"].value_counts().reindex(CLASS_GROUPS)
    print("Class group sizes:", dict(group_counts))

    print("Extracting MFCC features and running 5-fold CV (MFCC + RBF-SVM)...")
    predictions_df, fold_summary, cv_summary = cross_validate_classifier(n_folds=5, seed=0)

    per_class_rows = []
    for group in CLASS_GROUPS:
        true_mask = predictions_df["true"] == group
        n = int(true_mask.sum())
        recall = float((predictions_df.loc[true_mask, "pred"] == group).mean()) if n else float("nan")
        per_class_rows.append({"class_group": group, "n": n, "recall": recall})
    per_class_df = pd.DataFrame(per_class_rows).set_index("class_group")

    print("Computing confusion matrix and failure-mode breakdown...")
    counts_df = confusion_counts(predictions_df)
    recall_pct_df = confusion_recall_pct(counts_df)
    confusions_df = top_confusions(counts_df)

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_heatmap(counts_df, recall_pct_df, plots_dir / "heart_classifier_confusion_matrix.png")

    fold_counts = hs_df.groupby(["fold", "class_group"]).size().unstack(fill_value=0).reindex(columns=CLASS_GROUPS)

    acc_row = cv_summary.loc["accuracy"]
    f1_row = cv_summary.loc["macro_f1"]

    stat_tiles = "\n".join([
        stat_tile("Accuracy (5-fold CV)", f"{acc_row['mean']:.1%}", f"± {acc_row['ci95']:.1%} (95% CI, n=5 folds)"),
        stat_tile("Macro-F1 (5-fold CV)", f"{f1_row['mean']:.2f}", f"± {f1_row['ci95']:.2f} (95% CI, n=5 folds)"),
        stat_tile("Recordings", str(len(hs_df)), f"{len(CLASS_GROUPS)} class groups"),
    ])

    body = "\n\n".join([
        section(
            "Class-group sizes per fold",
            "leak-group folds (assign_hs_folds), stratified for unassigned recordings",
            df_to_html(fold_counts, index_label="fold"),
        ),
        section(
            "Per-fold accuracy / macro-F1",
            "MFCC (13 coef, mean+std) + RBF-SVM, one fresh fit per fold",
            df_to_html(fold_summary.set_index("fold"), index_label="fold"),
        ),
        section(
            "Across-fold mean ± 95% CI",
            "n=5 folds -- indicative, not decisive (PROTOCOL.md Sec. 6)",
            df_to_html(cv_summary, index_label="metric"),
        ),
        section(
            "Per-class-group recall",
            "pooled across all folds' held-out predictions",
            df_to_html(per_class_df, index_label="class_group"),
        ),
        section(
            "Failure mode analysis: confusion matrix",
            "pooled across all folds' held-out predictions -- each recording predicted exactly once",
            image_figure(
                "plots/heart_classifier_confusion_matrix.png",
                "rows = true class_group, columns = predicted; cell = count / row-normalized %",
            )
            + df_to_html(counts_df, index_label="true \\ pred", float_fmt="{:.0f}")
            + df_to_html(recall_pct_df, index_label="true \\ pred (row %)", float_fmt="{:.1f}"),
        ),
        section(
            "Failure mode analysis: dominant confusion per class",
            "most common wrong prediction for each true class_group, excluding the correct diagonal cell",
            df_to_html(confusions_df, index_label="true", float_fmt="{:.1%}"),
        ),
    ])

    html = report_shell(
        title="Condition A: Isolated Heart Sound Classifier",
        eyebrow="HLS-CMDS · PROTOCOL.md Sec. 5.3, Condition A",
        heading="Classifying clean heart sounds: the isolated-audio anchor",
        dek=(
            "First accuracy number in the project (every other result so far is SDR/SIR/SAR, "
            "in dB). MFCC + RBF-SVM, 4 grouped classes (Normal/Murmur/Extra Sound/Rhythm "
            "Disorder -- Yaqub et al.'s own HLS-CMDS scheme, see heart_classifier.py's "
            "docstring for the class-grouping decision), evaluated on the same leak-group "
            "5-fold split every separation baseline already uses."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>heart_classifier.py</code>'s module "
            "docstring for the class-grouping decision and architecture choice, and "
            "<code>split.py</code>'s <code>assign_hs_folds()</code> for how HS.csv recordings "
            "inherit the same leak-group fold assignment as every separation baseline.</p>"
        ),
    )

    report_path = write_report(results_dir() / "heart_classifier_report.html", html)
    print(f"\nReport written to {report_path}")
