"""
S7-06: second classifier architecture for the C2 knee-point robustness
check -- no longer optional now that C2 (the knee point), not
classification accuracy on its own, is the paper's headline result, per
the Sprint 0 re-scope. The charter's original "one architecture only"
rule held when the classifier was a measuring instrument for a secondary
result (see heart_classifier.py's own docstring for that reasoning, still
valid on its own terms); once the knee point IS the paper, the first
reviewer question is whether it is a property of separation quality or of
the one architecture (heart_classifier.py's MFCC + RBF-SVM) that measured
it. This module answers that question with a second, architecturally
distinct classifier: log-mel spectrogram + a shallow CNN -- PROTOCOL.md
Sec. 5.3's own other named small-model option (its docstring names both
"log-mel spectrogram + a shallow CNN" and "MFCC + SVM"; Architecture 1
took the second, this takes the first).

Deliberately different in every respect that could matter for a
robustness check, not just "a second SVM": a 2D time-frequency
representation instead of pooled summary statistics, and a learned,
gradient-trained model instead of a kernel method. If both architectures'
knee points agree (sdr_knee_point.py's cross-architecture comparison), the
knee is a property of separation quality, not an artifact of one
classifier's particular decision boundary; if they disagree, that
disagreement is itself the more interesting finding (see the task brief
and PROTOCOL.md Sec. 5.3.3).

Still respects the same n=50/small-fold-training-set overfitting risk
heart_classifier.py's own docstring raised as the reason Architecture 1
avoided a CNN in the first place -- this module doesn't dismiss that
risk, it tests it directly: the CNN here is deliberately tiny (2 conv
blocks, global average pooling, no dense hidden layer -- a few thousand
parameters, not the hundreds of thousands a "real" spectrogram CNN would
use) specifically so a robustness *disagreement* traces to something more
interesting than "one of the two models was simply too big for the data."

Reuses everything architecture-agnostic from heart_classifier.py directly
rather than duplicating it: the class-grouping decision (CLASS_GROUPS,
HEART_TYPE_TO_GROUP, add_class_group), the leak-group fold assignment
(assign_classifier_folds), and the confusion-matrix analysis
(confusion_counts/confusion_recall_pct/top_confusions/
plot_confusion_heatmap, generic over any true/pred predictions_df). Only
feature extraction, the model, and the fit/predict loop are
architecture-specific and defined here.

Backend contract (what condition_b.py / sdr_accuracy_curve.py /
sdr_knee_point.py call generically via a `backend` module parameter, so
they run unchanged against either architecture):
    train_fold_classifiers(hs_df, n_folds) -> {fold: fitted classifier}
    predict_one(clf, y, sr) -> class_group label

Usage:
    from heart_classifier_cnn import cross_validate_classifier

    predictions_df, fold_summary, cv_summary = cross_validate_classifier()
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import accuracy_score, f1_score

from heart_classifier import CLASS_GROUPS, aggregate_ci95, assign_classifier_folds
from load_dataset import load_audio

N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256
RANDOM_SEED = 0
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Log-mel spectrogram, shape (N_MELS, T) -- a 2D time-frequency
    representation for the CNN, unlike Architecture 1's pooled MFCC
    summary statistics. Every HS.csv recording is exactly 60,000 samples
    (15s @ 4000 Hz, confirmed project-wide -- see README.md's audio-quality
    audit), so every recording produces the identical (N_MELS, T) shape
    with no padding/truncation needed.
    """
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.log(mel + 1e-6).astype(np.float32)


class _ShallowCNN(nn.Module):
    """2 conv blocks + global average pool + a single linear head --
    deliberately tiny (see module docstring) for n=50 recordings (as few
    as ~35-40/fold in training)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(Fnn.relu(self.bn1(self.conv1(x))))
        x = self.pool(Fnn.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


class CNNClassifier:
    """
    Thin (.fit/.predict) adapter around _ShallowCNN, matching the same
    duck-typed interface heart_classifier.py's sklearn Pipeline satisfies
    -- lets both architectures share condition_b.py's/
    sdr_accuracy_curve.py's downstream code unchanged via predict_one().
    """

    def __init__(self, class_labels=CLASS_GROUPS, seed: int = RANDOM_SEED, max_epochs: int = MAX_EPOCHS):
        self.class_labels = list(class_labels)
        self.seed = seed
        self.max_epochs = max_epochs
        self.model: _ShallowCNN | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNClassifier":
        torch.manual_seed(self.seed)
        label_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        y_idx = np.array([label_to_idx[label] for label in y])

        counts = np.bincount(y_idx, minlength=len(self.class_labels)).astype(np.float32)
        weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
        weights = weights / weights.sum() * len(self.class_labels)

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        self.model = _ShallowCNN(n_classes=len(self.class_labels))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.model.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call .fit() before .predict()"
        self.model.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            idx = self.model(X_t).argmax(dim=1).numpy()
        return np.array([self.class_labels[i] for i in idx])


def make_classifier(max_epochs: int = MAX_EPOCHS) -> CNNClassifier:
    return CNNClassifier(max_epochs=max_epochs)


def features_for(df: pd.DataFrame, path_col: str = "audio_path") -> np.ndarray:
    feats = [extract_features(*load_audio(path, sr=None)) for path in df[path_col]]
    return np.stack(feats)


def predict_one(clf: CNNClassifier, y: np.ndarray, sr: int) -> str:
    """Backend contract -- see this module's docstring."""
    return clf.predict(extract_features(y, sr)[np.newaxis, ...])[0]


def train_fold_classifiers(hs_df: pd.DataFrame, n_folds: int = 5, max_epochs: int = MAX_EPOCHS) -> dict[int, CNNClassifier]:
    """Backend contract -- see this module's docstring. Mirrors
    heart_classifier.train_fold_classifiers()'s per-fold training loop
    exactly, over this module's own features/model instead. max_epochs is
    overridable (matching convtasnet.py's own testability pattern) so
    tests can train a handful of epochs instead of the real MAX_EPOCHS."""
    X = features_for(hs_df)
    y = hs_df["class_group"].to_numpy()
    folds = hs_df["fold"].to_numpy()

    classifiers = {}
    for k in range(n_folds):
        train_mask = folds != k
        if not train_mask.any():
            continue
        clf = make_classifier(max_epochs=max_epochs)
        clf.fit(X[train_mask], y[train_mask])
        classifiers[k] = clf
    return classifiers


def cross_validate_classifier(
    n_folds: int = 5, seed: int = RANDOM_SEED, max_epochs: int = MAX_EPOCHS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Condition A for this architecture -- mirrors
    heart_classifier.cross_validate_classifier() exactly, over this
    module's own features/model instead. max_epochs is overridable for
    fast tests (see train_fold_classifiers)."""
    hs_df = assign_classifier_folds(n_folds=n_folds, seed=seed)
    classifiers = train_fold_classifiers(hs_df, n_folds=n_folds, max_epochs=max_epochs)

    pred_rows = []
    for k, clf in classifiers.items():
        test_df = hs_df[hs_df["fold"] == k]
        for _, row in test_df.iterrows():
            y, sr = load_audio(row["audio_path"], sr=None)
            pred = predict_one(clf, y, sr)
            pred_rows.append({"fold": k, "Heart Sound ID": row["Heart Sound ID"], "true": row["class_group"], "pred": pred})

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
            "macro_f1": f1_score(fold_df["true"], fold_df["pred"], average="macro", labels=CLASS_GROUPS, zero_division=0),
        })
    fold_summary = pd.DataFrame(fold_rows)
    cv_summary = aggregate_ci95(fold_summary)
    return predictions_df, fold_summary, cv_summary


if __name__ == "__main__":
    from heart_classifier import confusion_counts, confusion_recall_pct, plot_confusion_heatmap, top_confusions
    from report_utils import df_to_html, image_figure, report_shell, results_dir, section, stat_tile, write_report

    print("Architecture 2 (S7-06): log-mel spectrogram + shallow CNN, Condition A (isolated heart sounds)...")
    predictions_df, fold_summary, cv_summary = cross_validate_classifier(n_folds=5, seed=0)

    counts_df = confusion_counts(predictions_df)
    recall_pct_df = confusion_recall_pct(counts_df)
    confusions_df = top_confusions(counts_df)

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_heatmap(
        counts_df, recall_pct_df, plots_dir / "heart_classifier_cnn_confusion_matrix.png",
        title="Architecture 2 (log-mel + CNN): confusion matrix",
    )

    acc_row = cv_summary.loc["accuracy"]
    f1_row = cv_summary.loc["macro_f1"]

    stat_tiles = "\n".join([
        stat_tile("Accuracy (5-fold CV)", f"{acc_row['mean']:.1%}", f"± {acc_row['ci95']:.1%} (95% CI, n=5 folds)"),
        stat_tile("Macro-F1 (5-fold CV)", f"{f1_row['mean']:.2f}", f"± {f1_row['ci95']:.2f} (95% CI, n=5 folds)"),
        stat_tile("Architecture", "log-mel + CNN", "Architecture 2, S7-06"),
    ])

    body = "\n\n".join([
        section(
            "Per-fold accuracy / macro-F1",
            f"log-mel ({N_MELS} bins) + shallow CNN, {MAX_EPOCHS} epochs/fold, one fresh fit per fold",
            df_to_html(fold_summary.set_index("fold"), index_label="fold"),
        ),
        section(
            "Across-fold mean ± 95% CI",
            "n=5 folds -- indicative, not decisive (PROTOCOL.md Sec. 6)",
            df_to_html(cv_summary, index_label="metric"),
        ),
        section(
            "Failure mode analysis: confusion matrix",
            "pooled across all folds' held-out predictions",
            image_figure("plots/heart_classifier_cnn_confusion_matrix.png", "Architecture 2 confusion matrix")
            + df_to_html(confusions_df, index_label="true", float_fmt="{:.1%}"),
        ),
    ])

    html = report_shell(
        title="Architecture 2: Log-Mel CNN Classifier",
        eyebrow="HLS-CMDS · S7-06 · robustness check, Condition A",
        heading="Second architecture: log-mel spectrogram + shallow CNN",
        dek=(
            "The C2 knee-point robustness check's second classifier -- architecturally distinct from "
            "Architecture 1's MFCC+SVM (2D spectrogram vs. pooled summary statistics, gradient-trained "
            "CNN vs. kernel method). Same class-grouping decision, same leak-group folds. See "
            "sdr_knee_point.py for the cross-architecture knee-point comparison this feeds into."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>heart_classifier_cnn.py</code>'s module docstring "
            "for the architecture choice and why it's a genuine robustness check, not just a second "
            "SVM; PROTOCOL.md Sec. 5.3.3 for the full writeup.</p>"
        ),
    )

    report_path = write_report(results_dir() / "heart_classifier_cnn_report.html", html)
    print(f"\nReport written to {report_path}")
