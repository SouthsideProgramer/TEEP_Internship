"""
Waveform + mel-spectrogram grids, one representative recording per class.

For each Heart Sound Type (10 classes) and Lung Sound Type (6 classes),
picks one representative .wav file from HS.csv / LS.csv and renders:
    - a waveform grid (plots/waveforms_per_class.png)
    - a mel-spectrogram grid (plots/spectrograms_per_class.png)

Usage:
    python src/visualization/plot_per_class.py
"""
import sys
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_dataset import load_hs, load_ls


def _representative_rows(df, type_col):
    """One row per class: the first recording of that type in the CSV."""
    return df.sort_values(type_col).groupby(type_col, as_index=False).first()


def _grid_shape(n, ncols=4):
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    return nrows, ncols


def plot_waveform_grid(rows, title, ncols=4):
    nrows, ncols = _grid_shape(len(rows), ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    axs_flat = axs.flatten()

    for ax, (_, row) in zip(axs_flat, rows.iterrows()):
        y, sr = librosa.load(row["audio_path"], sr=None)
        librosa.display.waveshow(y, sr=sr, color="black", ax=ax)
        ax.set_title(row["class_label"], fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    for ax in axs_flat[len(rows):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_spectrogram_grid(rows, title, ncols=4):
    nrows, ncols = _grid_shape(len(rows), ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    axs_flat = axs.flatten()

    img = None
    for ax, (_, row) in zip(axs_flat, rows.iterrows()):
        y, sr = librosa.load(row["audio_path"], sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=sr / 2, ax=ax)
        ax.set_title(row["class_label"], fontsize=10)

    for ax in axs_flat[len(rows):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    if img is not None:
        fig.colorbar(img, ax=axs_flat[: len(rows)].tolist(), format="%+2.0f dB", label="dB")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def main():
    from report_utils import image_figure, report_shell, results_dir, section, write_report

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hs_reps = _representative_rows(load_hs(), "Heart Sound Type")
    hs_reps["class_label"] = hs_reps["Heart Sound Type"]

    ls_reps = _representative_rows(load_ls(), "Lung Sound Type")
    ls_reps["class_label"] = ls_reps["Lung Sound Type"]

    print(f"Rendering waveform + spectrogram grids for {len(hs_reps)} heart / {len(ls_reps)} lung classes...")

    fig = plot_waveform_grid(hs_reps, "Heart Sound Types — representative waveforms")
    fig.savefig(plots_dir / "waveforms_heart_per_class.png", dpi=120)
    plt.close(fig)

    fig = plot_waveform_grid(ls_reps, "Lung Sound Types — representative waveforms")
    fig.savefig(plots_dir / "waveforms_lung_per_class.png", dpi=120)
    plt.close(fig)

    fig = plot_spectrogram_grid(hs_reps, "Heart Sound Types — representative mel-spectrograms")
    fig.savefig(plots_dir / "spectrograms_heart_per_class.png", dpi=120)
    plt.close(fig)

    fig = plot_spectrogram_grid(ls_reps, "Lung Sound Types — representative mel-spectrograms")
    fig.savefig(plots_dir / "spectrograms_lung_per_class.png", dpi=120)
    plt.close(fig)

    body = "\n\n".join([
        section("Heart waveforms", f"{len(hs_reps)} classes", image_figure("plots/waveforms_heart_per_class.png", "heart waveforms per class")),
        section("Lung waveforms", f"{len(ls_reps)} classes", image_figure("plots/waveforms_lung_per_class.png", "lung waveforms per class")),
        section("Heart spectrograms", f"{len(hs_reps)} classes", image_figure("plots/spectrograms_heart_per_class.png", "heart spectrograms per class")),
        section("Lung spectrograms", f"{len(ls_reps)} classes", image_figure("plots/spectrograms_lung_per_class.png", "lung spectrograms per class")),
    ])

    html = report_shell(
        title="Per-Class Waveforms & Spectrograms",
        eyebrow="HLS-CMDS · plot_per_class.py",
        heading="One representative recording per class",
        dek="Waveform and mel-spectrogram grids, one representative recording per Heart Sound Type (10 classes) and Lung Sound Type (6 classes).",
        stat_tiles="",
        body=body,
        footer="<p><strong>Method.</strong> See <code>plot_per_class.py</code>.</p>",
    )
    report_path = write_report(results_dir() / "plot_per_class_report.html", html)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
