"""
Mel-Spectrogram Plotter

Plots the Mel-spectrograms of multiple audio files using `librosa` and
`matplotlib`. See code_description.md for the feature list.

## Citation:
If you use this code or the associated dataset in your research, please cite the following paper:
- Y. Torabi, S. Shirani and J. P. Reilly,
"Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions,
doi: 10.1109/IEEEDATA.2025.3566012.

## Copyright:
© 2024 by Yasaman Torabi. All rights reserved.
"""

import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for load_dataset + report_utils
from load_dataset import load_hs, load_ls


def _example_rows():
    """
    Three illustrative recordings pulled straight from the live dataset
    (two heart conditions + one lung condition) instead of a fixed
    Examples/ folder, which the dataset no longer ships.
    """
    hs_df, ls_df = load_hs(), load_ls()
    heart_af = hs_df[hs_df["Heart Sound Type"] == "Atrial Fibrillation"].iloc[0]
    heart_s3 = hs_df[hs_df["Heart Sound Type"] == "S3"].iloc[0]
    lung_wheeze = ls_df[ls_df["Lung Sound Type"] == "Wheezing"].iloc[0]
    return [
        (heart_af["audio_path"], "Atrial Fibrillation"),
        (heart_s3["audio_path"], "S3"),
        (lung_wheeze["audio_path"], "Wheezing"),
    ]


def plot_mel_spectrograms(audio_files):
    """
    Plots mel-spectrograms for multiple audio files in a vertical figure with a shared color bar.

    Parameters:
    - audio_files: list of (audio_path, title) tuples

    Returns the created matplotlib Figure.
    """
    # Set up the figure with 3 rows and 1 column using gridspec
    fig = plt.figure(figsize=(8, 12))  # Adjust the figure size for vertical orientation
    gs = gridspec.GridSpec(len(audio_files) + 1, 1, height_ratios=[1] * len(audio_files) + [0.05])  # Last row for the colorbar

    axs = [plt.subplot(gs[i]) for i in range(len(audio_files))]

    # Plot spectrograms
    for i, (audio_path, title) in enumerate(audio_files):
        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Generate the mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2048)  # Limit frequency to 2048 Hz
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot the spectrogram in the corresponding subplot
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=2048, ax=axs[i])

        # Set the title and axis labels
        axs[i].set_title(title)
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Frequency (Hz)')

    # Add a color bar to the bottom of the last subplot
    cbar_ax = plt.subplot(gs[len(audio_files)])  # Create a new axis for the colorbar
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB', orientation='horizontal')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the color bar on the bottom
    return fig


if __name__ == "__main__":
    from report_utils import image_figure, report_shell, results_dir, section, write_report

    examples = _example_rows()
    titles = [title for _path, title in examples]

    print(f"Plotting mel-spectrograms for {len(examples)} example recordings...")
    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "spectrogram_examples.png"

    fig = plot_mel_spectrograms(examples)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    html = report_shell(
        title="Mel-Spectrogram Examples",
        eyebrow="HLS-CMDS · audio_spectrogram.py",
        heading="Example heart/lung mel-spectrograms",
        dek="Mel-spectrograms for the same three illustrative recordings as audio_plotter.py's waveform report.",
        stat_tiles="",
        body=section(
            "Mel-spectrograms",
            ", ".join(titles),
            image_figure(f"plots/{output_path.name}", ", ".join(titles)),
        ),
        footer="<p><strong>Method.</strong> See <code>audio_spectrogram.py</code>'s <code>plot_mel_spectrograms()</code>.</p>",
    )
    report_path = write_report(results_dir() / "audio_spectrogram_report.html", html)
    print(f"Report written to {report_path}")
