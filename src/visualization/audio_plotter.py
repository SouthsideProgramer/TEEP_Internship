"""
Audio Waveform Plotter

Visualizes and plots time-domain waveforms of audio files and plays them
back using IPython. See code_description.md for the feature list.

## Citation:
If you use this code or the associated dataset in your research, please cite the following paper:
- Y. Torabi, S. Shirani and J. P. Reilly,
"Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions,
doi: 10.1109/IEEEDATA.2025.3566012.

## Copyright:
© 2024 by Yasaman Torabi. All rights reserved.
"""

import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import IPython.display as ipd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

def plot_audio_waveform(audio_path, ax, title):
    """
    Plots the waveform of an audio file.

    Parameters:
    - audio_path: str, path to the audio file
    - ax: matplotlib axis, axis to plot on
    - title: str, title of the plot
    """
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    y, sr = librosa.load(audio_path)

    librosa.display.waveshow(y, sr=sr, color='black', ax=ax)
    ax.set_title(title)
    ax.set_ylim([-0.05, 0.05])
    ax.set_xlim([0, len(y) / sr])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

def plot_combined_waveforms(audio_paths, titles, output_file='combined_plots.png'):
    """
    Plots waveforms for multiple audio files and saves the figure as an image.

    Parameters:
    - audio_paths: list of str, paths to the audio files
    - titles: list of str, titles for each plot
    - output_file: str, name of the output file for the figure
    """
    num_plots = len(audio_paths)
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    for i in range(num_plots):
        plot_audio_waveform(audio_paths[i], axs[i], titles[i])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def play_audio(audio_path):
    """
    Plays the audio using IPython's Audio player.

    Parameters:
    - audio_path: str, path to the audio file
    """
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    y, sr = librosa.load(audio_path)
    
    return ipd.Audio(y, rate=sr)


if __name__ == "__main__":
    from report_utils import image_figure, report_shell, results_dir, section, write_report

    examples = _example_rows()
    audio_files = [path for path, _title in examples]
    titles = [title for _path, title in examples]

    print(f"Plotting waveforms for {len(examples)} example recordings...")
    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "combined_plots.png"
    plot_combined_waveforms(audio_files, titles, output_file=str(output_path))
    plt.close("all")

    html = report_shell(
        title="Waveform Examples",
        eyebrow="HLS-CMDS · audio_plotter.py",
        heading="Example heart/lung waveforms",
        dek=(
            "Time-domain waveforms for three illustrative recordings (two heart "
            "conditions, one lung condition), pulled live from HS.csv/LS.csv."
        ),
        stat_tiles="",
        body=section(
            "Combined waveforms",
            ", ".join(titles),
            image_figure(f"plots/{output_path.name}", ", ".join(titles)),
        ),
        footer="<p><strong>Method.</strong> See <code>audio_plotter.py</code>'s <code>plot_combined_waveforms()</code>.</p>",
    )
    report_path = write_report(results_dir() / "audio_plotter_report.html", html)
    print(f"Report written to {report_path}")
