"""
Sound Types Donut Chart Visualization

Generates a multi-layered donut chart of heart/lung/mixed sound counts
using `matplotlib`. See code_description.md for the feature list.

## Citation:
If you use this code or the associated dataset in your research, please cite the following paper:
- Y. Torabi, S. Shirani and J. P. Reilly, 
"Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS),"in IEEE Data Descriptions,
doi: 10.1109/IEEEDATA.2025.3566012.

## Copyright:
© 2024 by Yasaman Torabi. All rights reserved.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_sound_type_donut_chart():
    """
    Builds the 3-layer donut chart (mixed/heart/lung sound type counts) and returns the created Figure.
    """
    heart_values_updated = [9, 6, 7, 5, 4, 2, 6, 5, 3, 3]
    lung_values_updated = [12, 7, 5, 8, 9, 9]
    mix_values_updated = [9, 10, 10, 16, 13, 12, 8, 11, 12, 9, 18, 16, 13, 21, 24, 18]

    heart_labels_updated = ['Normal Heart', 'Late Diastolic Murmur', 'Mid Systolic Murmur',
                            'Late Systolic Murmur', 'Atrial Fibrillation', 'Fourth Heart Sound',
                            'Early Systolic Murmur', 'Third Heart Sound', 'Tachycardia',
                            'Atrioventricular Block']
    lung_labels_updated = ['Normal Lung', 'Wheezing', 'Crackles', 'Rhonchi',
                           'Pleural Rub', 'Gurgling']
    mix_labels_updated = heart_labels_updated + lung_labels_updated

    color_mapping = {
        'Normal Heart': '#1f77b4', 'Late Diastolic Murmur': '#ff7f0e', 'Mid Systolic Murmur': '#2ca02c',
        'Late Systolic Murmur': '#d62728', 'Atrial Fibrillation': '#9467bd', 'Fourth Heart Sound': '#8c564b',
        'Early Systolic Murmur': '#e377c2', 'Third Heart Sound': '#7f7f7f', 'Tachycardia': '#bcbd22',
        'Atrioventricular Block': '#17becf', 'Normal Lung': '#9edae5', 'Wheezing': '#ff9896',
        'Crackles': '#98df8a', 'Rhonchi': '#c5b0d5', 'Pleural Rub': '#ffbb78', 'Gurgling': '#c49c94'
    }

    heart_colors = [color_mapping[label] for label in heart_labels_updated]
    lung_colors = [color_mapping[label] for label in lung_labels_updated]
    mix_colors = [color_mapping[label] for label in mix_labels_updated]

    legend_labels_with_counts = [
        f'Normal Heart (18)', f'Late Diastolic Murmur (16)', f'Mid Systolic Murmur (17)',
        f'Late Systolic Murmur (21)', f'Atrial Fibrillation (17)', f'Fourth Heart Sound (14)',
        f'Early Systolic Murmur (14)', f'Third Heart Sound (16)', f'Tachycardia (15)',
        f'Atrioventricular Block (12)', f'Normal Lung (30)', f'Wheezing (23)',
        f'Crackles (18)', f'Rhonchi (29)', f'Pleural Rub (33)', f'Gurgling (27)'
    ]

    fig, ax = plt.subplots(figsize=(10, 10))

    wedges, texts = ax.pie(mix_values_updated, radius=1, colors=mix_colors, startangle=90, pctdistance=0.85)
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle)) * 0.85
        y = np.sin(np.deg2rad(angle)) * 0.85
        ax.text(x, y, str(mix_values_updated[i]), ha='center', va='center', fontsize=10)

    wedges, texts = ax.pie(heart_values_updated, radius=0.75, colors=heart_colors, startangle=90, pctdistance=0.75)
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle)) * 0.6
        y = np.sin(np.deg2rad(angle)) * 0.6
        ax.text(x, y, str(heart_values_updated[i]), ha='center', va='center', fontsize=10)

    wedges, texts = ax.pie(lung_values_updated, radius=0.5, colors=lung_colors, startangle=90, pctdistance=0.65)
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle)) * 0.45
        y = np.sin(np.deg2rad(angle)) * 0.45
        ax.text(x, y, str(lung_values_updated[i]), ha='center', va='center', fontsize=10)

    centre_circle = plt.Circle((0, 0), 0.3, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.title('Sound Types in the Dataset', fontsize=16, fontweight='bold', y=0.95)

    plt.legend(labels=legend_labels_with_counts, loc='center left', bbox_to_anchor=(1, 0.5), title="Sound Types")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from report_utils import image_figure, report_shell, results_dir, section, write_report

    print("Building sound-type donut chart...")
    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "donut_chart.png"

    fig = plot_sound_type_donut_chart()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    html = report_shell(
        title="Sound Types Donut Chart",
        eyebrow="HLS-CMDS · donut_chart.py",
        heading="Heart / lung / mixed sound type counts",
        dek="Three-layer donut chart of sound-type counts across the dataset (mixed outer, heart middle, lung inner).",
        stat_tiles="",
        body=section("Donut chart", "heart / lung / mixed", image_figure(f"plots/{output_path.name}", "sound type donut chart")),
        footer="<p><strong>Method.</strong> See <code>donut_chart.py</code>'s <code>plot_sound_type_donut_chart()</code>.</p>",
    )
    report_path = write_report(results_dir() / "donut_chart_report.html", html)
    print(f"Report written to {report_path}")
