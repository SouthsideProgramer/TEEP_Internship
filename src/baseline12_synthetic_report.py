"""
Baselines 1 (bandpass, S3-01) and 2 (supervised NMF, S3-02) re-run on the
synthetic evaluation set (S1-09/S1-10/S1-13), with the 36 native additive
rows alongside as a secondary column -- not a rebuild, both implementations
are unchanged; only the evaluation substrate is new. The synthetic-vs-native
comparison is itself audit evidence for whether the synthetic substrate
tracks the dataset's own real mixtures (see first_sdr_table.py's broader
version of the same comparison across all 5 baselines).

Usage:
    python baseline12_synthetic_report.py
"""
import numpy as np
import pandas as pd

from baselines import fit_bandpass_baseline, make_supervised_nmf_baseline
from metrics import SOURCE_LABELS
from synthetic_mix import SYNTHETIC_MIX_N_FOLDS, SNR_SWEEP_DB, build_synthetic_set, evaluate_synthetic

BASELINE_SPECS = [
    ("Baseline 1 (bandpass, S3-01)", fit_bandpass_baseline),
    ("Baseline 2 (supervised NMF, S3-02)", make_supervised_nmf_baseline(seed=0)),
]


def _mean_ci95(values) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(values))
    ci = 1.96 * float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, ci, n


def _fmt_mean_ci(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return "—"
    if np.isnan(ci):
        return f"{mean:.2f} dB (n=1, no CI)"
    return f"{mean:.2f} ± {ci:.2f} dB"


if __name__ == "__main__":
    from eval_harness import cross_validate
    from load_dataset import load_mix, verify_additive_triplets
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Loading native additive subset...")
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid_mix_df = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(f"  {len(valid_mix_df)} / {len(mix_df)} native rows are additive")

    print("Building synthetic mixing set (S1-09/S1-10/S1-13)...")
    synthetic_df = build_synthetic_set(n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)
    print(f"  {len(synthetic_df)} synthetic rows")

    rows = []
    for label, fit_fn in BASELINE_SPECS:
        print(f"Running {label}...")
        print(f"  synthetic ({len(synthetic_df)} rows)...")
        synth_results, _fold, _cv = evaluate_synthetic(fit_fn, synthetic_df, n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)

        print(f"  native additive ({len(valid_mix_df)} rows)...")
        native_results, _fold, _cv = cross_validate(fit_fn, n_folds=5, seed=0, mix_df=valid_mix_df)

        for source in SOURCE_LABELS:
            synth_src = synth_results[synth_results["source"] == source]
            native_src = native_results[native_results["source"] == source]
            row = {"baseline": label, "source": source, "synthetic_n": len(synth_src), "native_n": len(native_src)}
            for metric in ("sdr", "sir", "sar"):
                m, ci, _n = _mean_ci95(synth_src[metric].values)
                row[f"synthetic_{metric}"] = _fmt_mean_ci(m, ci)
                row[f"_synthetic_{metric}_mean"] = m
            for metric in ("sdr", "sir", "sar"):
                m, ci, _n = _mean_ci95(native_src[metric].values)
                row[f"native_{metric}"] = _fmt_mean_ci(m, ci)
                row[f"_native_{metric}_mean"] = m
            rows.append(row)

    table_df = pd.DataFrame(rows)
    display_cols = ["synthetic_n", "synthetic_sdr", "synthetic_sir", "synthetic_sar", "native_n", "native_sdr", "native_sir", "native_sar"]
    display_df = table_df.set_index(["baseline", "source"])[display_cols]

    interp_lines = [
        "<p>Synthetic (primary) vs. native-additive (secondary) SDR, per baseline/source "
        "(synthetic mean minus native mean, dB — the gap is itself audit evidence for how "
        "the synthetic set's SNR sweep relates to the native rows' near-noiseless construction, "
        "not a discrepancy to explain away):</p><ul>",
    ]
    for _, row in table_df.iterrows():
        gap = row["_synthetic_sdr_mean"] - row["_native_sdr_mean"]
        interp_lines.append(f"<li><strong>{row['baseline']}</strong> ({row['source']}): {gap:+.2f} dB</li>")
    interp_lines.append("</ul>")

    stat_tiles = "\n".join([
        stat_tile("Baselines", "2", "S3-01 bandpass, S3-02 supervised NMF"),
        stat_tile("Synthetic n", str(len(synthetic_df)), f"{SYNTHETIC_MIX_N_FOLDS} folds x {len(SNR_SWEEP_DB)} SNR levels"),
        stat_tile("Native additive n", str(len(valid_mix_df)), f"of {len(mix_df)} Mix.csv rows"),
    ])

    body = "\n\n".join([
        section(
            "SDR / SIR / SAR: synthetic (primary) vs. native additive (secondary)",
            "mean ± 95% CI, n stated per cell",
            df_to_html(display_df, index_label="baseline / source"),
        ),
        section("Interpretation", "substrate comparison as audit evidence", "\n".join(interp_lines)),
    ])

    html = report_shell(
        title="Baselines 1+2 on the Synthetic Set",
        eyebrow="HLS-CMDS · S3-01/S3-02 re-run, synthetic substrate",
        heading="Baseline 1 (bandpass) + Baseline 2 (supervised NMF): synthetic vs. native",
        dek=(
            "Both implementations are unchanged from their original S3-01/S3-02 landing -- only the "
            "evaluation substrate is new (S1-09/S1-10/S1-13's synthetic mixing set, replacing native "
            "Mix.csv pairs as the primary evaluation substrate). Native additive rows (n=36) are kept "
            "alongside as a secondary column; see the full 5-baseline table "
            "(results/first_sdr_sir_sar_table.html) for the Han &amp; Quan literature column and "
            "Baselines 3-5."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>synthetic_mix.py</code> / <code>baselines.py</code> "
            "/ <code>code_description.md</code>.</p>"
        ),
    )

    report_path = write_report(results_dir() / "baseline1_2_synthetic_report.html", html)
    print(f"\nReport written to {report_path}")
