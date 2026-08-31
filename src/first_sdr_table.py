"""
The first SDR/SIR/SAR table across all five separation baselines
(bandpass, supervised NMF, standard NMF, MSSA, EVMD), three columns per
(baseline, source):

  1. Synthetic set (S1-09/S1-10/S1-13, synthetic_mix.py) -- the primary
     evaluation substrate now, n in the thousands.
  2. Native additive Mix.csv rows (n=36) -- reported alongside, not as a
     footnote: agreement or gap between the two substrates is itself audit
     evidence for whether the synthetic substrate is a valid stand-in.
  3. Han & Quan's own Table I (`papers/Cardiorespiratory_Sound_Separation_
     Using_Singular_Spectrum_Analysis.pdf`, Sec. III), marked as their setup
     (their own 50-pair synthetic set, their own hyperparameters) -- not
     recomputed, no CI.

Every computed cell (columns 1-2) states mean, 95% CI, and n explicitly, so
a tight synthetic-column CI (n in the thousands) is never confused with a
tight native-column CI (n=36).

Usage:
    python first_sdr_table.py
"""
import numpy as np
import pandas as pd

from baseline.baseline1 import fit_bandpass_baseline
from baseline.baseline2 import make_supervised_nmf_baseline
from baseline.baseline3 import make_standard_nmf_baseline
from baseline.baseline4 import fit_ssa_baseline
from baseline.baseline5 import fit_evmd_baseline
from metrics import SOURCE_LABELS
from synthetic_mix import SYNTHETIC_MIX_N_FOLDS, SNR_SWEEP_DB, build_synthetic_set, evaluate_synthetic

BASELINE_SPECS = [
    ("Baseline 1 (bandpass)", fit_bandpass_baseline, False),
    ("Baseline 2 (supervised NMF)", make_supervised_nmf_baseline(seed=0), False),
    ("Baseline 3 (standard NMF)", make_standard_nmf_baseline(seed=0), False),
    ("Baseline 4 (MSSA)", fit_ssa_baseline, False),
    ("Baseline 5 (EVMD)", fit_evmd_baseline, True),
]

EVMD_SYNTHETIC_PER_STRATUM = 5

HAN_QUAN_TABLE_I = {
    "Baseline 1 (bandpass)": {
        "heart_sdr": 5.7, "lung_sdr": -5.7, "heart_corr": 91.4, "lung_corr": 29.7,
        "note": "Table I \"Butterworth Filter\" row -- direct match to this baseline's method.",
    },
    "Baseline 2 (supervised NMF)": {
        "heart_sdr": 1.0, "lung_sdr": -2.3, "heart_corr": 50.0, "lung_corr": 10.2,
        "note": "Table I \"NMF\" row -- their blind NMF, not the same method as this baseline's pretrained-dictionary supervised NMF; same method family only.",
    },
    "Baseline 3 (standard NMF)": {
        "heart_sdr": 1.0, "lung_sdr": -2.3, "heart_corr": 50.0, "lung_corr": 10.2,
        "note": "Table I \"NMF\" row -- closer in spirit to this baseline's blind per-mixture NMF than Baseline 2's, but still not a literal match (different rank/STFT/denoising choices).",
    },
    "Baseline 4 (MSSA)": {
        "heart_sdr": 26.4, "lung_sdr": 5.3, "heart_corr": 99.2, "lung_corr": 80.5,
        "note": "Table I \"MSSA\" row -- their own proposed method, direct match to this baseline's method.",
    },
    "Baseline 5 (EVMD)": None,
}


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


def _stratified_subsample(df: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    """Evenly sample `per_stratum` rows from each (fold, snr_db) group -- keeps the
    subsample spread across the full difficulty sweep and every fold, not skewed."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby(["fold", "snr_db"]):
        idx = rng.choice(group.index.to_numpy(), size=min(per_stratum, len(group)), replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts).reset_index(drop=True)


def _build_table(synthetic_df: pd.DataFrame, evmd_synthetic_df: pd.DataFrame, valid_mix_df: pd.DataFrame) -> pd.DataFrame:
    from eval_harness import cross_validate

    rows = []
    for label, fit_fn, expensive in BASELINE_SPECS:
        this_synthetic_df = evmd_synthetic_df if expensive else synthetic_df

        print(f"Running {label}...")
        print(f"  synthetic ({len(this_synthetic_df)} rows)...")
        synth_results, _fold, _cv = evaluate_synthetic(fit_fn, this_synthetic_df, n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)

        print(f"  native additive ({len(valid_mix_df)} rows)...")
        native_results, _fold, _cv = cross_validate(fit_fn, n_folds=5, seed=0, mix_df=valid_mix_df)

        hq = HAN_QUAN_TABLE_I.get(label)

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

            if hq is None:
                row["hanquan_n"] = "n/a"
                row["hanquan_sdr"] = "not applicable"
                row["hanquan_corr"] = "not applicable"
                row["hanquan_note"] = "EVMD source paper reports no separation metric at all -- this table is what fills that gap."
            else:
                row["hanquan_n"] = "50 (their synthetic set)"
                row["hanquan_sdr"] = f"{hq[f'{source}_sdr']:.1f} dB"
                row["hanquan_corr"] = f"{hq[f'{source}_corr']:.1f}%"
                row["hanquan_note"] = hq["note"]

            rows.append(row)

    return pd.DataFrame(rows)


def _interpretation_html(table_df: pd.DataFrame) -> str:
    lines = [
        "<p>Synthetic-vs-native SDR agreement per baseline/source (synthetic mean minus "
        "native mean, dB -- near 0 means the two substrates agree; the synthetic set's "
        "much larger n makes its CI tight regardless of this gap, so a tight CI here is "
        "not itself evidence of agreement with the native rows):</p><ul>"
    ]
    for _, row in table_df.iterrows():
        gap = row["_synthetic_sdr_mean"] - row["_native_sdr_mean"]
        lines.append(f"<li><strong>{row['baseline']}</strong> ({row['source']}): {gap:+.2f} dB</li>")
    lines.append("</ul>")
    lines.append(
        "<p>The EVMD source paper (<code>[edgelung]</code>) runs its separation stage "
        "directly on HLS-CMDS mixtures but reports no SDR/SIR/SAR or correlation for "
        "either source -- only downstream lung-classification accuracy. Baseline 5's "
        "columns above are, as far as this project has found, the first separation-quality "
        "numbers computed for that method on this dataset.</p>"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from load_dataset import load_mix, verify_additive_triplets
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Loading native additive subset...")
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid_mix_df = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(f"  {len(valid_mix_df)} / {len(mix_df)} native rows are additive")

    print("Building synthetic mixing set (S1-09/S1-10/S1-13)...")
    synthetic_df = build_synthetic_set(n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)
    evmd_synthetic_df = _stratified_subsample(synthetic_df, EVMD_SYNTHETIC_PER_STRATUM, seed=0)
    print(f"  {len(synthetic_df)} rows (full); Baseline 5's stratified subsample: {len(evmd_synthetic_df)} rows")

    table_df = _build_table(synthetic_df, evmd_synthetic_df, valid_mix_df)

    display_cols = [
        "synthetic_n", "synthetic_sdr", "synthetic_sir", "synthetic_sar",
        "native_n", "native_sdr", "native_sir", "native_sar",
        "hanquan_n", "hanquan_sdr", "hanquan_corr", "hanquan_note",
    ]
    display_df = table_df.set_index(["baseline", "source"])[display_cols]

    stat_tiles = "\n".join([
        stat_tile("Methods", "5", "bandpass, supervised NMF, standard NMF, MSSA, EVMD"),
        stat_tile("Synthetic n", str(len(synthetic_df)), f"{SYNTHETIC_MIX_N_FOLDS} folds x {len(SNR_SWEEP_DB)} SNR levels"),
        stat_tile("Native additive n", str(len(valid_mix_df)), f"of {len(mix_df)} Mix.csv rows"),
    ])

    body = "\n\n".join([
        section(
            "SDR / SIR / SAR by method and source",
            "mean ± 95% CI, n stated per cell",
            df_to_html(display_df, index_label="baseline / source"),
        ),
        section("Interpretation", "substrate agreement + the gap this table fills", _interpretation_html(table_df)),
    ])

    html = report_shell(
        title="First SDR/SIR/SAR Table",
        eyebrow="HLS-CMDS · separation quality, synthetic substrate",
        heading="First SDR/SIR/SAR table: 5 methods, 3 substrates",
        dek=(
            "Synthetic set (S1-09/S1-10/S1-13) is now the primary evaluation substrate "
            "(only 36/145 native Mix.csv rows are additive). Native additive rows are "
            "reported alongside as a distinct column, not folded in -- the agreement or "
            "gap between the two substrates is itself audit evidence. Han &amp; Quan's own "
            "Table I is a third, static column marked as their setup (their 50-pair "
            "synthetic set, their hyperparameters, no recomputed CI). Baseline 5 (EVMD)'s "
            "synthetic column runs on a stratified subsample (n stated in its row) -- the "
            "full K=2..10 VMD sweep at the full synthetic n (~1500 rows) was extrapolated at "
            "~6 hours from measured per-row cost, outside this "
            "session's timebox; see BACKLOG.md."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>synthetic_mix.py</code> / "
            "<code>baselines.py</code> / <code>code_description.md</code> for construction, "
            "baseline, and interpretation details.</p>"
        ),
    )

    report_path = write_report(results_dir() / "first_sdr_sir_sar_table.html", html)
    print(f"\nReport written to {report_path}")
