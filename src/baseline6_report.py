"""
Baseline 6: Conv-TasNet-lite (convtasnet.py) -- the first neural
separation baseline, run through this project's two evaluation
substrates: the synthetic mixing set (S1-09/S1-10/S1-13, the project's
primary substrate -- see synthetic_mix.py) and the 36 native additive
Mix.csv rows as a secondary column, same two-column pattern as
baseline12_synthetic_report.py used for Baselines 1+2.

Baseline 6 is not zero-training like Baselines 1/4/5 or a fixed-dictionary
fit like Baseline 2/3: a full k-fold run retrains the network once per
fold (see convtasnet.make_convtasnet_baseline's docstring), so wall-clock
here is dominated by training, not evaluation -- this script prints
measured per-fold timing and each fold's training diagnostics
(epochs actually run, best validation SI-SDR) rather than assuming the
config in convtasnet.py trained to convergence.

Usage:
    python baseline6_report.py
"""
import time

import numpy as np
import pandas as pd

from convtasnet import BATCH_SIZE, MAX_EPOCHS, STEPS_PER_EPOCH, ConvTasNetLite, make_convtasnet_baseline
from metrics import SOURCE_LABELS


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
    from synthetic_mix import SYNTHETIC_MIX_N_FOLDS, build_synthetic_set, evaluate_synthetic

    label = "Baseline 6 (Conv-TasNet-lite)"
    fit_fn = make_convtasnet_baseline(seed=0, max_epochs=MAX_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE)

    print("Loading native additive subset...")
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid_mix_df = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(f"  {len(valid_mix_df)} / {len(mix_df)} native rows are additive")

    print("Building synthetic mixing set (S1-09/S1-10/S1-13)...")
    synthetic_df = build_synthetic_set(n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)
    print(f"  {len(synthetic_df)} synthetic rows")

    # Wrap fit_fn to capture per-fold training diagnostics as a side channel
    # (train_info is stashed on the returned separate_fn -- see
    # convtasnet.make_convtasnet_baseline's docstring) without changing
    # eval_harness/synthetic_mix's fit_and_separate_fn contract.
    train_diagnostics = []

    def _instrumented_fit_fn(hs_allowed, ls_allowed):
        t0 = time.time()
        separate_fn = fit_fn(hs_allowed, ls_allowed)
        elapsed = time.time() - t0
        info = dict(separate_fn.train_info)
        info["fit_seconds"] = round(elapsed, 1)
        train_diagnostics.append(info)
        print(f"  fold fit: {elapsed:.1f}s, {info['epochs_trained']} epochs, best val SI-SDR {info['best_val_si_sdr']:.2f} dB")
        return separate_fn

    print(f"Running {label} on the synthetic set ({len(synthetic_df)} rows, {SYNTHETIC_MIX_N_FOLDS}-fold retrain)...")
    t_synth0 = time.time()
    synth_results, _fold, _cv = evaluate_synthetic(_instrumented_fit_fn, synthetic_df, n_folds=SYNTHETIC_MIX_N_FOLDS, seed=0)
    synth_elapsed = time.time() - t_synth0
    print(f"  synthetic set total: {synth_elapsed:.1f}s")

    synth_diagnostics = train_diagnostics
    train_diagnostics = []
    print(f"Running {label} on the native additive subset ({len(valid_mix_df)} rows, 5-fold retrain)...")
    t_native0 = time.time()
    native_results, _fold, _cv = cross_validate(_instrumented_fit_fn, n_folds=5, seed=0, mix_df=valid_mix_df)
    native_elapsed = time.time() - t_native0
    print(f"  native additive total: {native_elapsed:.1f}s")
    native_diagnostics = train_diagnostics

    rows = []
    for source in SOURCE_LABELS:
        synth_src = synth_results[synth_results["source"] == source]
        native_src = native_results[native_results["source"] == source]
        row = {"source": source, "synthetic_n": len(synth_src), "native_n": len(native_src)}
        for metric in ("sdr", "sir", "sar"):
            m, ci, _n = _mean_ci95(synth_src[metric].values)
            row[f"synthetic_{metric}"] = _fmt_mean_ci(m, ci)
        for metric in ("sdr", "sir", "sar"):
            m, ci, _n = _mean_ci95(native_src[metric].values)
            row[f"native_{metric}"] = _fmt_mean_ci(m, ci)
        rows.append(row)

    table_df = pd.DataFrame(rows).set_index("source")
    display_cols = ["synthetic_n", "synthetic_sdr", "synthetic_sir", "synthetic_sar", "native_n", "native_sdr", "native_sir", "native_sar"]

    def _diag_table(diagnostics: list) -> pd.DataFrame:
        return pd.DataFrame(diagnostics).rename_axis("fold").reset_index().set_index("fold")

    stat_tiles = "\n".join([
        stat_tile("Baseline", "6", "Conv-TasNet-lite, first neural model"),
        stat_tile("Sample rate", "4000 Hz", "trained from scratch, no resampling"),
        stat_tile("Model size", f"{sum(p.numel() for p in ConvTasNetLite().parameters()):,}", "parameters"),
    ])

    body = "\n\n".join([
        section(
            "SDR / SIR / SAR: synthetic (primary) vs. native additive (secondary)",
            "mean ± 95% CI, n stated per cell",
            df_to_html(table_df[display_cols], index_label="source"),
        ),
        section(
            "Training diagnostics, synthetic-set folds",
            f"{SYNTHETIC_MIX_N_FOLDS} folds, config: max_epochs={MAX_EPOCHS}, steps_per_epoch={STEPS_PER_EPOCH}, batch_size={BATCH_SIZE}",
            df_to_html(_diag_table(synth_diagnostics), index_label="fold"),
        ),
        section(
            "Training diagnostics, native-additive folds",
            "5 folds, same training config",
            df_to_html(_diag_table(native_diagnostics), index_label="fold"),
        ),
        section(
            "Wall-clock",
            "measured this run, not assumed",
            f'<p class="mono-block">synthetic set ({len(synthetic_df)} rows, {SYNTHETIC_MIX_N_FOLDS}-fold retrain): {synth_elapsed:.0f}s'
            f"<br>native additive ({len(valid_mix_df)} rows, 5-fold retrain): {native_elapsed:.0f}s</p>",
        ),
    ])

    html = report_shell(
        title="Baseline 6: Conv-TasNet-lite",
        eyebrow="HLS-CMDS · first neural separation baseline",
        heading="Baseline 6 (Conv-TasNet-lite): synthetic vs. native",
        dek=(
            "First learned end-to-end separation baseline -- encoder/TCN-separator/decoder "
            "architecture (Conv-TasNet's own design, sized down for this dataset's ~50 "
            "recordings/class per fold), trained from scratch at this dataset's native 4000 Hz "
            "(no pretrained speech-separation checkpoint applies at this sample rate -- see "
            "convtasnet.py's module docstring for the sample-rate decision). A full k-fold run "
            "retrains the network once per fold from that fold's own leakage-safe pool; per-fold "
            "training diagnostics and measured wall-clock are reported below rather than assumed."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>convtasnet.py</code> / "
            "<code>code_description.md</code> for architecture, training-loop, and "
            "sample-rate-decision details. Compare against Baselines 1-5 in "
            "<code>results/first_sdr_sir_sar_table.html</code> / "
            "<code>results/baselines_report.html</code>.</p>"
        ),
    )

    report_path = write_report(results_dir() / "baseline6_report.html", html)
    print(f"\nReport written to {report_path}")
