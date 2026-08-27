"""
Plot the SDR-versus-compute plane: separation quality (heart SDR, native
additive 36-row subset) against two independent compute-cost axes --
MACs (compute_cost.py, theoretical/analytic) and desktop latency
(latency.py, real measured wall-clock) -- for all six separation
baselines.

Two panels, not one collapsed "compute" axis, because MACs and latency
were already found to diverge substantially for at least one method
(latency.py's own finding: Conv-TasNet-lite has the second-highest MACs
but the *lowest* measured latency of all six methods, an implementation-
efficiency effect, not a computation-cost one). Collapsing to a single
compute axis here would hide exactly the kind of divergence this project
already found once and should be checked for again, not assumed away.

SDR values are reused from this project's own already-published, real
measurements, not re-run from scratch: SDR is a stable property of a
fixed-seed method on a fixed dataset (unlike latency, which is sensitive
to machine conditions *at measurement time* and would need re-measuring
to stay valid -- SDR does not). Baselines 1-5's native-additive heart SDR
(mean +/- 95% CI, pooled over the 36 rows) come from `results/
first_sdr_sir_sar_table.html` (src/first_sdr_table.py, generated
2026-08-24). Baseline 6 postdates that report (added 2026-08-25) and was
measured fresh in this same session, in the identical pooled-95%-CI
convention, specifically so all six numbers are apples-to-apples rather
than mixing that convention with PROTOCOL.md's own separate
mean-+/-std-across-folds convention for the same baselines (a real,
avoidable inconsistency this module deliberately does not introduce).

MACs (compute_cost.py) and latency (latency.py) are pulled directly from
those modules -- MACs cheaply recomputed fresh each run (fast, no
expensive audio processing beyond one Conv-TasNet forward pass and one
classifier fit); latency reused from its own already-measured, disclosed
(and machine-load-caveated) report rather than re-measured here, since a
fresh re-measurement would take as long as the original run and add no
new information over reusing that already-real number.

Pareto dominance: baseline A is dominated (on a given compute axis) by
baseline B if B has >= SDR and <= compute, with at least one strict
inequality -- i.e. B is at least as good on both axes and strictly better
on one. Marked distinctly (hollow vs. filled markers) so a reader can see
at a glance which methods are strictly worse choices on that axis, not
just visually lower/righter on the plot.

Usage:
    python sdr_compute_plane.py
"""
from pathlib import Path

# Heart SDR, native additive 36-row subset, mean +/- 95% CI (pooled across
# the 36 rows). Baselines 1-5: results/first_sdr_sir_sar_table.html
# (src/first_sdr_table.py, generated 2026-08-24) -- read directly from
# that report, not retyped from memory. Baseline 6: measured fresh this
# session (2026-08-28) via eval_harness.cross_validate on the same 36-row
# subset, same pooled-95%-CI formula, since it postdates that report.
SDR_HEART_DB = {
    "Baseline 1 (bandpass)": (5.46, 2.57),
    "Baseline 2 (supervised NMF)": (2.58, 1.80),
    "Baseline 3 (standard NMF)": (3.35, 1.82),
    "Baseline 4 (MSSA)": (5.02, 2.81),
    "Baseline 5 (EVMD)": (3.61, 2.27),
    "Baseline 6 (Conv-TasNet-lite)": None,  # filled in by measure_baseline6_sdr() -- see __main__
}

# Desktop latency (median wall-clock, ms), from results/latency_report.html
# (src/latency.py, measured 2026-08-27 under disclosed heavy machine
# contention -- see that report for the load-average caveat). Reused
# as-is rather than re-measured: a fresh run would cost as much wall-clock
# time as the original and add no new information.
LATENCY_MS = {
    "Baseline 1 (bandpass)": 36.003,
    "Baseline 2 (supervised NMF)": 5200.305,
    "Baseline 3 (standard NMF)": 21035.749,
    "Baseline 4 (MSSA)": 12909.875,
    "Baseline 5 (EVMD)": 90075.068,
    "Baseline 6 (Conv-TasNet-lite)": 23.906,
}


def measure_baseline6_sdr(n_folds: int = 5, seed: int = 0) -> tuple[float, float]:
    """Baseline 6's native-additive heart SDR, in the same pooled-95%-CI
    convention as first_sdr_table.py's Baselines 1-5 (see module
    docstring for why this one baseline needs a fresh measurement)."""
    import numpy as np

    from convtasnet import make_convtasnet_baseline
    from eval_harness import cross_validate
    from load_dataset import load_mix, verify_additive_triplets

    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    valid_mix_df = mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)

    results_df, _fold_summary, _cv_summary = cross_validate(
        make_convtasnet_baseline(seed=seed), n_folds=n_folds, seed=seed, mix_df=valid_mix_df
    )
    heart_sdr = results_df.loc[results_df["source"] == "heart", "sdr"].to_numpy()
    n = len(heart_sdr)
    mean = float(heart_sdr.mean())
    ci95 = 1.96 * float(heart_sdr.std(ddof=1) / (n**0.5))
    return mean, ci95


def compute_macs() -> dict[str, float]:
    """MACs per inference, one per separation baseline (compute_cost.py) --
    excludes the classifier row, which has no SDR to plot against."""
    from compute_cost import bandpass_macs, evmd_macs, mssa_macs, standard_nmf_macs, supervised_nmf_macs
    from compute_cost import convtasnet_params_and_macs

    rows = [bandpass_macs(), supervised_nmf_macs(), standard_nmf_macs(), mssa_macs(), evmd_macs(), convtasnet_params_and_macs()]
    return {r["method"]: r["macs_per_inference"] for r in rows}


def is_pareto_dominated(name: str, sdr: dict[str, float], compute: dict[str, float]) -> bool:
    """True if some other method has >= SDR and <= compute, with at least
    one strict inequality -- i.e. name is never the better choice on this
    axis."""
    for other, other_sdr in sdr.items():
        if other == name:
            continue
        better_or_equal = other_sdr >= sdr[name] and compute[other] <= compute[name]
        strictly_better = other_sdr > sdr[name] or compute[other] < compute[name]
        if better_or_equal and strictly_better:
            return True
    return False


def plot_sdr_compute_plane(sdr_mean: dict, sdr_ci: dict, macs: dict, latency_ms: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(sdr_mean.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, compute, xlabel, title in [
        (axes[0], macs, "MACs per inference (log scale)", "SDR vs. MACs (theoretical compute)"),
        (axes[1], latency_ms, "Median wall-clock latency, ms (log scale)", "SDR vs. desktop latency (measured, contended machine)"),
    ]:
        for method in methods:
            dominated = is_pareto_dominated(method, sdr_mean, compute)
            short_label = method.split(" (")[1].rstrip(")")
            ax.errorbar(
                compute[method], sdr_mean[method], yerr=sdr_ci[method],
                fmt="o", markersize=9,
                markerfacecolor="none" if dominated else None,
                markeredgewidth=2, capsize=4,
                label=f"{short_label} *" if dominated else short_label,
            )
            ax.annotate(
                method.split(" (")[0].replace("Baseline ", "B"),
                (compute[method], sdr_mean[method]),
                textcoords="offset points", xytext=(8, 4), fontsize=9,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Heart SDR, native additive rows (dB)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("SDR vs. compute: hollow markers = Pareto-dominated on that axis (* in legend)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd

    from report_utils import df_to_html, image_figure, report_shell, results_dir, section, stat_tile, write_report

    print("Measuring Baseline 6's native-additive heart SDR (postdates first_sdr_table.html)...")
    b6_mean, b6_ci = measure_baseline6_sdr()
    print(f"  Baseline 6 (Conv-TasNet-lite): {b6_mean:.2f} +/- {b6_ci:.2f} dB (95% CI, n=36)")
    sdr_full = dict(SDR_HEART_DB)
    sdr_full["Baseline 6 (Conv-TasNet-lite)"] = (b6_mean, b6_ci)

    sdr_mean = {k: v[0] for k, v in sdr_full.items()}
    sdr_ci = {k: v[1] for k, v in sdr_full.items()}

    print("Computing MACs per method (compute_cost.py)...")
    macs = compute_macs()

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "sdr_compute_plane.png"
    plot_sdr_compute_plane(sdr_mean, sdr_ci, macs, LATENCY_MS, plot_path)

    rows = []
    for method in sdr_mean:
        rows.append({
            "method": method,
            "heart_sdr_db": sdr_mean[method],
            "sdr_ci95": sdr_ci[method],
            "macs": macs[method],
            "latency_ms": LATENCY_MS[method],
            "pareto_dominated_macs": is_pareto_dominated(method, sdr_mean, macs),
            "pareto_dominated_latency": is_pareto_dominated(method, sdr_mean, LATENCY_MS),
        })
    table_df = pd.DataFrame(rows).set_index("method")

    n_dominated_macs = table_df["pareto_dominated_macs"].sum()
    n_dominated_latency = table_df["pareto_dominated_latency"].sum()

    stat_tiles = "\n".join([
        stat_tile("Methods plotted", str(len(sdr_mean)), "6 separation baselines"),
        stat_tile("Dominated on MACs axis", f"{n_dominated_macs}/6", "strictly worse SDR+MACs than another method"),
        stat_tile("Dominated on latency axis", f"{n_dominated_latency}/6", "strictly worse SDR+latency than another method"),
    ])

    body = "\n\n".join([
        section(
            "SDR vs. compute (both axes)",
            "hollow markers = Pareto-dominated on that axis; error bars = 95% CI (n=36 native additive rows)",
            image_figure("plots/sdr_compute_plane.png", "SDR vs. MACs and SDR vs. latency"),
        ),
        section(
            "Underlying data",
            "heart SDR (native additive, pooled 95% CI) + MACs (compute_cost.py) + latency (latency.py, measured under disclosed machine contention)",
            df_to_html(table_df, index_label="method", float_fmt="{:.2f}"),
        ),
    ])

    html = report_shell(
        title="SDR vs. Compute Plane",
        eyebrow="HLS-CMDS · separation quality vs. compute cost",
        heading="The SDR-versus-compute plane, two ways",
        dek=(
            "Separation quality (heart SDR) against two independent compute-cost axes -- MACs "
            "(theoretical) and desktop latency (measured) -- for all six separation baselines. Plotted "
            "as two panels, not one, because the two compute axes were already found to diverge "
            "substantially for at least one method (latency.py's own finding on Conv-TasNet-lite)."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>sdr_compute_plane.py</code>'s module docstring for "
            "the provenance of every number plotted (which report/session each came from) and the "
            "Pareto-dominance definition.</p>"
        ),
    )

    report_path = write_report(results_dir() / "sdr_compute_plane_report.html", html)
    print(f"\nReport written to {report_path}")
    print(table_df.to_string())
