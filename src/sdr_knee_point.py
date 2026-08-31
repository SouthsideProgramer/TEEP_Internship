"""
S6-04: plot the accuracy-vs-SDR curve and identify the knee point -- the
project's headline figure, per the Sprint 0 re-scope (the separation track
finished early enough that C2, not C1, became the paper's primary
contribution; see BACKLOG.md). Treat everything upstream of this module
(split.py's leak-safe folds, the six separation baselines, heart_
classifier.py's Condition A, degradation.py's S6-01 interpolation scheme,
sdr_sweep.py's S6-02 generation, condition_b.py's weight-sharing setup,
sdr_accuracy_curve.py's S6-03 point-by-point measurement) as scaffolding
for this one curve.

Why this curve is the thing no prior work has: Yaqub et al.
(`[spectrotemporal]`) report accuracy collapsing 89%->41% with no
separation-quality number attached to either end; the HLS-CMDS descriptor
paper (Torabi et al.) and every separation-quality table in this project's
own PROTOCOL.md Sec. 5.2 report SDR with no downstream classification
accuracy attached. This is the first place both axes live on one plot for
this dataset.

Knee-point definition -- operationalizing the charter's own wording
(PROTOCOL.md Sec. 4: "the SDR below which separation actively hurts
classification accuracy, relative to not separating at all"), not an
arbitrary curvature-based "elbow" heuristic: the SDR at which a baseline's
accuracy(SDR) curve crosses the **no-separation reference accuracy**
(condition_b.evaluate_no_separation() -- classifying the raw, unseparated
mixture directly), walking from high SDR (clean) down to low SDR (hard).
Above the knee, separating helps (or at least doesn't hurt); below it, a
classifier would have done as well or better without separating at all.
Found by linear interpolation between the two adjacent *measured* points
bracketing the crossing -- not a smoothed fit, since smoothing over only
len(TARGET_SDR_GRID_DB) points per baseline (each averaging a handful of
rows) would manufacture precision the data doesn't support. n per point is
always reported alongside the knee estimate for exactly this reason.

Scope and honesty note: this module works off whatever baselines happen to
have measured accuracy-curve data available. If S6-02's generation hasn't
finished all six baselines yet, the figure it renders is real but partial
-- and its own report says so explicitly (which baselines are present,
which are still pending) rather than silently plotting fewer lines than
the full picture will eventually have.

Cross-architecture robustness check (S7-06) -- no longer optional once C2
is the paper's headline result, not a secondary result the charter's
"one architecture, it's just a measuring instrument" rule was written
for. Once the knee point IS the paper, the first reviewer question is
whether it's a property of separation quality or of the one architecture
that measured it. `run_pipeline_for_backend()` runs this module's entire
curve-and-knee-point pipeline for a given classifier backend
(heart_classifier.py, Architecture 1; heart_classifier_cnn.py,
Architecture 2), and `compare_knee_points()` reports, per baseline,
whether the two architectures' knee points agree within a stated
tolerance -- agreement means the knee is a property of separation
quality; disagreement is a more interesting finding than the one
originally planned, not a failure of the check.

Usage:
    from sdr_knee_point import find_knee_points, plot_curve_with_knee_points, run_pipeline_for_backend, compare_knee_points
"""
from pathlib import Path

import pandas as pd


def find_knee_point(curve_for_one_baseline: pd.DataFrame, no_separation_accuracy: float) -> dict:
    """
    curve_for_one_baseline: rows for a single baseline with
    'mean_achieved_sdr' and 'accuracy' columns
    (sdr_accuracy_curve.summarize_accuracy_curve()'s output, filtered to
    one baseline).

    Returns {"knee_sdr": float | None, "status": str, "n_points": int}.
    status is one of:
      "crossed"        -- accuracy crosses the no-separation reference
                           somewhere in the measured range, walking from
                           high SDR to low SDR; knee_sdr is the
                           interpolated crossing point.
      "noisy_crossing" -- no clean single high->low crossing was found
                           (small-n noise), but a low->high crossing
                           exists when scanning the other direction --
                           reported distinctly, not silently treated the
                           same as "crossed".
      "always_above"   -- separation beats the no-separation reference at
                           every measured SDR; no knee in range.
      "always_below"   -- separation never beats the no-separation
                           reference at any measured SDR.
      "ambiguous"       -- neither a clean nor a noisy crossing found
                           (should only happen with very few points).
    """
    ordered = curve_for_one_baseline.sort_values("mean_achieved_sdr", ascending=False).reset_index(drop=True)
    sdrs = ordered["mean_achieved_sdr"].to_numpy()
    accs = ordered["accuracy"].to_numpy()

    above = accs >= no_separation_accuracy
    if above.all():
        return {"knee_sdr": None, "status": "always_above", "n_points": len(ordered)}
    if not above.any():
        return {"knee_sdr": None, "status": "always_below", "n_points": len(ordered)}

    for i in range(len(above) - 1):
        if above[i] and not above[i + 1]:
            sdr_hi, sdr_lo = sdrs[i], sdrs[i + 1]
            acc_hi, acc_lo = accs[i], accs[i + 1]
            knee = (sdr_hi + sdr_lo) / 2 if acc_hi == acc_lo else (
                sdr_hi + (no_separation_accuracy - acc_hi) / (acc_lo - acc_hi) * (sdr_lo - sdr_hi)
            )
            return {"knee_sdr": float(knee), "status": "crossed", "n_points": len(ordered)}

    for i in range(len(above) - 1):
        if not above[i] and above[i + 1]:
            return {"knee_sdr": float(sdrs[i]), "status": "noisy_crossing", "n_points": len(ordered)}

    return {"knee_sdr": None, "status": "ambiguous", "n_points": len(ordered)}


def find_knee_points(curve_df: pd.DataFrame, no_separation_accuracy: float) -> pd.DataFrame:
    """find_knee_point() for every baseline present in curve_df."""
    rows = []
    for baseline, group in curve_df.groupby("baseline", sort=False):
        rows.append({"baseline": baseline, **find_knee_point(group, no_separation_accuracy)})
    return pd.DataFrame(rows).set_index("baseline")


def present_baselines_for(root: Path) -> tuple[list[str], list[str]]:
    """(present, missing) baseline labels -- a baseline "counts" once at
    least one of its cached separated .wav files exists on disk."""
    from sdr_sweep import BASELINE_LABELS, estimate_path, native_additive_rows

    mix_df_check = native_additive_rows()
    first_mixed_id = mix_df_check.iloc[0]["Mixed Sound ID"]
    present = [
        label for label in BASELINE_LABELS
        if estimate_path(root, label, first_mixed_id, "heart").is_file()
    ]
    missing = [label for label in BASELINE_LABELS if label not in present]
    return present, missing


def run_pipeline_for_backend(backend, present_baselines: list[str], root: Path, n_folds: int = 5, seed: int = 0) -> dict:
    """
    The full curve-and-knee-point pipeline (S6-03 measurement + reference
    points + knee-point detection) for one classifier backend. Used both
    for a single-architecture report and, twice, for S7-06's
    cross-architecture robustness comparison.
    """
    from condition_b import ISOLATED_LABEL, NO_SEPARATION_LABEL, evaluate_condition_b, evaluate_no_separation, summarize_by_baseline
    from sdr_accuracy_curve import load_sweep_provenance, measure_accuracy_at_each_sdr_point, summarize_accuracy_curve
    from sdr_sweep import build_provenance_from_cache, native_additive_rows
    from split import assign_folds

    try:
        sweep_df = load_sweep_provenance()
        sweep_df = sweep_df[sweep_df["baseline"].isin(present_baselines)]
    except FileNotFoundError:
        mix_df = assign_folds(mix_df=native_additive_rows(), n_folds=n_folds, seed=seed)
        sweep_df = build_provenance_from_cache(mix_df, present_baselines, cache_root=root, n_folds=n_folds, seed=seed)

    results_df = measure_accuracy_at_each_sdr_point(sweep_df=sweep_df, backend=backend, n_folds=n_folds, seed=seed)
    curve_df = summarize_accuracy_curve(results_df)

    isolated_predictions = evaluate_condition_b(baseline_labels=[], backend=backend, n_folds=n_folds, seed=seed)
    isolated_accuracy = float(summarize_by_baseline(isolated_predictions).loc[ISOLATED_LABEL, "accuracy"])

    no_sep_predictions = evaluate_no_separation(backend=backend, n_folds=n_folds, seed=seed)
    no_sep_summary = summarize_by_baseline(no_sep_predictions)
    no_separation_accuracy = float(no_sep_summary.loc[NO_SEPARATION_LABEL, "accuracy"])
    no_separation_sdr = float(no_sep_summary.loc[NO_SEPARATION_LABEL, "mean_achieved_sdr"])

    knee_points_df = find_knee_points(curve_df, no_separation_accuracy)

    return {
        "results_df": results_df,
        "curve_df": curve_df,
        "knee_points_df": knee_points_df,
        "isolated_accuracy": isolated_accuracy,
        "no_separation_accuracy": no_separation_accuracy,
        "no_separation_sdr": no_separation_sdr,
    }


def compare_knee_points(knee_points_by_backend: dict[str, pd.DataFrame], tolerance_db: float = 3.0) -> pd.DataFrame:
    """
    S7-06's actual verdict: for each baseline present in every backend's
    knee_points_df, compare the estimated knee_sdr across architectures.
    "agrees" is True only when *every* architecture found a clean
    "crossed" knee within tolerance_db dB of each other; any other
    combination of statuses is reported as an inconclusive comparison
    (agrees=None), not silently coerced into a yes/no -- a "no clean
    knee" status on either side means there is nothing to compare
    numerically yet, which is itself informative, not a null result.
    """
    backend_names = list(knee_points_by_backend.keys())
    if len(backend_names) < 2:
        raise ValueError("compare_knee_points needs at least two backends")
    baselines = set.intersection(*(set(df.index) for df in knee_points_by_backend.values()))

    rows = []
    for baseline in sorted(baselines):
        row = {"baseline": baseline}
        knees, statuses = {}, {}
        for name in backend_names:
            entry = knee_points_by_backend[name].loc[baseline]
            row[f"{name}_knee_sdr"] = entry["knee_sdr"]
            row[f"{name}_status"] = entry["status"]
            knees[name], statuses[name] = entry["knee_sdr"], entry["status"]

        if all(status == "crossed" for status in statuses.values()):
            values = list(knees.values())
            spread = max(values) - min(values)
            row["knee_sdr_spread_db"] = spread
            row["agrees"] = bool(spread <= tolerance_db)
        else:
            row["knee_sdr_spread_db"] = float("nan")
            row["agrees"] = None

        rows.append(row)
    return pd.DataFrame(rows).set_index("baseline")


def plot_curve_with_knee_points(
    curve_df: pd.DataFrame,
    no_separation_accuracy: float,
    isolated_accuracy: float | None,
    knee_points_df: pd.DataFrame,
    output_path: Path,
    missing_baselines: list[str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for baseline, group in curve_df.groupby("baseline", sort=False):
        group = group.sort_values("mean_achieved_sdr")
        (line,) = ax.plot(group["mean_achieved_sdr"], group["accuracy"], marker="o", label=baseline)
        if baseline in knee_points_df.index:
            knee = knee_points_df.loc[baseline, "knee_sdr"]
            if pd.notna(knee):
                ax.axvline(knee, color=line.get_color(), linestyle=":", alpha=0.6, linewidth=1.2)

    ax.axhline(no_separation_accuracy, color="black", linestyle="--", linewidth=1.3, label="No separation (raw mixture)")
    if isolated_accuracy is not None:
        ax.axhline(isolated_accuracy, color="gray", linestyle="-.", linewidth=1.3, label="Isolated (ground truth)")

    ax.set_xlabel("Measured SDR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1)
    title = "Accuracy vs. separation quality, with knee points"
    if missing_baselines:
        title += f"\n(partial: {len(missing_baselines)} baseline(s) not yet measured)"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import heart_classifier
    import heart_classifier_cnn
    from report_utils import df_to_html, image_figure, report_shell, results_dir, section, stat_tile, write_report
    from sdr_sweep import cache_dir

    BACKENDS = {"heart_classifier": heart_classifier, "heart_classifier_cnn": heart_classifier_cnn}

    print("Determining which baselines have real separated audio cached (S6-02)...")
    root = cache_dir()
    present_baselines, missing_baselines = present_baselines_for(root)
    print(f"  present: {present_baselines}")
    if missing_baselines:
        print(f"  NOT YET GENERATED (S6-02 still running?): {missing_baselines}")

    pipeline_results = {}
    for name, backend in BACKENDS.items():
        print(f"Running the curve-and-knee-point pipeline for {name} (S7-06)...")
        pipeline_results[name] = run_pipeline_for_backend(backend, present_baselines, root)

    print("Comparing knee points across architectures (S7-06's robustness verdict)...")
    comparison_df = compare_knee_points({name: r["knee_points_df"] for name, r in pipeline_results.items()})

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_arch_sections = []
    for name, r in pipeline_results.items():
        plot_path = plots_dir / f"sdr_knee_point_curve_{name}.png"
        plot_curve_with_knee_points(
            r["curve_df"], r["no_separation_accuracy"], r["isolated_accuracy"], r["knee_points_df"], plot_path,
            missing_baselines=missing_baselines,
        )
        per_arch_sections.append(section(
            f"{name}",
            f"isolated {r['isolated_accuracy']:.1%}, no-separation {r['no_separation_accuracy']:.1%}",
            image_figure(f"plots/{plot_path.name}", f"{name} accuracy-vs-SDR curve")
            + df_to_html(r["knee_points_df"], index_label="baseline", float_fmt="{:.2f}"),
        ))

    stat_tiles = "\n".join([
        stat_tile("Baselines measured", f"{len(present_baselines)}/6", ", ".join(present_baselines) or "none yet"),
        stat_tile(
            "Architecture 1 isolated acc.", f"{pipeline_results['heart_classifier']['isolated_accuracy']:.1%}", "MFCC + SVM",
        ),
        stat_tile(
            "Architecture 2 isolated acc.", f"{pipeline_results['heart_classifier_cnn']['isolated_accuracy']:.1%}",
            "log-mel + CNN",
            ok=pipeline_results["heart_classifier_cnn"]["isolated_accuracy"] >= 0.4,
        ),
    ])

    partial_note = (
        f'<p class="note"><strong>Partial figure.</strong> {len(missing_baselines)} of 6 baselines '
        f"({', '.join(missing_baselines)}) had no cached separated audio yet when this ran — "
        "re-run <code>make sdr-knee-point</code> once S6-02's generation finishes for the complete "
        "six-method figure.</p>"
        if missing_baselines else
        '<p class="note">All 6 separation baselines measured — this is the complete figure.</p>'
    )

    arch2_isolated = pipeline_results["heart_classifier_cnn"]["isolated_accuracy"]
    arch1_isolated = pipeline_results["heart_classifier"]["isolated_accuracy"]
    robustness_caveat = (
        "<p class='note'><strong>Read the comparison below with this in mind:</strong> Architecture 2 "
        f"(log-mel + CNN) own Condition A accuracy ({arch2_isolated:.1%}) is substantially below "
        f"Architecture 1's ({arch1_isolated:.1%}) in this first configuration (no hyperparameter search "
        "-- see heart_classifier_cnn_report.html for its confusion matrix, which shows a collapse "
        "toward predicting one class). A knee-point disagreement could reflect a genuine "
        "separation-quality-dependent effect, or it could simply reflect Architecture 2 not yet being "
        "a reliable enough classifier on its own to support the comparison -- both are real "
        "possibilities this first run cannot distinguish between.</p>"
        if arch2_isolated < 0.4 else ""
    )

    body = "\n\n".join([
        section("Status", "", partial_note + robustness_caveat),
        section(
            "Cross-architecture knee-point comparison (S7-06)",
            f"agrees = both architectures found a clean knee within {3.0} dB of each other; "
            "None = at least one architecture found no clean crossing to compare",
            df_to_html(comparison_df, index_label="baseline", float_fmt="{:.2f}"),
        ),
        section("Per-architecture curves and knee points", "", "\n\n".join(per_arch_sections)),
    ])

    html = report_shell(
        title="SDR Knee Point Robustness",
        eyebrow="HLS-CMDS · S6-04 + S7-06 · headline figure + robustness check",
        heading="Accuracy vs. SDR: where separation stops helping, checked against two architectures",
        dek=(
            "The first plot to put HLS-CMDS classification accuracy and separation-quality SDR on "
            "one axis (Yaqub et al. report accuracy with no SDR attached; this project's own "
            "separation baselines report SDR with no downstream accuracy), now repeated against a "
            "second, architecturally distinct classifier (S7-06) to check whether the knee point is "
            "a property of separation quality or an artifact of one classifier's decision boundary."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>sdr_knee_point.py</code>'s module docstring for "
            "the knee-point definition and the robustness-check design, "
            "<code>heart_classifier_cnn.py</code> for Architecture 2, and PROTOCOL.md Sec. 4/5.3.3 "
            "for the full C2 framing.</p>"
        ),
    )

    report_path = write_report(results_dir() / "sdr_knee_point_report.html", html)
    print(f"\nReport written to {report_path}")
