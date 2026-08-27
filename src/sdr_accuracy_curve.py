"""
S6-03: measure downstream classification accuracy at every point in S6-02's
controlled SDR sweep -- the actual accuracy-vs-SDR curve PROTOCOL.md Sec.
5.3.1 designed the x-axis for, and Sec. 4's C2 knee-point question needs.
Widened across the Sprint 6 holiday gap: Sprint 6 has only three working
days (24 Sep Thu, 29 Sep Tue, 30 Sep Wed -- 25/28 Sep are public holidays).
Design (S6-01) and sweep generation (S6-02) already happened in S5, so
this ticket and S6-04 (not this one -- presumably the curve/knee-point
write-up) are what's left, which is what fits in three days.

This script does *not* redo generation. It consumes S6-02's already-built
outputs directly:
  - results/sdr_sweep_provenance.csv (sdr_sweep.build_sdr_sweep()'s table:
    one row per baseline x mixed_id x source x target_sdr)
  - results/sdr_sweep_cache/ (each baseline's real separated audio, cached
    once per (baseline, fold) by build_sdr_sweep())
and Condition B's weight-sharing classifier setup (condition_b.py): the
*same* per-fold trained classifier used throughout PROTOCOL.md Sec. 5.3,
never retrained on separated or degraded audio.

Only source == "heart" rows are measured -- this project's classifier
(heart_classifier.py) classifies heart sounds only; the sweep's lung-source
rows exist for separation-quality bookkeeping (metrics.py's 2-source BSS
Eval needs both sources scored together) but have no classification
counterpart.

Scripted to run unattended, per the brief: results are checkpointed to
results/sdr_accuracy_curve.csv incrementally (every CHECKPOINT_EVERY rows),
and a re-run skips (baseline, mixed_id, target_sdr) points already present
in that file -- safe to interrupt (e.g. across the holiday gap) and resume
without redoing completed work or needing supervision.

Usage:
    from sdr_accuracy_curve import measure_accuracy_at_each_sdr_point

    results_df = measure_accuracy_at_each_sdr_point()  # reads/writes results/sdr_accuracy_curve.csv
"""
from pathlib import Path

import pandas as pd

import heart_classifier
from condition_b import build_condition_b_fold_basis
from heart_classifier import HEART_TYPE_TO_GROUP
from report_utils import results_dir
from sdr_sweep import cache_dir, synthesize_sweep_row

CHECKPOINT_EVERY = 25


def provenance_path() -> Path:
    return results_dir() / "sdr_sweep_provenance.csv"


def output_path(backend=heart_classifier) -> Path:
    """Backend-specific checkpoint file (S7-06) -- Architecture 1 (default)
    keeps the original filename for backward compatibility; other backends
    get their own module-name-suffixed file so the two never collide."""
    suffix = "" if backend is heart_classifier else f"_{backend.__name__}"
    return results_dir() / f"sdr_accuracy_curve{suffix}.csv"


def load_sweep_provenance(path: Path | None = None) -> pd.DataFrame:
    path = path if path is not None else provenance_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run `make sdr-sweep` (S6-02) first to generate the sweep this "
            "script measures accuracy over."
        )
    return pd.read_csv(path)


def _load_checkpoint(path: Path) -> tuple[list[dict], set]:
    if not path.is_file():
        return [], set()
    existing = pd.read_csv(path)
    done = set(zip(existing["baseline"], existing["mixed_id"], existing["target_sdr"]))
    return existing.to_dict("records"), done


def measure_accuracy_at_each_sdr_point(
    sweep_df: pd.DataFrame | None = None,
    n_folds: int = 5,
    seed: int = 0,
    cache_root: Path | None = None,
    out_path: Path | None = None,
    resume: bool = True,
    checkpoint_every: int = CHECKPOINT_EVERY,
    mix_df: pd.DataFrame | None = None,
    backend=heart_classifier,
) -> pd.DataFrame:
    """
    For every source == "heart" row in sweep_df (defaults to loading
    results/sdr_sweep_provenance.csv): reconstruct the degraded audio
    (sdr_sweep.synthesize_sweep_row(), cheap -- the expensive separation
    step already happened in S6-02), classify it with the fold-appropriate
    trained classifier (Condition B's weight-sharing setup, never
    retrained on degraded/separated audio), and record whether the
    prediction matches the row's true class_group.

    backend (S7-06): the classifier module to use -- default
    heart_classifier (Architecture 1, MFCC+SVM); pass
    heart_classifier_cnn (Architecture 2) to measure the robustness-check
    curve for the second architecture. Also determines out_path's default
    (see below) so the two architectures' checkpoints never collide.

    Returns the full results table (existing checkpoint rows + newly
    measured ones), and leaves it written to out_path as a side effect.
    """
    sweep_df = sweep_df if sweep_df is not None else load_sweep_provenance()
    heart_rows = sweep_df[sweep_df["source"] == "heart"].reset_index(drop=True)

    fold_mix_df, hs_df = build_condition_b_fold_basis(n_folds=n_folds, seed=seed, mix_df=mix_df)
    classifiers = backend.train_fold_classifiers(hs_df, n_folds=n_folds)
    root = cache_root if cache_root is not None else cache_dir()
    path = out_path if out_path is not None else output_path(backend)

    mix_by_id = fold_mix_df.set_index("Mixed Sound ID")

    results, done = _load_checkpoint(path) if resume else ([], set())
    if done:
        print(f"Resuming: {len(done)} (baseline, mixed_id, target_sdr) points already measured.")

    n_measured_this_run = 0
    for i, row in heart_rows.iterrows():
        key = (row["baseline"], row["mixed_id"], row["target_sdr"])
        if key in done:
            continue

        # Trust the provenance row's own fold/label (recorded once by
        # sdr_sweep.build_sdr_sweep()) rather than re-deriving them; assert
        # they agree with an independent recomputation as a consistency
        # check, not a silent assumption that the two never drift apart.
        fold = int(row["fold"])
        true = HEART_TYPE_TO_GROUP[row["Heart Sound Type"]]
        recomputed_fold = int(mix_by_id.loc[row["mixed_id"], "fold"])
        assert recomputed_fold == fold, (
            f"fold mismatch for {row['mixed_id']}: sweep provenance says {fold}, "
            f"recomputed classifier fold basis says {recomputed_fold}"
        )

        degraded, sr = synthesize_sweep_row(row, mix_df=fold_mix_df, cache_root=root)
        pred = backend.predict_one(classifiers[fold], degraded, sr)

        results.append({
            "baseline": row["baseline"],
            "mixed_id": row["mixed_id"],
            "fold": fold,
            "target_sdr": row["target_sdr"],
            "achieved_sdr": row["achieved_sdr"],
            "alpha": row["alpha"],
            "clamped_to_baseline_floor": row["clamped_to_baseline_floor"],
            "true": true,
            "pred": pred,
            "correct": bool(pred == true),
        })
        done.add(key)
        n_measured_this_run += 1

        if n_measured_this_run % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(path, index=False)
            print(f"  ... {len(results)}/{len(heart_rows)} points measured (checkpoint written)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(path, index=False)
    print(f"Done: {len(results_df)}/{len(heart_rows)} points measured, written to {path}")
    return results_df


def summarize_accuracy_curve(results_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (baseline, target_sdr) -- the curve's actual data points,
    x = mean achieved_sdr (not target_sdr itself, since achieved varies
    slightly row to row and clamped points sit at that baseline's own
    floor regardless of target -- see sdr_sweep.py's clamped_to_baseline_floor)."""
    rows = []
    for (baseline, target), group in results_df.groupby(["baseline", "target_sdr"], sort=False):
        rows.append({
            "baseline": baseline,
            "target_sdr": target,
            "n": len(group),
            "mean_achieved_sdr": group["achieved_sdr"].mean(),
            "accuracy": group["correct"].mean(),
            "pct_clamped": group["clamped_to_baseline_floor"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["baseline", "target_sdr"], ascending=[True, False]).reset_index(drop=True)


def plot_accuracy_curve(curve_df: pd.DataFrame, output_path: Path, isolated_accuracy: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for baseline, group in curve_df.groupby("baseline", sort=False):
        group = group.sort_values("mean_achieved_sdr")
        ax.plot(group["mean_achieved_sdr"], group["accuracy"], marker="o", label=baseline)

    if isolated_accuracy is not None:
        ax.axhline(isolated_accuracy, color="black", linestyle="--", linewidth=1, label="Isolated (ground truth)")

    ax.set_xlabel("Measured SDR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs. separation quality (S6-03)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from report_utils import df_to_html, image_figure, report_shell, section, stat_tile, write_report

    print("Measuring downstream classification accuracy at each point in the SDR sweep (S6-03)...")
    print("(unattended-safe: re-running this script resumes from results/sdr_accuracy_curve.csv)")
    results_df = measure_accuracy_at_each_sdr_point()

    curve_df = summarize_accuracy_curve(results_df)

    isolated_accuracy = None
    try:
        from condition_b import ISOLATED_LABEL, evaluate_condition_b, summarize_by_baseline

        isolated_predictions = evaluate_condition_b(baseline_labels=[])
        isolated_accuracy = float(summarize_by_baseline(isolated_predictions).loc[ISOLATED_LABEL, "accuracy"])
    except Exception as exc:  # pragma: no cover -- best-effort reference line, not load-bearing
        print(f"  (could not compute the isolated reference line: {exc})")

    plots_dir = results_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "sdr_accuracy_curve.png"
    plot_accuracy_curve(curve_df, plot_path, isolated_accuracy=isolated_accuracy)

    n_points = len(curve_df)
    overall_pct_clamped = results_df["clamped_to_baseline_floor"].mean()

    stat_tiles = "\n".join([
        stat_tile("Points measured", str(len(results_df)), f"{n_points} (baseline, target-SDR) groups"),
        stat_tile("Isolated accuracy", f"{isolated_accuracy:.1%}" if isolated_accuracy is not None else "n/a", "reference line"),
        stat_tile("Clamped to floor", f"{overall_pct_clamped:.1%}", "of all measured points"),
    ])

    body = "\n\n".join([
        section(
            "Accuracy vs. measured SDR",
            "one line per separation baseline, dashed reference = isolated ground truth",
            image_figure("plots/sdr_accuracy_curve.png", "accuracy vs. SDR curve"),
        ),
        section(
            "Curve data",
            "accuracy per (baseline, target SDR), x = mean achieved SDR across the rows in that group",
            df_to_html(curve_df.set_index(["baseline", "target_sdr"]), index_label="baseline / target_sdr", float_fmt="{:.3f}"),
        ),
    ])

    html = report_shell(
        title="SDR Accuracy Curve",
        eyebrow="HLS-CMDS · S6-03 · accuracy vs. separation quality",
        heading="Downstream accuracy at each point in the controlled SDR sweep",
        dek=(
            "Measures the Condition B classifier (same trained weights per fold as Condition A) "
            "on every point S6-02's degradation sweep generated -- the actual x-axis PROTOCOL.md "
            "Sec. 5.3.1 designed, now populated with real accuracy measurements instead of just "
            "separation-quality numbers."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>sdr_accuracy_curve.py</code>'s module docstring, "
            "<code>sdr_sweep.py</code> (S6-02) for how the sweep was generated, and "
            "<code>condition_b.py</code> for the weight-sharing classifier setup this reuses.</p>"
        ),
    )

    report_path = write_report(results_dir() / "sdr_accuracy_curve_report.html", html)
    print(f"\nReport written to {report_path}")
