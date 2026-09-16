"""
Bootstrap confidence intervals for the knee points of sdr_knee_point.py.

Why. Every point on the accuracy-vs-SDR curve is an accuracy over the 36
native additive rows (one row = 2.8 points), and the knee is where that
curve crosses the no-separation reference -- itself an accuracy over the
same 36 rows. A knee quoted to 0.1 dB from that is a point estimate with
no stated uncertainty; report.tex already refuses to quote individual
knee values for this reason. This module puts numbers on it.

What is resampled. The unit is the mixture row. Each replicate draws 36
rows with replacement from the 36 mixed_ids, and *every* quantity the
knee depends on is recomputed on that same draw -- the per-target-level
accuracy, the per-target-level mean achieved SDR (the curve's x-axis),
and the no-separation reference accuracy -- so the resampled reference
moves with the resampled curve rather than being held at its point
estimate. The knee is then found with the identical function the
headline figure uses (sdr_knee_point.find_knee_point): descending-SDR
walk, first high-to-low crossing, linear interpolation between the two
bracketing measured points.

Two resampling schemes are run, because the 36 rows are not independent
(split.py's leak groups: rows sharing a heart or lung recording):
  rows   -- i.i.d. row bootstrap. Standard; optimistic if rows within a
            leak group are correlated.
  groups -- cluster bootstrap over the leak groups present in the 36-row
            basis: draw groups with replacement, take all their rows.
            Honours the dependence structure; coarser (fewer units).

What is reported per (architecture, baseline). Because a replicate may
find no crossing at all, the knee's sampling distribution is a mixture of
a status and a value, and both are reported: the fraction of replicates
in each status ("crossed" / "always_above" / "always_below" /
"noisy_crossing" / "ambiguous"), and, over the replicates that crossed,
the median and 2.5/97.5 percentiles of the knee SDR. Quoting only the
percentile interval would hide that, for some baselines, a third of
replicates have no knee.

Inputs (all produced by the existing pipeline; nothing is re-classified):
  results/sdr_accuracy_curve.csv                       Arch 1, per row x target
  results/sdr_accuracy_curve_heart_classifier_cnn.csv  Arch 2, per row x target
  results/no_separation_predictions_<backend>.csv      per-row reference
      (written by this module on first run via
      condition_b.evaluate_no_separation, since sdr_knee_point.py does not
      persist it)

Usage:
    make sdr-knee-bootstrap
    python sdr_knee_bootstrap.py --n-boot 2000 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import heart_classifier
import heart_classifier_cnn
from condition_b import NO_SEPARATION_LABEL, build_condition_b_fold_basis, evaluate_no_separation
from report_utils import results_dir
from sdr_accuracy_curve import output_path as curve_path
from sdr_knee_point import find_knee_point

BACKENDS = {"heart_classifier": heart_classifier, "heart_classifier_cnn": heart_classifier_cnn}
STATUSES = ["crossed", "always_above", "always_below", "noisy_crossing", "ambiguous"]


def no_separation_path(backend_name: str) -> Path:
    return results_dir() / f"no_separation_predictions_{backend_name}.csv"


def load_or_measure_no_separation(backend_name: str, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per-row correctness of the raw-mixture reference; cached because the
    CNN backend retrains five folds to produce it."""
    path = no_separation_path(backend_name)
    if path.is_file():
        return pd.read_csv(path)
    df = evaluate_no_separation(backend=BACKENDS[backend_name], n_folds=n_folds, seed=seed)
    df["correct"] = df["true"] == df["pred"]
    df.to_csv(path, index=False)
    return df


def leak_groups_for_basis(n_folds: int = 5, seed: int = 0) -> pd.Series:
    """mixed_id -> leak_group on the 36-row Condition B fold basis."""
    mix_df, _ = build_condition_b_fold_basis(n_folds=n_folds, seed=seed)
    return mix_df.set_index("Mixed Sound ID")["leak_group"]


def curve_and_knee_on_rows(
    per_row: pd.DataFrame, no_sep_correct: pd.Series, sampled_ids: np.ndarray, baseline: str
) -> dict:
    """Recompute one baseline's curve, its reference, and its knee on a
    multiset of mixed_ids (duplicates count as many times as drawn)."""
    counts = pd.Series(sampled_ids).value_counts()
    sub = per_row[per_row["baseline"] == baseline]
    sub = sub[sub["mixed_id"].isin(counts.index)]
    w = sub["mixed_id"].map(counts).to_numpy(dtype=float)
    g = sub.assign(w=w).groupby("target_sdr")
    curve = pd.DataFrame({
        "mean_achieved_sdr": g.apply(lambda d: np.average(d["achieved_sdr"], weights=d["w"])),
        "accuracy": g.apply(lambda d: np.average(d["correct"].astype(float), weights=d["w"])),
    }).reset_index()
    ref = float(np.average(no_sep_correct.loc[counts.index].astype(float), weights=counts.to_numpy(dtype=float)))
    return find_knee_point(curve, ref)


def bootstrap_backend(
    backend_name: str, n_boot: int, seed: int, scheme: str, n_folds: int = 5, split_seed: int = 0
) -> pd.DataFrame:
    per_row = pd.read_csv(curve_path(BACKENDS[backend_name]))
    per_row["correct"] = per_row["correct"].astype(bool)
    no_sep = load_or_measure_no_separation(backend_name, n_folds=n_folds, seed=split_seed)
    no_sep_correct = no_sep.set_index("mixed_id")["correct"].astype(bool)

    ids = np.array(sorted(per_row["mixed_id"].unique()))
    assert set(ids) == set(no_sep_correct.index), "curve rows and reference rows must be the same 36 mixtures"
    groups = leak_groups_for_basis(n_folds=n_folds, seed=split_seed)
    group_members = {g: np.array(sorted(m.index)) for g, m in groups.loc[ids].groupby(groups.loc[ids])}
    group_keys = np.array(sorted(group_members))

    rng = np.random.default_rng(seed)
    baselines = list(per_row["baseline"].unique())
    draws = {b: [] for b in baselines}
    for _ in range(n_boot):
        if scheme == "rows":
            sampled = rng.choice(ids, size=len(ids), replace=True)
        elif scheme == "groups":
            picked = rng.choice(group_keys, size=len(group_keys), replace=True)
            sampled = np.concatenate([group_members[k] for k in picked])
        else:
            raise ValueError(scheme)
        for b in baselines:
            draws[b].append(curve_and_knee_on_rows(per_row, no_sep_correct, sampled, b))

    rows = []
    for b in baselines:
        d = pd.DataFrame(draws[b])
        knees = d.loc[d["status"] == "crossed", "knee_sdr"].dropna().to_numpy(dtype=float)
        row = {"backend": backend_name, "scheme": scheme, "baseline": b, "n_boot": n_boot}
        for st in STATUSES:
            row[f"p_{st}"] = float((d["status"] == st).mean())
        row["knee_median"] = float(np.median(knees)) if len(knees) else float("nan")
        row["knee_ci_lo"] = float(np.percentile(knees, 2.5)) if len(knees) else float("nan")
        row["knee_ci_hi"] = float(np.percentile(knees, 97.5)) if len(knees) else float("nan")
        row["n_units"] = len(ids) if scheme == "rows" else len(group_keys)
        rows.append(row)
    return pd.DataFrame(rows)


def point_estimates(backend_name: str, n_folds: int = 5, split_seed: int = 0) -> pd.DataFrame:
    """The headline (non-resampled) knee per baseline, via the same path, so
    the table can show it next to the interval."""
    per_row = pd.read_csv(curve_path(BACKENDS[backend_name]))
    no_sep = load_or_measure_no_separation(backend_name, n_folds=n_folds, seed=split_seed)
    ids = np.array(sorted(per_row["mixed_id"].unique()))
    rows = []
    for b in per_row["baseline"].unique():
        k = curve_and_knee_on_rows(per_row, no_sep.set_index("mixed_id")["correct"].astype(bool), ids, b)
        rows.append({"backend": backend_name, "baseline": b, "knee_point": k["knee_sdr"], "status_point": k["status"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, section, stat_tile, write_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tables = []
    points = []
    for name in BACKENDS:
        print(f"{name}: reference per row ...")
        pts = point_estimates(name)
        points.append(pts)
        for scheme in ("rows", "groups"):
            print(f"  bootstrap {scheme}, {args.n_boot} replicates ...")
            tables.append(bootstrap_backend(name, args.n_boot, args.seed, scheme))
    boot = pd.concat(tables, ignore_index=True)
    pts = pd.concat(points, ignore_index=True)
    out = boot.merge(pts, on=["backend", "baseline"], how="left")
    out.to_csv(results_dir() / "sdr_knee_bootstrap.csv", index=False)

    show_cols = ["backend", "scheme", "baseline", "status_point", "knee_point", "p_crossed", "p_always_above",
                 "p_always_below", "p_noisy_crossing", "knee_median", "knee_ci_lo", "knee_ci_hi", "n_units"]
    pd.set_option("display.width", 250)
    print(out[show_cols].round(3).to_string(index=False))

    sections = []
    for name in BACKENDS:
        for scheme in ("rows", "groups"):
            sub = out[(out["backend"] == name) & (out["scheme"] == scheme)].set_index("baseline")[show_cols[3:]]
            sections.append(section(
                f"{name} -- {scheme} bootstrap",
                f"{args.n_boot} replicates, seed {args.seed}, {int(sub['n_units'].iloc[0])} resampling units",
                df_to_html(sub, index_label="baseline", float_fmt="{:.2f}"),
            ))
    html = report_shell(
        title="Knee-Point Bootstrap",
        eyebrow="HLS-CMDS · knee-point uncertainty",
        heading="How sure is each knee point?",
        dek=("Row and leak-group bootstrap of the accuracy-vs-SDR curve, its no-separation reference and the knee, "
             "recomputed per replicate with sdr_knee_point.find_knee_point. p_* columns are the fraction of "
             "replicates ending in each status; the interval is over the replicates that crossed."),
        stat_tiles=stat_tile("Replicates", str(args.n_boot), "per (architecture, baseline, scheme)"),
        body="\n\n".join(sections),
        footer="<p><strong>Method.</strong> See <code>sdr_knee_bootstrap.py</code>'s module docstring.</p>",
    )
    print(f"\nReport written to {write_report(results_dir() / 'sdr_knee_bootstrap_report.html', html)}")
