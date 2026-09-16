"""
Pointwise uncertainty for the accuracy-vs-SDR trajectories, and the
publication figure drawn from it.

For every (architecture, baseline, target level) from the exact-root sweep:
    accuracy pooled over the 36 rows (the quantity the trajectories plot),
    a 95% pointwise interval from a leak-group cluster bootstrap (all rows
    of a drawn group enter together; B replicates),
    n_rows contributing, n_leak_groups, n_attainable (rows whose target
    was reached), n_clamped (unattainable: placed at the method's real
    output), n_multi_root (rows whose curve has more than one alpha at the
    target; the smallest is used), n_perm_flip (rows where mir_eval's
    permutation would have swapped the assignment), mean achieved SDR.
The no-separation and isolated references get the same bootstrap.

Outputs:
    results/trajectory_points.csv
    results/plots/trajectories_bands.png / .pdf  -- two panels (Arch 1, Arch 2),
        pooled accuracy with pointwise bands, n_attainable annotated per point,
        clamped-dominated points hollow, reference lines with their own bands.

Usage:
    make trajectory-points
"""
import argparse

import numpy as np
import pandas as pd

import heart_classifier
import heart_classifier_cnn
from condition_b import build_condition_b_fold_basis
from report_utils import results_dir
from sdr_accuracy_curve import output_path as curve_path, provenance_path
from sdr_knee_bootstrap import load_or_measure_no_separation
from sdr_sweep import BASELINE_LABELS

BACKENDS = {"heart_classifier": heart_classifier, "heart_classifier_cnn": heart_classifier_cnn}
ARCH_NAME = {"heart_classifier": "Architecture 1 (MFCC+SVM)", "heart_classifier_cnn": "Architecture 2 (log-Mel+CNN)"}


def _groups() -> pd.Series:
    mix_df, _ = build_condition_b_fold_basis(n_folds=5, seed=0)
    return mix_df.set_index("Mixed Sound ID")["leak_group"]


def _cluster_boot_acc(correct: pd.Series, groups: pd.Series, n_boot: int, rng) -> tuple[float, float]:
    keys = groups.unique()
    members = {k: correct.index[groups.values == k] for k in keys}
    accs = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([members[k] for k in drawn])
        accs[b] = correct.loc[idx].mean()
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def build_points(n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    prov = pd.read_csv(provenance_path())
    prov = prov[prov["source"] == "heart"][["baseline", "mixed_id", "target_sdr", "attainable", "n_roots", "permutation_flipped", "curve_monotone"]]
    groups = _groups()
    rng = np.random.default_rng(seed)
    rows = []
    for name, backend in BACKENDS.items():
        cur = pd.read_csv(curve_path(backend)).merge(prov, on=["baseline", "mixed_id", "target_sdr"], how="left")
        cur["correct"] = cur["correct"].astype(bool)
        for (bl, t), g in cur.groupby(["baseline", "target_sdr"], sort=False):
            c = g.set_index("mixed_id")["correct"].astype(float)
            gg = groups.loc[c.index]
            lo, hi = _cluster_boot_acc(c, gg, n_boot, rng)
            rows.append({
                "backend": name, "baseline": bl, "target_sdr": float(t),
                "accuracy": float(c.mean()), "ci_lo": lo, "ci_hi": hi,
                "n_rows": int(len(c)), "n_leak_groups": int(gg.nunique()),
                "n_attainable": int(g["attainable"].sum()), "n_clamped": int((~g["attainable"].astype(bool)).sum()),
                "n_multi_root": int((g["n_roots"] > 1).sum()), "n_perm_flip": int(g["permutation_flipped"].sum()),
                "n_non_monotone": int((~g["curve_monotone"].astype(bool)).sum()),
                "mean_achieved_sdr": float(g["achieved_sdr"].mean()),
            })
        # references
        no_sep = load_or_measure_no_separation(name).set_index("mixed_id")["correct"].astype(float)
        lo, hi = _cluster_boot_acc(no_sep, groups.loc[no_sep.index], n_boot, rng)
        rows.append({"backend": name, "baseline": "Raw mixture (no separation)", "target_sdr": np.nan,
                     "accuracy": float(no_sep.mean()), "ci_lo": lo, "ci_hi": hi, "n_rows": len(no_sep),
                     "n_leak_groups": int(groups.loc[no_sep.index].nunique())})
        can = pd.read_csv(results_dir() / "canonical_rows.csv")
        iso = can[(can["method"] == "Isolated (ground truth)") & (can["source"] == "heart")].set_index("mixed_id")
        arch = "svm" if name == "heart_classifier" else "cnn"
        c = (iso[f"pred_{arch}"] == iso["true_class"]).astype(float)
        lo, hi = _cluster_boot_acc(c, groups.loc[c.index], n_boot, rng)
        rows.append({"backend": name, "baseline": "Isolated (ground truth)", "target_sdr": np.nan,
                     "accuracy": float(c.mean()), "ci_lo": lo, "ci_hi": hi, "n_rows": len(c),
                     "n_leak_groups": int(groups.loc[c.index].nunique())})
    return pd.DataFrame(rows)


def plot(points: pd.DataFrame, path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, name in zip(axes, BACKENDS):
        sub = points[points["backend"] == name]
        for i, bl in enumerate(BASELINE_LABELS):
            p = sub[sub["baseline"] == bl].sort_values("mean_achieved_sdr")
            short = bl.replace("Baseline ", "B").split(" (")[0] + " " + bl.split("(")[1].rstrip(")")
            ax.fill_between(p["mean_achieved_sdr"], p["ci_lo"], p["ci_hi"], color=colors[i], alpha=0.10, lw=0)
            ax.plot(p["mean_achieved_sdr"], p["accuracy"], "-", color=colors[i], lw=1.4, label=short)
            full = p["n_attainable"] >= 18
            ax.plot(p.loc[full, "mean_achieved_sdr"], p.loc[full, "accuracy"], "o", color=colors[i], ms=5)
            ax.plot(p.loc[~full, "mean_achieved_sdr"], p.loc[~full, "accuracy"], "o", mfc="white", mec=colors[i], ms=5)
            for _, r in p.iterrows():
                ax.annotate(str(int(r["n_attainable"])), (r["mean_achieved_sdr"], r["accuracy"]),
                            textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color=colors[i])
        ref = sub[sub["baseline"] == "Raw mixture (no separation)"].iloc[0]
        iso = sub[sub["baseline"] == "Isolated (ground truth)"].iloc[0]
        ax.axhspan(ref["ci_lo"], ref["ci_hi"], color="k", alpha=0.06, lw=0)
        ax.axhline(ref["accuracy"], color="k", ls="--", lw=1, label=f"raw mixture ({100*ref['accuracy']:.1f}%)")
        ax.axhline(iso["accuracy"], color="gray", ls="-.", lw=1, label=f"isolated ({100*iso['accuracy']:.1f}%)")
        ax.set_title(ARCH_NAME[name], fontsize=10)
        ax.set_xlabel("mean achieved heart SDR at the target level (dB)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pooled accuracy, 36 native additive rows")
    axes[0].legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pts = build_points(args.n_boot, args.seed)
    pts.to_csv(results_dir() / "trajectory_points.csv", index=False)
    (results_dir() / "plots").mkdir(exist_ok=True)
    plot(pts, results_dir() / "plots" / "trajectories_bands")
    pd.set_option("display.width", 250)
    show = pts[pts["target_sdr"].notna()]
    print(show[["backend", "baseline", "target_sdr", "accuracy", "ci_lo", "ci_hi", "n_attainable", "n_clamped", "n_multi_root", "n_perm_flip", "mean_achieved_sdr"]].round(3).to_string(index=False))
    print(pts[pts["target_sdr"].isna()][["backend", "baseline", "accuracy", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print(f"\nwrote {results_dir() / 'trajectory_points.csv'} and plots/trajectories_bands.{{png,pdf}}")
