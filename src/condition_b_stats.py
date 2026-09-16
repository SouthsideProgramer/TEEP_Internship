"""
Statistics for Condition B that respect what the data are: 36 paired
row-level observations that are not independent (leak groups), rather
than five fold-level means.

For every separator and both classifier architectures, from
results/canonical_rows.csv (canonical_rows.py):

  accuracy and its 95% CI           -- leak-group cluster bootstrap
  delta = acc_isolated - acc_sep    -- paired by row, cluster bootstrap CI
  permutation p                     -- paired sign-flip test at the leak-group level:
                                       within each resample, every group's per-row
                                       (isolated - separated) differences are flipped
                                       together with probability 1/2
  McNemar p (exact, mid-p)          -- on the row-paired discordant counts; ignores
                                       dependence and is reported for completeness only

The cluster bootstrap resamples the 24 leak groups of the 36-row basis
with replacement and takes all rows of each drawn group, so within-group
correlation is honoured. Percentile intervals, B replicates.

Also reported: the same quantities against the raw-mixture reference
(delta_ref = acc_sep - acc_raw), since the knee-point question is
"better than not separating", not "as good as isolated".

Usage:
    make condition-b-stats
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from report_utils import results_dir
from sdr_sweep import BASELINE_LABELS

ISOLATED = "Isolated (ground truth)"
RAW = "Baseline 0 (raw mixture, no separation)"


def _correct_matrix(df: pd.DataFrame, arch: str) -> tuple[pd.DataFrame, pd.Series]:
    """rows = mixed_id, columns = method, value = correct (bool); plus leak group per row."""
    h = df[df["source"] == "heart"]
    c = (h[f"pred_{arch}"] == h["true_class"]).astype(float)
    mat = pd.DataFrame({"mixed_id": h["mixed_id"], "method": h["method"], "c": c}).pivot(index="mixed_id", columns="method", values="c")
    groups = h.drop_duplicates("mixed_id").set_index("mixed_id")["leak_group"].loc[mat.index]
    return mat, groups


def cluster_bootstrap(mat: pd.DataFrame, groups: pd.Series, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    keys = groups.unique()
    members = {k: mat.index[groups.values == k] for k in keys}
    accs = {m: [] for m in mat.columns}
    for _ in range(n_boot):
        drawn = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([members[k] for k in drawn])
        sub = mat.loc[idx]
        for m in mat.columns:
            accs[m].append(sub[m].mean())
    return {m: np.asarray(v) for m, v in accs.items()}


def permutation_p(diff: pd.Series, groups: pd.Series, n_perm: int, seed: int) -> float:
    """Sign-flip test of mean(diff) == 0 with flips applied per leak group."""
    rng = np.random.default_rng(seed)
    keys = groups.unique()
    gidx = {k: (groups.values == k) for k in keys}
    obs = abs(diff.mean())
    d = diff.to_numpy()
    count = 0
    for _ in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=len(keys))
        s = np.empty_like(d)
        for f, k in zip(flips, keys):
            s[gidx[k]] = f
        if abs((d * s).mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def mcnemar_midp(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """b = discordant with a correct & b wrong, c = the reverse; exact binomial mid-p."""
    n_ab = int(((a == 1) & (b == 0)).sum())
    n_ba = int(((a == 0) & (b == 1)).sum())
    n = n_ab + n_ba
    if n == 0:
        return n_ab, n_ba, 1.0
    k = min(n_ab, n_ba)
    p_two = 2 * stats.binom.cdf(k, n, 0.5) - stats.binom.pmf(k, n, 0.5)  # mid-p
    return n_ab, n_ba, float(min(1.0, p_two))


def run(df: pd.DataFrame, n_boot: int = 5000, n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for arch in ("svm", "cnn"):
        mat, groups = _correct_matrix(df, arch)
        boots = cluster_bootstrap(mat, groups, n_boot, seed)
        for method in [ISOLATED, RAW, *BASELINE_LABELS]:
            acc = mat[method].mean()
            lo, hi = np.percentile(boots[method], [2.5, 97.5])
            rec = {"arch": arch, "method": method, "n": len(mat), "accuracy": acc, "acc_ci_lo": lo, "acc_ci_hi": hi}
            if method != ISOLATED:
                d = mat[ISOLATED] - mat[method]
                bd = boots[ISOLATED] - boots[method]
                rec.update({
                    "delta_vs_isolated": d.mean(),
                    "delta_ci_lo": np.percentile(bd, 2.5), "delta_ci_hi": np.percentile(bd, 97.5),
                    "perm_p_vs_isolated": permutation_p(d, groups, n_perm, seed),
                })
                n_ab, n_ba, p_mc = mcnemar_midp(mat[ISOLATED], mat[method])
                rec.update({"mcnemar_iso_only": n_ab, "mcnemar_sep_only": n_ba, "mcnemar_midp": p_mc})
            if method not in (ISOLATED, RAW):
                d = mat[method] - mat[RAW]
                bd = boots[method] - boots[RAW]
                rec.update({
                    "delta_vs_raw": d.mean(),
                    "delta_raw_ci_lo": np.percentile(bd, 2.5), "delta_raw_ci_hi": np.percentile(bd, 97.5),
                    "perm_p_vs_raw": permutation_p(d, groups, n_perm, seed),
                })
            rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, section, stat_tile, write_report

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(results_dir() / "canonical_rows.csv")
    out = run(df, args.n_boot, args.n_perm, args.seed)
    out.to_csv(results_dir() / "condition_b_stats.csv", index=False)
    pd.set_option("display.width", 300)
    cols = ["arch", "method", "accuracy", "acc_ci_lo", "acc_ci_hi", "delta_vs_isolated", "delta_ci_lo", "delta_ci_hi",
            "perm_p_vs_isolated", "mcnemar_midp", "delta_vs_raw", "delta_raw_ci_lo", "delta_raw_ci_hi", "perm_p_vs_raw"]
    print(out[cols].round(3).to_string(index=False))

    n_groups = df.drop_duplicates("mixed_id")["leak_group"].nunique()
    sections = []
    for arch in ("svm", "cnn"):
        sub = out[out["arch"] == arch].set_index("method")[cols[2:]]
        sections.append(section(f"Architecture {'1 (MFCC+SVM)' if arch == 'svm' else '2 (log-Mel+CNN)'}",
                                f"{args.n_boot} cluster-bootstrap replicates over {n_groups} leak groups; {args.n_perm} sign-flip permutations",
                                df_to_html(sub, index_label="method", float_fmt="{:.3f}")))
    html = report_shell(
        title="Condition B Statistics", eyebrow="HLS-CMDS · Condition B · row-level, leak-group-aware inference",
        heading="Accuracy, paired deltas and their uncertainty without pretending 36 rows are 5 folds or 36 independent draws",
        dek="Cluster bootstrap over leak groups for accuracy and paired-delta CIs; group-level sign-flip permutation test; McNemar for completeness.",
        stat_tiles=stat_tile("Resampling unit", f"{n_groups} leak groups", "36 rows"),
        body="\n\n".join(sections),
        footer="<p><strong>Method.</strong> See <code>condition_b_stats.py</code>'s module docstring.</p>",
    )
    print(f"\nReport written to {write_report(results_dir() / 'condition_b_stats_report.html', html)}")
