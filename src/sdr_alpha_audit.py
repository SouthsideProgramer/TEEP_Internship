"""
Audit of the degradation scheme's SDR(alpha) curves: is SDR monotone in
alpha, is every target attainable, how many roots does each target have,
and does the bisection-found alpha in the sweep provenance agree with a
dense-grid root under a stated selection rule?

Why the audit is cheap enough to be exhaustive. degradation.py's blend is
    d(alpha) = (1 - alpha) * g + alpha * s,
linear in alpha. mir_eval's BSS Eval decomposition of an estimate into
(s_true, e_spat, e_interf, e_artif) is a fixed linear projection onto the
span of the (delayed) references -- it depends on the references only --
so every component of d(alpha) is the same convex combination of the
components of g and of s:
    comp(alpha) = (1 - alpha) * comp(g) + alpha * comp(s).
SDR(alpha) = 10 log10(||s_true + e_spat||^2(alpha) / ||e_interf + e_artif||^2(alpha))
is therefore a ratio of two quadratics in alpha, exactly, from two
decompositions per (baseline, row, source) instead of one BSS Eval call
per grid point. A 1001-point grid for all 6 x 36 x 2 costs 432
decompositions plus arithmetic. The closed form is verified against
mir_eval's own bss_eval_sources at several interior alphas before it is
trusted (see check_closed_form); the one thing it cannot reproduce is a
change of the *permutation* mir_eval would pick at some alpha, and the
check would catch that as a mismatch.

Selection rule for a target with several roots (pre-stated here, applied
by sdr_sweep.py's grid mode): the root on the branch that starts at the
ground truth, i.e. the SMALLEST alpha at which SDR(alpha) first descends
through the target when walking from alpha=0 (clean) toward alpha=1 (the
method's real output). A target above SDR(0) or below min SDR over the
grid is reported as unattainable, not clamped.

Outputs:
  results/sdr_alpha_curves.csv      one row per (baseline, mixed_id, source, alpha) -- the dense curves
  results/sdr_alpha_audit.csv       one row per (baseline, mixed_id, source): monotone?, n_reversals,
                                    max_reversal_db, sdr_at_0, sdr_at_1, sdr_min, sdr_max, alpha_at_min
  results/sdr_alpha_targets.csv     one row per (baseline, mixed_id, source, target): attainable?, n_roots,
                                    alpha_grid_root (rule above), alpha_bisection (from provenance), |diff|,
                                    sdr_at_bisection_alpha
  results/sdr_alpha_audit_report.html
  results/plots/sdr_alpha_curves.png

Usage:
    make sdr-alpha-audit
"""
import numpy as np
import pandas as pd

from degradation import ALPHA_GRID, grid_roots, sdr_curve_closed_form
from load_dataset import load_audio
from metrics import SOURCE_LABELS, evaluate_heart_lung
from report_utils import results_dir
from sdr_accuracy_curve import provenance_path
from sdr_sweep import BASELINE_LABELS, cache_dir, estimate_path, native_additive_rows

REVERSAL_TOL_DB = 1e-6  # numerical noise floor for "SDR went up while alpha went up"


def check_closed_form(heart_ref, lung_ref, heart_est, lung_est, source: str, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)) -> float:
    """Max |closed-form - mir_eval| in dB over a few alphas, through the
    project's own evaluate_heart_lung (permutation search included)."""
    refs = np.stack([heart_ref, lung_ref])
    j = SOURCE_LABELS.index(source)
    g, s = (heart_ref, heart_est) if source == "heart" else (lung_ref, lung_est)
    cf = sdr_curve_closed_form(refs, g, s, j, np.array(alphas))
    worst = 0.0
    for a, v in zip(alphas, cf):
        d = (1 - a) * g + a * s
        he, le = (d, lung_est) if source == "heart" else (heart_est, d)
        direct = evaluate_heart_lung(heart_ref, lung_ref, he, le)[source]["sdr"]
        worst = max(worst, abs(direct - v))
    return worst


def audit_curve(alphas: np.ndarray, sdr: np.ndarray) -> dict:
    d = np.diff(sdr)
    reversals = d > REVERSAL_TOL_DB  # SDR rising as alpha rises = non-monotone (expected direction is falling)
    return {
        "monotone_decreasing": bool(not reversals.any()),
        "n_reversals": int(reversals.sum()),
        "max_reversal_db": float(d[reversals].max()) if reversals.any() else 0.0,
        "sdr_at_0": float(sdr[0]), "sdr_at_1": float(sdr[-1]),
        "sdr_min": float(sdr.min()), "sdr_max": float(sdr.max()),
        "alpha_at_min": float(alphas[int(sdr.argmin())]),
    }


def select_root(alphas, sdr, target) -> tuple[float | None, int, bool]:
    """(alpha under the pre-stated rule, n_roots, attainable). Rule: smallest
    alpha at which the curve descends through target starting from alpha=0.
    Not attainable if target > SDR(0) or target < min SDR on the grid."""
    if target > sdr[0] + 1e-9 or target < sdr.min() - 1e-9:
        return None, 0, False
    roots = grid_roots(alphas, sdr, target)
    return (min(roots) if roots else None), len(roots), bool(roots)


def run_audit(baselines=BASELINE_LABELS, alphas=ALPHA_GRID, n_check_rows: int = 6) -> dict:
    mix_df = native_additive_rows()
    root = cache_dir()
    prov = pd.read_csv(provenance_path())
    targets = sorted(prov["target_sdr"].unique())

    curve_rows, audit_rows, target_rows, checks = [], [], [], []
    for label in baselines:
        for k, (_, row) in enumerate(mix_df.iterrows()):
            mid = row["Mixed Sound ID"]
            heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
            lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
            heart_est, _ = load_audio(str(estimate_path(root, label, mid, "heart")), sr=None)
            lung_est, _ = load_audio(str(estimate_path(root, label, mid, "lung")), sr=None)
            n = min(map(len, (heart_ref, lung_ref, heart_est, lung_est)))
            heart_ref, lung_ref, heart_est, lung_est = heart_ref[:n], lung_ref[:n], heart_est[:n], lung_est[:n]
            refs = np.stack([heart_ref, lung_ref])
            for source in SOURCE_LABELS:
                j = SOURCE_LABELS.index(source)
                g, s = (heart_ref, heart_est) if source == "heart" else (lung_ref, lung_est)
                sdr = sdr_curve_closed_form(refs, g, s, j, alphas)
                if k < n_check_rows:
                    checks.append({"baseline": label, "mixed_id": mid, "source": source,
                                   "max_abs_err_db": check_closed_form(heart_ref, lung_ref, heart_est, lung_est, source)})
                curve_rows.extend({"baseline": label, "mixed_id": mid, "source": source, "alpha": float(a), "sdr": float(v)}
                                  for a, v in zip(alphas, sdr))
                audit_rows.append({"baseline": label, "mixed_id": mid, "source": source, **audit_curve(alphas, sdr)})
                pv = prov[(prov["baseline"] == label) & (prov["mixed_id"] == mid) & (prov["source"] == source)].set_index("target_sdr")
                for t in targets:
                    a_rule, n_roots, attainable = select_root(alphas, sdr, t)
                    a_bis = float(pv.loc[t, "alpha"]) if t in pv.index else float("nan")
                    sdr_at_bis = float(np.interp(a_bis, alphas, sdr)) if np.isfinite(a_bis) else float("nan")
                    target_rows.append({
                        "baseline": label, "mixed_id": mid, "source": source, "target_sdr": t,
                        "attainable": attainable, "n_roots": n_roots,
                        "alpha_grid_root": a_rule, "alpha_bisection": a_bis,
                        "abs_alpha_diff": abs(a_rule - a_bis) if (a_rule is not None and np.isfinite(a_bis)) else float("nan"),
                        "sdr_at_bisection_alpha": sdr_at_bis,
                        "bisection_clamped": bool(pv.loc[t, "clamped_to_baseline_floor"]) if t in pv.index else None,
                    })
        print(f"  {label}: done")
    return {
        "curves": pd.DataFrame(curve_rows), "audit": pd.DataFrame(audit_rows),
        "targets": pd.DataFrame(target_rows), "checks": pd.DataFrame(checks),
    }


def plot_curves(curves: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(curves["baseline"].unique())
    fig, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 6), sharex=True, sharey="row")
    for c, label in enumerate(labels):
        for r, source in enumerate(SOURCE_LABELS):
            ax = axes[r, c]
            sub = curves[(curves["baseline"] == label) & (curves["source"] == source)]
            for _, g in sub.groupby("mixed_id"):
                ax.plot(g["alpha"], g["sdr"], lw=0.6, alpha=0.6)
            if r == 0:
                ax.set_title(label.replace("Baseline ", "B").split(" (")[0] + " " + label.split("(")[1].rstrip(")"), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{source} SDR (dB)")
            if r == 1:
                ax.set_xlabel(r"$\alpha$")
            ax.grid(alpha=0.3)
    fig.suptitle(r"SDR($\alpha$) for every native additive row, closed form on a 1001-point grid", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    from report_utils import df_to_html, image_figure, report_shell, section, stat_tile, write_report

    print("Dense SDR(alpha) audit over the 36 native additive rows, 6 baselines, both sources ...")
    out = run_audit()
    rd = results_dir()
    out["curves"].to_csv(rd / "sdr_alpha_curves.csv", index=False)
    out["audit"].to_csv(rd / "sdr_alpha_audit.csv", index=False)
    out["targets"].to_csv(rd / "sdr_alpha_targets.csv", index=False)
    (rd / "plots").mkdir(exist_ok=True)
    plot_curves(out["curves"], rd / "plots" / "sdr_alpha_curves.png")

    audit, tg, checks = out["audit"], out["targets"], out["checks"]
    pd.set_option("display.width", 250)
    print("\nclosed form vs mir_eval, max |err| dB:", checks["max_abs_err_db"].max())
    mono = audit.groupby(["baseline", "source"]).agg(
        n=("monotone_decreasing", "size"), n_non_monotone=("monotone_decreasing", lambda x: int((~x).sum())),
        max_reversal_db=("max_reversal_db", "max"),
        sdr0_min=("sdr_at_0", "min"), sdr1_min=("sdr_at_1", "min"), sdr1_max=("sdr_at_1", "max"),
    )
    print(mono.round(3).to_string())
    att = tg.groupby(["baseline", "source", "target_sdr"]).agg(
        n=("attainable", "size"), n_attainable=("attainable", "sum"), n_multi_root=("n_roots", lambda x: int((x > 1).sum())),
        max_alpha_diff=("abs_alpha_diff", "max"), n_bisection_clamped=("bisection_clamped", "sum"),
    )
    print(att.round(4).to_string())
    heart_att = tg[tg["source"] == "heart"]
    unattainable_not_clamped = heart_att[(~heart_att["attainable"]) & (~heart_att["bisection_clamped"].astype(bool))]
    print("\nheart targets unattainable on the grid but NOT clamped by the bisection run:", len(unattainable_not_clamped))
    big = heart_att[heart_att["abs_alpha_diff"] > 0.01]
    print("heart targets where |alpha_grid - alpha_bisection| > 0.01:", len(big), "of", int(heart_att["attainable"].sum()), "attainable")

    tiles = "\n".join([
        stat_tile("Closed form vs mir_eval", f"{checks['max_abs_err_db'].max():.2e} dB", "max abs error, 5 alphas x 6 rows x 6 baselines x 2 sources"),
        stat_tile("Non-monotone curves", f"{int((~audit['monotone_decreasing']).sum())} / {len(audit)}", f"reversal > {REVERSAL_TOL_DB:g} dB anywhere on the grid"),
        stat_tile("Unattainable & not clamped (heart)", str(len(unattainable_not_clamped)), "targets the bisection reported a value for"),
        stat_tile("|Δalpha| > 0.01 (heart)", f"{len(big)} / {int(heart_att['attainable'].sum())}", "grid rule vs bisection, attainable targets"),
    ])
    body = "\n\n".join([
        section("Every curve", "one line per row; the curves are exact, not sampled",
                image_figure("plots/sdr_alpha_curves.png", "SDR(alpha) per row")),
        section("Monotonicity per (baseline, source)", "n rows, how many have any reversal, largest reversal",
                df_to_html(mono, index_label="baseline / source", float_fmt="{:.3f}")),
        section("Target attainability and root count per (baseline, source, target)",
                "n_multi_root = targets with more than one alpha root; max_alpha_diff = grid-rule root vs the sweep's bisection alpha",
                df_to_html(att, index_label="baseline / source / target", float_fmt="{:.4f}")),
        section("Closed-form check", "max |closed-form SDR - mir_eval SDR| per checked curve",
                df_to_html(checks.set_index(["baseline", "mixed_id", "source"]), index_label="curve", float_fmt="{:.2e}")),
    ])
    html = report_shell(
        title="SDR(alpha) Audit", eyebrow="HLS-CMDS · degradation scheme · monotonicity and attainability",
        heading="Is SDR monotone in alpha, and did the bisection find the right alpha?",
        dek="Exact SDR(alpha) for every (baseline, row, source) from two BSS Eval decompositions, on a 1001-point grid; "
            "compared against the sweep provenance's bisection alphas under a pre-stated root-selection rule.",
        stat_tiles=tiles, body=body,
        footer="<p><strong>Method.</strong> See <code>sdr_alpha_audit.py</code>'s module docstring.</p>",
    )
    print(f"\nReport written to {write_report(rd / 'sdr_alpha_audit_report.html', html)}")
