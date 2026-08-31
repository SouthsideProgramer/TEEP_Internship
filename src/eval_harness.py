"""
K-fold evaluation harness for heart/lung separation models on the mix set.

See code_description.md for how it wires split.py's fold assignment into
metrics.py's BSS Eval and how results get aggregated per-fold and across
folds.

Usage:
    from eval_harness import cross_validate

    def fit_and_separate(hs_allowed, ls_allowed):
        dictionary = my_dictionary_learning(hs_allowed, ls_allowed)
        def separate(mixed, sr):
            return my_separation(mixed, sr, dictionary)
        return separate

    results_df, fold_summary, cv_summary = cross_validate(fit_and_separate, n_folds=5, seed=0)
"""
import pandas as pd

from load_dataset import load_hs, load_ls
from metrics import SOURCE_LABELS, evaluate_dataset
from split import assign_folds, dictionary_pool


def cross_validate(
    fit_and_separate_fn,
    n_folds: int = 5,
    seed: int = 0,
    compute_permutation: bool = True,
    mix_df: pd.DataFrame | None = None,
):
    """
    Run leakage-safe k-fold cross-validation.

    fit_and_separate_fn: callable(hs_allowed: pd.DataFrame, ls_allowed: pd.DataFrame) -> separate_fn
        Called once per fold with that fold's allowed dictionary-fitting pool
        (HS.csv/LS.csv rows, in load_dataset.load_hs()/load_ls() format, with
        every recording that appears in the held-out fold's mixtures already
        removed). Must return separate_fn(mixed, sr) -> (heart_est, lung_est).

    mix_df: optional pre-filtered Mix.csv rows (load_dataset.load_mix() format,
        without fold columns -- those are computed here) to restrict
        evaluation to, e.g. load_dataset.verify_additive_triplets()'s valid
        subset. Folds are then assigned within that subset only. Defaults to
        the full Mix.csv.

    Returns:
        results_df: one row per (fold, Mixed Sound ID, source) -> sdr/sir/sar
                     plus the row's class/location labels.
        fold_summary: mean sdr/sir/sar per (fold, source).
        cv_summary: mean +/- std across folds per source -- the headline
                     cross-validated result.
    """
    hs_df, ls_df = load_hs(), load_ls()
    mix_df = assign_folds(mix_df=mix_df, n_folds=n_folds, seed=seed)

    fold_results = []
    for k in sorted(mix_df["fold"].unique()):
        eval_rows = mix_df[mix_df["fold"] == k].reset_index(drop=True)
        hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=k)

        separate_fn = fit_and_separate_fn(hs_allowed, ls_allowed)
        result = evaluate_dataset(separate_fn, mix_df=eval_rows, compute_permutation=compute_permutation)
        result["fold"] = k
        fold_results.append(result)

    results_df = pd.concat(fold_results, ignore_index=True)
    fold_summary = aggregate_by_fold(results_df)
    cv_summary = aggregate_across_folds(fold_summary)
    return results_df, fold_summary, cv_summary


def aggregate_by_fold(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean sdr/sir/sar per (fold, source)."""
    return results_df.groupby(["fold", "source"])[["sdr", "sir", "sar"]].mean().reset_index()


def summarize_pooled(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and median sdr/sir/sar per source, pooled directly over every row
    in results_df (i.e. every evaluated mixture individually), not averaged
    within folds first like aggregate_by_fold/aggregate_across_folds does.

    Distinct from cv_summary's across-fold dispersion: per
    TEEP2026_Sprint0_Review, averaging within folds before taking std understates
    the true row-to-row spread (e.g. an observed 6.46+/-3.71 dB across-fold
    figure vs. 6.30+/-39.38 dB pooled across rows for the same data) -- the
    two are not interchangeable, and a report should say which one it means.
    This function's numbers are the row-level ones; cv_summary's are the
    fold-level ones.
    """
    return results_df.groupby("source")[["sdr", "sir", "sar"]].agg(["mean", "median", "std"])


def aggregate_across_folds(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Mean +/- std across folds per source -- the CV estimate of how the
    model generalizes to held-out (leakage-free) mixtures, with std
    reflecting fold-to-fold variance rather than pooling all rows together.
    """
    return (
        fold_summary.groupby("source")[["sdr", "sir", "sar"]]
        .agg(["mean", "std"])
        .reindex(SOURCE_LABELS)
    )


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    def identity_baseline(_hs_allowed, _ls_allowed):
        def separate(mixed, _sr):
            return mixed, mixed

        return separate

    print("Running eval harness plumbing smoke test (no-separation baseline, 5 folds)...")
    results_df, fold_summary, cv_summary = cross_validate(identity_baseline, n_folds=5, seed=0)
    print(f"Evaluated {len(results_df)} (fold, mix row, source) rows across {results_df['fold'].nunique()} folds")

    stat_tiles = "\n".join([
        stat_tile("Evaluations", str(len(results_df)), "rows"),
        stat_tile("Folds", str(results_df["fold"].nunique()), "folds"),
        stat_tile("Sources", str(len(SOURCE_LABELS)), " / ".join(SOURCE_LABELS)),
    ])

    body = "\n\n".join([
        section(
            "Per-fold means",
            "mean sdr/sir/sar per (fold, source)",
            df_to_html(fold_summary.set_index(["fold", "source"]), index_label="fold / source"),
        ),
        section(
            "Across-fold mean ± std",
            "headline CV result (no-separation baseline, plumbing check only)",
            df_to_html(cv_summary, index_label="source"),
        ),
    ])

    html = report_shell(
        title="Eval Harness Smoke Test",
        eyebrow="HLS-CMDS · k-fold plumbing check",
        heading="No-separation baseline through the full harness",
        dek=(
            "Smoke test only — separate_fn just returns the mixture unchanged, so "
            "this exercises fold assignment + BSS Eval wiring end-to-end, not "
            "separation quality."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer="<p><strong>Method.</strong> See <code>eval_harness.py</code>'s <code>cross_validate()</code>.</p>",
    )

    report_path = write_report(results_dir() / "eval_harness_smoke_test.html", html)
    print(f"Report written to {report_path}")
