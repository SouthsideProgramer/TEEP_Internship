"""
BSS Eval separation metrics (SDR / SIR / SAR) for the heart/lung mix set.

See code_description.md for details on the mir_eval wrapper and why the
FutureWarning below is silenced.

Usage:
    from metrics import evaluate_heart_lung, evaluate_dataset

    metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
    # {"heart": {"sdr": .., "sir": .., "sar": ..}, "lung": {...}}

    # Or run a separation function over the whole Mix.csv and aggregate by class:
    results_df = evaluate_dataset(my_separate_fn)
    summarize_by_class(results_df, "heart", "Heart Sound Type")
"""
import warnings

import numpy as np
import pandas as pd
from mir_eval.separation import bss_eval_sources

from load_dataset import load_audio, load_mix

SOURCE_LABELS = ("heart", "lung")


def bss_eval(reference_sources, estimated_sources, compute_permutation=True) -> dict:
    """
    Raw BSS Eval call for an arbitrary number of sources.

    reference_sources, estimated_sources: array-like, shape (n_sources, n_samples)
        (a single 1-D source is also accepted and treated as n_sources=1)

    Returns {"sdr": array, "sir": array, "sar": array, "perm": array}, one
    value per source, ordered to match `reference_sources` (see `perm` if
    `compute_permutation=True`).
    """
    reference_sources = np.atleast_2d(reference_sources)
    estimated_sources = np.atleast_2d(estimated_sources)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sdr, sir, sar, perm = bss_eval_sources(
            reference_sources, estimated_sources, compute_permutation=compute_permutation
        )
    return {"sdr": sdr, "sir": sir, "sar": sar, "perm": perm}


def evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est, compute_permutation=True) -> dict:
    """
    BSS Eval for the heart/lung 2-source case.

    All four arrays must be 1-D and the same length; a length mismatch (e.g.
    from a reconstruction that doesn't preserve exact sample count) is
    trimmed to the shortest of the four rather than raising.

    Returns {"heart": {"sdr": float, "sir": float, "sar": float},
             "lung":  {"sdr": float, "sir": float, "sar": float}}
    """
    heart_ref, lung_ref, heart_est, lung_est = (
        np.asarray(a) for a in (heart_ref, lung_ref, heart_est, lung_est)
    )
    n = min(len(heart_ref), len(lung_ref), len(heart_est), len(lung_est))

    reference = np.stack([heart_ref[:n], lung_ref[:n]])
    estimated = np.stack([heart_est[:n], lung_est[:n]])
    result = bss_eval(reference, estimated, compute_permutation=compute_permutation)

    return {
        label: {"sdr": float(result["sdr"][i]), "sir": float(result["sir"][i]), "sar": float(result["sar"][i])}
        for i, label in enumerate(SOURCE_LABELS)
    }


def evaluate_mix_row(row, separate_fn, compute_permutation=True) -> dict:
    """
    Evaluate one Mix.csv row (as produced by load_dataset.load_mix()).

    separate_fn: callable(mixed: np.ndarray, sr: int) -> (heart_est, lung_est)
        the separation model / algorithm under test.
    """
    heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
    lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
    mixed, _ = load_audio(row["mixed_audio_path"], sr=None)

    heart_est, lung_est = separate_fn(mixed, sr)
    metrics = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est, compute_permutation=compute_permutation)
    metrics["mixed_id"] = row["Mixed Sound ID"]
    return metrics


def evaluate_dataset(separate_fn, mix_df=None, compute_permutation=True) -> pd.DataFrame:
    """
    Run separate_fn over every Mix.csv triplet.

    Returns a tidy DataFrame with one row per (Mixed Sound ID, source):
    Mixed Sound ID, source, Heart Sound Type, Lung Sound Type, Location, sdr, sir, sar.
    """
    mix_df = mix_df if mix_df is not None else load_mix()
    rows = []
    for _, row in mix_df.iterrows():
        result = evaluate_mix_row(row, separate_fn, compute_permutation=compute_permutation)
        for source in SOURCE_LABELS:
            rows.append({
                "Mixed Sound ID": row["Mixed Sound ID"],
                "source": source,
                "Heart Sound Type": row["Heart Sound Type"],
                "Lung Sound Type": row["Lung Sound Type"],
                "Location": row["Location"],
                "sdr": result[source]["sdr"],
                "sir": result[source]["sir"],
                "sar": result[source]["sar"],
            })
    return pd.DataFrame(rows)


def summarize_by_class(results_df: pd.DataFrame, source: str, class_col: str) -> pd.DataFrame:
    """Mean/std/count SDR/SIR/SAR for one source ('heart' or 'lung'), grouped by class_col."""
    subset = results_df[results_df["source"] == source]
    return subset.groupby(class_col)[["sdr", "sir", "sar"]].agg(["mean", "std", "count"])


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    # Smoke test against real dataset audio: two baselines with known-sane behavior.
    #   - identity separation (est == ref)   -> SDR/SIR/SAR should be very high (~inf)
    #   - mixed signal used as both estimates -> SDR should be low/negative (no separation at all)
    print("Running metrics smoke test against real dataset audio...")
    mix_df = load_mix()
    row = mix_df.iloc[0]
    heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
    lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)

    def identity_separate(mixed, sr):
        return heart_ref, lung_ref

    def no_separation(mixed, sr):
        return mixed, mixed

    def _metrics_df(m: dict) -> pd.DataFrame:
        return pd.DataFrame([{"source": s, **m[s]} for s in SOURCE_LABELS]).set_index("source")

    identity_metrics = evaluate_mix_row(row, identity_separate)
    no_sep_metrics = evaluate_mix_row(row, no_separation)

    stat_tiles = "\n".join([
        stat_tile("Test row", str(row["Mixed Sound ID"]), "Mixed Sound ID"),
        stat_tile("Heart type", row["Heart Sound Type"], ""),
        stat_tile("Lung type", row["Lung Sound Type"], ""),
    ])

    body = "\n\n".join([
        section(
            "Identity separation (est == ref)",
            "expected: very high SDR/SIR/SAR",
            df_to_html(_metrics_df(identity_metrics), index_label="source", float_fmt="{:.1f}"),
        ),
        section(
            "No separation (est == mixed for both)",
            "expected: low/negative SDR",
            df_to_html(_metrics_df(no_sep_metrics), index_label="source", float_fmt="{:.1f}"),
        ),
    ])

    html = report_shell(
        title="Metrics Smoke Test",
        eyebrow="HLS-CMDS · BSS Eval sanity check",
        heading="Identity vs. no-separation baseline",
        dek=(
            "Two known-sane behaviors on one real Mix.csv row: identity separation "
            "should score near-perfect SDR/SIR/SAR, no separation at all should score "
            "low/negative SDR."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer="<p><strong>Method.</strong> See <code>metrics.py</code>'s <code>evaluate_mix_row()</code>.</p>",
    )

    report_path = write_report(results_dir() / "metrics_smoke_test.html", html)
    print(f"Report written to {report_path}")
