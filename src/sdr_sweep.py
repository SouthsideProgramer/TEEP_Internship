"""
S6-02: generate the controlled SDR sweep dataset -- applies S6-01's
degradation scheme (degradation.py) to every one of the six real
separation baselines' own outputs (S4-03) on the native additive rows,
producing the swept dataset Sprint 6's classification-and-plotting step
will consume directly. Pulled forward from S6 into S5 alongside S6-01,
deliberately, so the holiday-shortened Sprint 6 (loses two working days --
25 Sep Fri and 28 Sep Mon) is left for measurement and plotting only, not
also generation. Ref code kept as S6-02.

Does not depend on the classifier (PROTOCOL.md Sec. 5.3) at all -- only on
the six separation baselines' own outputs -- so it runs independently of,
and in parallel with, S5-01 (the classifier itself).

Substrate: the 36 native additive Mix.csv rows (verify_additive_triplets),
same substrate degradation.py's own validation used -- real audio, and
already confirmed cheap enough to run every baseline (including EVMD) on
in full (BACKLOG.md's Baseline 5 entry). The synthetic set could extend
this later but is out of scope here.

Design, mirroring synthetic_mix.py's own split between "cheap provenance
table" and "audio reconstructed on demand": separation itself (the
expensive part, especially Baselines 2/5/6) is run exactly once per
(baseline, fold) and each row's real heart_est/lung_est is cached to disk
as .wav (cheap to store -- roughly 120 KB/recording, 6 baselines x 36 rows
x 2 sources ~= 50 MB total), never re-run per target-SDR grid point.
Degradation itself (degrade_toward_ground_truth) is cheap and
reproducible from that cache plus a stored alpha, so it is reconstructed
on demand (synthesize_sweep_row()) rather than also materializing every
degraded waveform to disk.

Target grid: 25 dB down to -5 dB, anchored above every baseline's own
measured range on this dataset (PROTOCOL.md Sec. 5.2's additive-subset
tables cluster in roughly 0-8 dB) so most grid points land as genuinely
interpolated (alpha < 1), not clamped to a baseline's own floor. See
degradation.py's docstring for why the sweep cannot go *below* a
baseline's own real SDR (alpha=1 is that method's real output; the scheme
interpolates toward it, not past it) -- clamped_to_baseline_floor flags
exactly this per row.

Usage:
    from sdr_sweep import build_sdr_sweep, synthesize_sweep_row

    sweep_df = build_sdr_sweep()  # provenance table + cached .wav estimates under results/sdr_sweep_cache/
    degraded, sr = synthesize_sweep_row(sweep_df.iloc[0])
"""
from pathlib import Path

import pandas as pd
import soundfile as sf

from degradation import degrade_row, degrade_toward_ground_truth, find_alphas_for_target_sdrs
from load_dataset import load_audio, load_hs, load_ls, load_mix, verify_additive_triplets
from metrics import SOURCE_LABELS
from report_utils import results_dir
from split import assign_folds, dictionary_pool

TARGET_SDR_GRID_DB = (25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0)
N_FOLDS = 5
SWEEP_SEED = 0

BASELINE_LABELS = (
    "Baseline 1 (bandpass)",
    "Baseline 2 (supervised NMF)",
    "Baseline 3 (standard NMF)",
    "Baseline 4 (MSSA)",
    "Baseline 5 (EVMD)",
    "Baseline 6 (Conv-TasNet-lite)",
)


def default_baseline_specs(seed: int = SWEEP_SEED) -> list[tuple[str, callable]]:
    """(label, fit_and_separate_fn) for all six separation baselines (S4-03) --
    not Baseline 0 (no separation at all), which has nothing to interpolate
    toward. Imports are local so this module (and anything that only needs
    synthesize_sweep_row()/the provenance schema) doesn't pay for torch/EVMD
    unless a sweep is actually being generated."""
    from baseline.baseline1 import fit_bandpass_baseline
    from baseline.baseline2 import make_supervised_nmf_baseline
    from baseline.baseline3 import make_standard_nmf_baseline
    from baseline.baseline4 import fit_ssa_baseline
    from baseline.baseline5 import fit_evmd_baseline
    from convtasnet import make_convtasnet_baseline

    fit_fns = [
        fit_bandpass_baseline,
        make_supervised_nmf_baseline(seed=seed),
        make_standard_nmf_baseline(seed=seed),
        fit_ssa_baseline,
        fit_evmd_baseline,
        make_convtasnet_baseline(seed=seed),
    ]
    return list(zip(BASELINE_LABELS, fit_fns))


def cache_dir() -> Path:
    d = results_dir() / "sdr_sweep_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_label(baseline_label: str) -> str:
    return baseline_label.split(" (")[0].replace(" ", "")


def estimate_path(root: Path, baseline_label: str, mixed_id: str, source: str) -> Path:
    """Where a baseline's real (undegraded) separated estimate for one
    mixture/source is cached -- public so condition_b.py can load it
    directly without going through synthesize_sweep_row()'s degradation."""
    return root / _safe_label(baseline_label) / f"{mixed_id}_{source}.wav"


def native_additive_rows() -> pd.DataFrame:
    """The 36 Mix.csv rows where mixed ~= a*(heart+lung) actually holds -- see
    load_dataset.verify_additive_triplets. Same substrate degradation.py's
    own validation and every other baseline's "additive-only" table use."""
    mix_df = load_mix()
    additivity = verify_additive_triplets(mix_df)
    return mix_df[mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)


def _target_grid_rows(
    label: str, fold: int, mix_row: pd.Series, heart_ref, lung_ref, heart_est, lung_est, target_grid, sources
) -> list[dict]:
    """Shared by build_sdr_sweep() and build_provenance_from_cache(): given
    one (baseline, row)'s ground truth + separated estimate, hit every
    target_grid value for every source and return the provenance rows."""
    rows = []
    for source in sources:
        alphas_by_target = find_alphas_for_target_sdrs(heart_ref, lung_ref, heart_est, lung_est, target_grid, source=source)
        for target in target_grid:
            alpha = alphas_by_target[float(target)]
            measured = degrade_row(heart_ref, lung_ref, heart_est, lung_est, alpha, source)[source]
            rows.append({
                "baseline": label,
                "fold": int(fold),
                "mixed_id": mix_row["Mixed Sound ID"],
                "source": source,
                "Heart Sound Type": mix_row["Heart Sound Type"],
                "Lung Sound Type": mix_row["Lung Sound Type"],
                "target_sdr": float(target),
                "alpha": alpha,
                "achieved_sdr": measured["sdr"],
                "achieved_sir": measured["sir"],
                "achieved_sar": measured["sar"],
                "clamped_to_baseline_floor": bool(alpha >= 1.0 - 1e-9),
            })
    return rows


def build_sdr_sweep(
    n_folds: int = N_FOLDS,
    seed: int = SWEEP_SEED,
    target_grid=TARGET_SDR_GRID_DB,
    sources=SOURCE_LABELS,
    mix_df: pd.DataFrame | None = None,
    baseline_specs: list[tuple[str, callable]] | None = None,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """
    Run every separation baseline once (leak-safe, one fit per fold) over
    `mix_df` (defaults to the 36 native additive rows), caching each
    baseline's real heart_est/lung_est to `cache_root` (defaults to
    results/sdr_sweep_cache/), then apply S6-01's degradation scheme
    (find_alpha_for_target_sdr) to hit every target_grid value for each
    (baseline, row, source). Returns the swept dataset's provenance table
    -- see synthesize_sweep_row() to reconstruct the actual degraded audio
    from it.

    mix_df/baseline_specs are overridable so a caller (e.g. a test) can pass
    a tiny subset instead of the full 36-row x 6-baseline sweep.
    """
    mix_df = mix_df if mix_df is not None else native_additive_rows()
    baseline_specs = baseline_specs if baseline_specs is not None else default_baseline_specs(seed=seed)
    root = cache_root if cache_root is not None else cache_dir()

    mix_df = assign_folds(mix_df=mix_df, n_folds=n_folds, seed=seed)
    hs_df, ls_df = load_hs(), load_ls()

    rows = []
    for label, fit_and_separate_fn in baseline_specs:
        for k in sorted(mix_df["fold"].unique()):
            eval_rows = mix_df[mix_df["fold"] == k]
            hs_allowed, ls_allowed = dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=k)
            separate_fn = fit_and_separate_fn(hs_allowed, ls_allowed)

            for _, row in eval_rows.iterrows():
                heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
                lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
                mixed, _ = load_audio(row["mixed_audio_path"], sr=None)
                heart_est, lung_est = separate_fn(mixed, sr)

                heart_path = estimate_path(root, label, row["Mixed Sound ID"], "heart")
                lung_path = estimate_path(root, label, row["Mixed Sound ID"], "lung")
                heart_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(heart_path, heart_est, sr)
                sf.write(lung_path, lung_est, sr)

                rows.extend(_target_grid_rows(label, k, row, heart_ref, lung_ref, heart_est, lung_est, target_grid, sources))

    return pd.DataFrame(rows)


def build_provenance_from_cache(
    mix_df: pd.DataFrame,
    baseline_labels,
    cache_root: Path | None = None,
    target_grid=TARGET_SDR_GRID_DB,
    sources=SOURCE_LABELS,
    n_folds: int = N_FOLDS,
    seed: int = SWEEP_SEED,
) -> pd.DataFrame:
    """
    Like build_sdr_sweep(), but skips separation entirely -- reads each
    baseline's already-cached heart_est/lung_est .wav files (written by a
    prior build_sdr_sweep() call) and only (re)computes the target-SDR
    grid's alpha/achieved-metric rows.

    Built so provenance can be assembled incrementally per baseline as
    each one's cache finishes, rather than only after every baseline in
    one build_sdr_sweep() call has completed -- e.g. Baseline 1's sweep can
    be measured while Baselines 2-6 are still generating in a separate,
    longer-running build_sdr_sweep() process.

    mix_df must already carry a 'fold' column (assign_folds()'s output) --
    pass the exact same mix_df/n_folds/seed build_sdr_sweep() used, or the
    (baseline, mixed_id) -> fold mapping recorded in the returned rows
    won't match what the cache was actually fit against.
    """
    root = cache_root if cache_root is not None else cache_dir()
    if "fold" not in mix_df.columns:
        mix_df = assign_folds(mix_df=mix_df, n_folds=n_folds, seed=seed)

    rows = []
    for label in baseline_labels:
        for _, row in mix_df.iterrows():
            heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
            lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
            heart_est, _ = load_audio(str(estimate_path(root, label, row["Mixed Sound ID"], "heart")), sr=None)
            lung_est, _ = load_audio(str(estimate_path(root, label, row["Mixed Sound ID"], "lung")), sr=None)

            rows.extend(
                _target_grid_rows(label, row["fold"], row, heart_ref, lung_ref, heart_est, lung_est, target_grid, sources)
            )

    return pd.DataFrame(rows)


def synthesize_sweep_row(
    row: pd.Series, mix_df: pd.DataFrame | None = None, cache_root: Path | None = None
) -> tuple:
    """
    Reconstruct the actual degraded audio for one build_sdr_sweep() row: the
    cheap part (degrade_toward_ground_truth) rebuilt on demand from the
    cached separated estimate (the expensive part, already run once by
    build_sdr_sweep) plus the row's own alpha. Returns (degraded_audio, sr).
    """
    mix_df = mix_df if mix_df is not None else native_additive_rows()
    root = cache_root if cache_root is not None else cache_dir()

    mix_row = mix_df[mix_df["Mixed Sound ID"] == row["mixed_id"]].iloc[0]
    ref_col = "heart_audio_path" if row["source"] == "heart" else "lung_audio_path"
    ground_truth, sr = load_audio(mix_row[ref_col], sr=None)
    est_path = estimate_path(root, row["baseline"], row["mixed_id"], row["source"])
    estimate, _ = load_audio(str(est_path), sr=None)
    return degrade_toward_ground_truth(ground_truth, estimate, row["alpha"]), sr


if __name__ == "__main__":
    import time

    import numpy as np

    from report_utils import df_to_html, report_shell, results_dir as _results_dir, section, stat_tile, write_report

    print("Loading native additive rows (S4-03 substrate)...")
    mix_df = native_additive_rows()
    print(f"  {len(mix_df)} rows")

    print(f"Generating the SDR sweep across all 6 separation baselines (target grid: {TARGET_SDR_GRID_DB} dB)...")
    t0 = time.time()
    sweep_df = build_sdr_sweep(n_folds=N_FOLDS, seed=SWEEP_SEED, mix_df=mix_df)
    elapsed = time.time() - t0
    print(f"  {len(sweep_df)} sweep rows generated in {elapsed:.1f}s")

    out_csv = _results_dir() / "sdr_sweep_provenance.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"  provenance written to {out_csv}")

    print("Self-consistency check: reconstructing a handful of rows via synthesize_sweep_row() and re-measuring...")
    from metrics import evaluate_heart_lung

    check_rows = sweep_df.sample(n=min(20, len(sweep_df)), random_state=0)
    reconstruction_errors = []
    for _, row in check_rows.iterrows():
        degraded, sr = synthesize_sweep_row(row, mix_df=mix_df)
        mix_row = mix_df[mix_df["Mixed Sound ID"] == row["mixed_id"]].iloc[0]
        heart_ref, _ = load_audio(mix_row["heart_audio_path"], sr=None)
        lung_ref, _ = load_audio(mix_row["lung_audio_path"], sr=None)
        other_source = "lung" if row["source"] == "heart" else "heart"
        other_ref = lung_ref if row["source"] == "heart" else heart_ref
        remeasured = evaluate_heart_lung(
            heart_ref, lung_ref, degraded if row["source"] == "heart" else other_ref,
            degraded if row["source"] == "lung" else other_ref,
        )[row["source"]]["sdr"]
        reconstruction_errors.append(abs(remeasured - row["achieved_sdr"]))
    max_reconstruction_error = float(np.max(reconstruction_errors))
    print(f"  max |re-measured - recorded| SDR across {len(check_rows)} sampled rows: {max_reconstruction_error:.4f} dB")

    sweep_df["abs_error_db"] = (sweep_df["achieved_sdr"] - sweep_df["target_sdr"]).abs()
    non_clamped = sweep_df[~sweep_df["clamped_to_baseline_floor"]]

    coverage = sweep_df.groupby(["baseline", "source"]).agg(
        n=("target_sdr", "size"), n_clamped=("clamped_to_baseline_floor", "sum")
    )
    coverage["pct_clamped"] = coverage["n_clamped"] / coverage["n"]
    coverage["mean_abs_error_db"] = non_clamped.groupby(["baseline", "source"])["abs_error_db"].mean()

    stat_tiles = "\n".join([
        stat_tile("Baselines", "6", "bandpass, sup. NMF, std. NMF, MSSA, EVMD, Conv-TasNet-lite"),
        stat_tile("Rows generated", str(len(sweep_df)), f"{len(mix_df)} mixtures x 2 sources x {len(TARGET_SDR_GRID_DB)} targets"),
        stat_tile("Reconstruction check", f"{max_reconstruction_error:.3f} dB", "max error, 20 sampled rows", ok=max_reconstruction_error < 0.1),
    ])

    body = "\n\n".join([
        section(
            "Target-SDR coverage per baseline / source",
            "n rows, how many clamped to that baseline's own real SDR (alpha=1), mean |achieved-target| for the rest",
            df_to_html(coverage, index_label="baseline / source", float_fmt="{:.2f}"),
        ),
        section(
            "Reconstruction self-check",
            "synthesize_sweep_row() re-measured against the recorded achieved_sdr, 20 sampled rows",
            f'<p class="note">Max absolute discrepancy: {max_reconstruction_error:.4f} dB (should be ~0 -- '
            f"this checks the cache-then-reconstruct pipeline is faithful, not just that the sweep ran).</p>",
        ),
    ])

    html = report_shell(
        title="SDR Sweep Generation",
        eyebrow="HLS-CMDS · S6-02 · controlled SDR sweep dataset",
        heading="Generated SDR sweep: 6 baselines x 36 native additive rows",
        dek=(
            "Provenance table for Sprint 6's classification-and-plotting step (results/"
            "sdr_sweep_provenance.csv) -- does not touch the classifier itself. Real "
            "heart_est/lung_est audio is cached once per (baseline, fold) under results/"
            "sdr_sweep_cache/; synthesize_sweep_row() reconstructs each degraded waveform "
            "on demand from that cache plus the row's own alpha."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> See <code>sdr_sweep.py</code>'s module docstring and "
            "<code>degradation.py</code> (S6-01) / PROTOCOL.md Sec. 5.3.1 for the degradation "
            "scheme this builds on.</p>"
        ),
    )

    report_path = write_report(_results_dir() / "sdr_sweep_report.html", html)
    print(f"\nReport written to {report_path}")
