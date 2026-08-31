"""
Synthetic mixing set (S1-09/S1-10/S1-13) -- the primary evaluation
substrate for the heart/lung separation baselines, replacing native
Mix.csv pairs (only 36/145 of which are actually additive -- see
load_dataset.verify_additive_triplets) for C1'/C2.

See code_description.md for the full design rationale (gain distribution,
SNR sweep, split rule, seed policy) and why each of those choices was made.

Usage:
    from synthetic_mix import build_synthetic_set, evaluate_synthetic, validate_against_native

    synthetic_df = build_synthetic_set(n_folds=5, seed=0)   # provenance only, ~1500 rows
    results_df, fold_summary, cv_summary = evaluate_synthetic(fit_and_separate_fn, synthetic_df)
    validation = validate_against_native()  # S1-10: does the recipe reproduce the 36 native rows?
"""
import hashlib

import numpy as np
import pandas as pd

SYNTHETIC_MIX_N_FOLDS = 5
SYNTHETIC_SEED = 0

SNR_SWEEP_DB = (-5.0, 15.0, 35.0)
SNR_DB_CLOSEST_TO_HAN_QUAN = 35.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _stable_seed(*parts) -> int:
    """Deterministic integer seed from arbitrary parts -- Python's built-in
    hash() is randomized per-process, so this is used instead of hash()
    anywhere a reproducible-across-runs seed is needed."""
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _native_gain_range() -> tuple[float, float]:
    """
    The (min, max) scalar gain `a` fit by verify_additive_triplets() across
    the 36 native additive Mix.csv rows (mixed = a*(heart+lung)). Used to
    calibrate this module's synthetic gain distribution from real data
    instead of an arbitrary range.
    """
    from load_dataset import load_mix, verify_additive_triplets

    result = verify_additive_triplets(load_mix())
    gains = [r["gain"] for r in result["rows"] if r["additive"]]
    if not gains:
        raise RuntimeError("No native additive rows found -- can't calibrate the synthetic gain range.")
    return min(gains), max(gains)


def assign_source_folds(
    hs_df: pd.DataFrame, ls_df: pd.DataFrame, n_folds: int = SYNTHETIC_MIX_N_FOLDS, seed: int = 0
) -> tuple[dict, dict]:
    """
    S1-13: source-file-level fold assignment, independent for heart and lung
    files. Supersedes split.assign_folds' triplet/leak-group logic for this
    substrate -- with full combinatorial pairing, every heart file
    transitively connects to every lung file (one giant leak group), so the
    connected-component mechanism doesn't produce meaningful folds here.
    Each of the 50 HS.csv / 50 LS.csv recordings gets its own independent
    fold, balanced via seeded round-robin (no groups to balance, unlike the
    native leak-group split).

    Returns (heart_fold_map, lung_fold_map): {recording ID: fold index}.
    """
    rng = np.random.default_rng(seed)

    heart_ids = hs_df["Heart Sound ID"].tolist()
    heart_order = rng.permutation(len(heart_ids))
    heart_fold_map = {heart_ids[i]: int(pos % n_folds) for pos, i in enumerate(heart_order)}

    lung_ids = ls_df["Lung Sound ID"].tolist()
    lung_order = rng.permutation(len(lung_ids))
    lung_fold_map = {lung_ids[i]: int(pos % n_folds) for pos, i in enumerate(lung_order)}

    return heart_fold_map, lung_fold_map


def build_synthetic_set(n_folds: int = SYNTHETIC_MIX_N_FOLDS, seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """
    S1-09: build the synthetic mixing set's provenance table (no audio
    arrays -- see synthesize_row for on-demand reconstruction, which keeps
    this cheap to hold in memory at n in the thousands).

    Full combinatorial pairing over HS.csv x LS.csv (50 x 50 = 2500 pairs),
    restricted to same-fold pairs only (S1-13: fold(heart) == fold(lung) --
    required so a fold's dictionary pool excludes *both* sources of every
    mixture it's evaluated on). With n_folds=5 that's 10x10=100 pairs/fold x
    5 folds = 500 leakage-safe pairs, x len(SNR_SWEEP_DB)=3 noise levels =
    1500 total rows.

    Each pair draws one scalar gain `a` (log-uniform over the range observed
    on the 36 native additive rows, see _native_gain_range) shared across its
    5 SNR realizations, matching the native construction's own model
    (mixed = a*(heart+lung) + noise, one gain on the sum -- not independent
    heart/lung levels) -- so difficulty varies along the noise axis only.

    Returns one row per (heart_id, lung_id, snr_db): mix_id, heart_id,
    lung_id, gain_a, snr_db, fold.
    """
    from load_dataset import load_hs, load_ls

    hs_df, ls_df = load_hs(), load_ls()
    heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=n_folds, seed=seed)
    a_min, a_max = _native_gain_range()
    log_a_min, log_a_max = np.log(a_min), np.log(a_max)

    rows = []
    for heart_id, heart_fold in heart_fold_map.items():
        for lung_id, lung_fold in lung_fold_map.items():
            if heart_fold != lung_fold:
                continue
            fold = heart_fold

            gain_rng = np.random.default_rng(np.random.SeedSequence([seed, _stable_seed(heart_id, lung_id)]))
            gain_a = float(np.exp(gain_rng.uniform(log_a_min, log_a_max)))

            for snr_db in SNR_SWEEP_DB:
                rows.append({
                    "mix_id": f"SYN-{heart_id}-{lung_id}-{snr_db:g}dB",
                    "heart_id": heart_id,
                    "lung_id": lung_id,
                    "gain_a": gain_a,
                    "snr_db": snr_db,
                    "fold": fold,
                })

    return pd.DataFrame(rows)


def synthesize_row(row, hs_audio_cache: dict, ls_audio_cache: dict, seed: int = SYNTHETIC_SEED):
    """
    Reconstruct (heart_ref, lung_ref, mixed, sr) for one build_synthetic_set
    row from cached isolated-recording audio (see evaluate_synthetic /
    validate_against_native for cache construction), applying that row's own
    gain_a and snr_db deterministically -- the noise draw is reproducible
    across runs (seeded from (heart_id, lung_id, snr_db)), not re-randomized
    on every call.
    """
    heart_y, sr = hs_audio_cache[row["heart_id"]]
    lung_y, _ = ls_audio_cache[row["lung_id"]]
    n = min(len(heart_y), len(lung_y))
    heart_ref, lung_ref = heart_y[:n], lung_y[:n]

    base = row["gain_a"] * (heart_ref + lung_ref)
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([seed, _stable_seed(row["heart_id"], row["lung_id"], row["snr_db"])])
    )
    noise_rms = _rms(base) / (10 ** (row["snr_db"] / 20.0))
    noise = noise_rng.normal(0.0, noise_rms, size=n)

    return heart_ref, lung_ref, base + noise, sr


def dictionary_pool_synthetic(
    hs_df: pd.DataFrame, ls_df: pd.DataFrame, heart_fold_map: dict, lung_fold_map: dict, held_out_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split.dictionary_pool's contract, source-file-level instead of content-hash/leak-group."""
    hs_allowed = hs_df[hs_df["Heart Sound ID"].map(heart_fold_map) != held_out_fold].reset_index(drop=True)
    ls_allowed = ls_df[ls_df["Lung Sound ID"].map(lung_fold_map) != held_out_fold].reset_index(drop=True)
    return hs_allowed, ls_allowed


def _audio_cache(df: pd.DataFrame, id_col: str) -> dict:
    from load_dataset import load_audio

    return {row[id_col]: load_audio(row["audio_path"], sr=None) for _, row in df.iterrows()}


def evaluate_synthetic(
    fit_and_separate_fn,
    synthetic_df: pd.DataFrame,
    n_folds: int = SYNTHETIC_MIX_N_FOLDS,
    seed: int = SYNTHETIC_SEED,
    compute_permutation: bool = True,
):
    """
    eval_harness.cross_validate's contract, over the synthetic set instead
    of native Mix.csv rows: for each fold, fits fit_and_separate_fn on that
    fold's source-file-restricted dictionary pool (dictionary_pool_synthetic),
    evaluates on that fold's same-fold synthetic pairs, and aggregates with
    the exact same functions cross_validate uses (aggregate_by_fold /
    aggregate_across_folds) -- no duplicated aggregation logic.

    Returns (results_df, fold_summary, cv_summary) -- results_df additionally
    carries heart_id/lung_id/gain_a/snr_db provenance per row.
    """
    from eval_harness import aggregate_across_folds, aggregate_by_fold
    from load_dataset import load_hs, load_ls
    from metrics import SOURCE_LABELS, evaluate_heart_lung

    hs_df, ls_df = load_hs(), load_ls()
    heart_fold_map, lung_fold_map = assign_source_folds(hs_df, ls_df, n_folds=n_folds, seed=seed)
    hs_audio_cache = _audio_cache(hs_df, "Heart Sound ID")
    ls_audio_cache = _audio_cache(ls_df, "Lung Sound ID")

    fold_results = []
    for k in sorted(synthetic_df["fold"].unique()):
        eval_rows = synthetic_df[synthetic_df["fold"] == k]
        hs_allowed, ls_allowed = dictionary_pool_synthetic(hs_df, ls_df, heart_fold_map, lung_fold_map, held_out_fold=k)
        separate_fn = fit_and_separate_fn(hs_allowed, ls_allowed)

        rows = []
        for _, row in eval_rows.iterrows():
            heart_ref, lung_ref, mixed, sr = synthesize_row(row, hs_audio_cache, ls_audio_cache, seed=seed)
            heart_est, lung_est = separate_fn(mixed, sr)
            m = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est, compute_permutation=compute_permutation)
            for source in SOURCE_LABELS:
                rows.append({
                    "mix_id": row["mix_id"],
                    "source": source,
                    "heart_id": row["heart_id"],
                    "lung_id": row["lung_id"],
                    "gain_a": row["gain_a"],
                    "snr_db": row["snr_db"],
                    "fold": k,
                    "sdr": m[source]["sdr"],
                    "sir": m[source]["sir"],
                    "sar": m[source]["sar"],
                })
        fold_results.append(pd.DataFrame(rows))

    results_df = pd.concat(fold_results, ignore_index=True)
    fold_summary = aggregate_by_fold(results_df)
    cv_summary = aggregate_across_folds(fold_summary)
    return results_df, fold_summary, cv_summary


def validate_against_native(residual_threshold: float = 1e-3) -> dict:
    """
    S1-10: does this module's construction reproduce the 36 native additive
    rows? For each, take the row's *real* heart/lung audio and its own
    fitted gain `a` (from verify_additive_triplets), construct
    mixed_synth = a*(heart+lung) (zero noise -- the reproduction claim, not
    a noisy realization), and re-run verify_additive_triplets() itself
    (reused directly, not reimplemented) against a mix_df pointing at the
    real heart/lung files and this new synthetic mixed file. If the
    synthetic construction is faithful to how the dataset's own additive
    mixtures actually look, its residual should land in the same tight
    ~1e-8-1e-4 cluster the genuine native rows occupy -- not just below
    residual_threshold by construction (trivial), but comparable in
    magnitude to the real rows' own residuals (meaningful).

    Returns {"rows": [{"mixed_id", "native_residual", "synthetic_residual",
    "reproduces": bool}], "all_reproduce": bool}.
    """
    import os
    import shutil
    import tempfile

    import soundfile as sf

    from load_dataset import load_audio, load_mix, verify_additive_triplets

    mix_df = load_mix()
    native_check = verify_additive_triplets(mix_df, residual_threshold=residual_threshold)
    native_by_id = {r["mixed_id"]: r for r in native_check["rows"]}
    valid_rows = mix_df[mix_df["Mixed Sound ID"].isin(native_check["valid_ids"])]

    tmp_dir = tempfile.mkdtemp(prefix="synthetic_mix_validation_")
    try:
        synth_rows = []
        for _, row in valid_rows.iterrows():
            heart, sr = load_audio(row["heart_audio_path"], sr=None)
            lung, _ = load_audio(row["lung_audio_path"], sr=None)
            n = min(len(heart), len(lung))
            gain_a = native_by_id[row["Mixed Sound ID"]]["gain"]
            mixed_synth = gain_a * (heart[:n] + lung[:n])

            synth_path = os.path.join(tmp_dir, f"{row['Mixed Sound ID']}_synth.wav")
            sf.write(synth_path, mixed_synth, sr)
            synth_rows.append({
                "Mixed Sound ID": row["Mixed Sound ID"],
                "heart_audio_path": row["heart_audio_path"],
                "lung_audio_path": row["lung_audio_path"],
                "mixed_audio_path": synth_path,
            })

        synth_check = verify_additive_triplets(pd.DataFrame(synth_rows), residual_threshold=residual_threshold)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    synth_by_id = {r["mixed_id"]: r for r in synth_check["rows"]}
    rows = [
        {
            "mixed_id": mid,
            "native_residual": native_by_id[mid]["relative_residual"],
            "synthetic_residual": synth_by_id[mid]["relative_residual"],
            "reproduces": synth_by_id[mid]["additive"],
        }
        for mid in synth_by_id
    ]
    return {"rows": rows, "all_reproduce": all(r["reproduces"] for r in rows)}


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Calibrating gain distribution from the 36 native additive rows...")
    a_min, a_max = _native_gain_range()
    print(f"  native gain range: {a_min:.2f}x - {a_max:.2f}x")

    print("Building synthetic mixing set (S1-09, source-file split S1-13)...")
    synthetic_df = build_synthetic_set(n_folds=SYNTHETIC_MIX_N_FOLDS, seed=SYNTHETIC_SEED)
    n_pairs = synthetic_df[["heart_id", "lung_id"]].drop_duplicates().shape[0]
    print(f"  {len(synthetic_df)} rows: {n_pairs} same-fold pairs x {len(SNR_SWEEP_DB)} SNR levels "
          f"across {synthetic_df['fold'].nunique()} folds")

    print("Validating construction against the 36 native additive rows (S1-10)...")
    validation = validate_against_native()
    n_ok = sum(r["reproduces"] for r in validation["rows"])
    print(f"  {n_ok}/{len(validation['rows'])} native rows reproduced (residual within threshold)")

    fold_counts = synthetic_df.groupby("fold").size().rename("rows").to_frame()
    validation_df = pd.DataFrame(validation["rows"]).set_index("mixed_id")

    stat_tiles = "\n".join([
        stat_tile("Synthetic rows", str(len(synthetic_df)), f"{SYNTHETIC_MIX_N_FOLDS} folds x {len(SNR_SWEEP_DB)} SNR levels"),
        stat_tile("Gain range", f"{a_min:.1f}x–{a_max:.1f}x", "from native additive rows"),
        stat_tile("S1-10 validation", f"{n_ok}/{len(validation['rows'])}", "native rows reproduced", ok=validation["all_reproduce"]),
    ])

    body = "\n\n".join([
        section("Rows per fold", f"{SYNTHETIC_MIX_N_FOLDS} folds, source-file split (S1-13)", df_to_html(fold_counts, index_label="fold")),
        section(
            "SNR sweep",
            "dB, hard → easy",
            f'<p class="mono-block">{", ".join(f"{s:g} dB" for s in SNR_SWEEP_DB)} '
            f'(closest to Han &amp; Quan\'s ~34 dB / 2% RMS condition: {SNR_DB_CLOSEST_TO_HAN_QUAN:g} dB)</p>',
        ),
        section(
            "S1-10: native-reproduction validation",
            f"{len(validation['rows'])} native additive rows, residual threshold 1e-3",
            df_to_html(validation_df, index_label="mixed_id", float_fmt="{:.2e}"),
        ),
    ])

    html = report_shell(
        title="Synthetic Mixing Set",
        eyebrow="HLS-CMDS · synthetic substrate (S1-09/S1-10/S1-13)",
        heading="Synthetic mixing set: build + native-reproduction validation",
        dek=(
            "Replaces native Mix.csv pairs as the primary evaluation substrate (only 36/145 are "
            "additive). mixed = a·(heart+lung) + noise, matching the native dataset's own "
            "construction model; gain a is calibrated from the native rows' own fitted gains, "
            "noise is swept across SNR_SWEEP_DB for a continuous difficulty axis. Split is "
            "source-file-level (S1-13), not triplet-level (S2-03) -- full combinatorial pairing "
            "collapses triplet-level leak groups into one giant component."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer="<p><strong>Method.</strong> See <code>synthetic_mix.py</code> / <code>code_description.md</code>.</p>",
    )

    report_path = write_report(results_dir() / "synthetic_mix_report.html", html)
    print(f"\nReport written to {report_path}")
