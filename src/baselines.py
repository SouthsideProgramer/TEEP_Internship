"""
Baseline 0 (raw mixture, no separation) plus the report-generation glue
that ties Baselines 0-4 together into results/baselines_report.html and
builds the SSA-paper-style synthetic evaluation set -- the "does the real
method even beat a filter" sanity floor referenced in PROTOCOL.md 5.2/8.

Baselines 1-5 each live in their own module under baseline/ (baseline1.py
.. baseline5.py, split out of what used to be one large baselines.py --
see code_description.md for the full writeup of each: papers, parameter
choices, the ablation vs. leakage-trap reasoning, and the interpretation
calls flagged as this project's own extensions of the source papers).

Usage:
    from baselines import fit_raw_mixture_baseline
    from baseline.baseline1 import fit_bandpass_baseline
    from baseline.baseline2 import make_supervised_nmf_baseline
    from baseline.baseline3 import make_standard_nmf_baseline
    from baseline.baseline4 import fit_ssa_baseline
    from baseline.baseline5 import fit_evmd_baseline
    from eval_harness import cross_validate

    results_df, fold_summary, cv_summary = cross_validate(fit_raw_mixture_baseline, n_folds=5)
"""
import numpy as np
import pandas as pd

from metrics import SOURCE_LABELS


def raw_mixture_separate(mixed: np.ndarray, _sr: int):
    """
    separate_fn(mixed, sr) -> (heart_est, lung_est) that performs no
    separation at all -- both "estimates" are just the mixture. This is the
    reference point every other baseline's SDR must be read against: SDR
    already reflects a mixture-vs-two-references orthogonal projection even
    with zero separation, so e.g. "Baseline 1 gets 5.46 dB SDR" is only
    meaningful once you know what passthrough alone already scores on the
    same rows. Also the fastest way to sanity-check the additivity axis
    itself (see build_synthetic_mixes/verify_additive_triplets): on rows
    where mixed != a*(heart+lung), a large chunk of the mixture's own energy
    isn't in either reference's span, so *every* method -- including this
    zero-op one -- inherits negative SAR from that non-additive residual.
    """
    return mixed, mixed


def fit_raw_mixture_baseline(_hs_allowed, _ls_allowed):
    """fit_and_separate_fn for eval_harness.cross_validate. No learned parameters, no dictionary pool needed."""
    return raw_mixture_separate


SYNTHETIC_N_HEART = 10
SYNTHETIC_N_LUNG = 5
SYNTHETIC_NOISE_RMS_FRAC = 0.02
SYNTHETIC_SEED = 0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def build_synthetic_mixes(seed: int = SYNTHETIC_SEED) -> list[dict]:
    """
    Reproduces the SSA paper's own evaluation-set recipe (Sec. III), not
    HLS_CMDS's own Mix.csv: pick SYNTHETIC_N_HEART heart + SYNTHETIC_N_LUNG
    lung recordings from this project's own HS.csv/LS.csv, form every
    combinatorial pair (10x5=50), sum each pair, then add Gaussian noise at
    SYNTHETIC_NOISE_RMS_FRAC of the combined signal's RMS amplitude. This
    gives this project a like-for-like comparison against the paper's Table I
    (this module's own real-mixture numbers, from `eval_harness`, are
    not directly comparable -- see PROTOCOL.md's `[ssa]` row in Sec. 2).
    """
    from load_dataset import load_audio, load_hs, load_ls

    hs_df, ls_df = load_hs(), load_ls()
    rng = np.random.default_rng(seed)
    heart_rows = hs_df.iloc[rng.choice(len(hs_df), size=SYNTHETIC_N_HEART, replace=False)]
    lung_rows = ls_df.iloc[rng.choice(len(ls_df), size=SYNTHETIC_N_LUNG, replace=False)]

    pairs = []
    for _, heart_row in heart_rows.iterrows():
        heart_y, sr = load_audio(heart_row["audio_path"], sr=None)
        for _, lung_row in lung_rows.iterrows():
            lung_y, _ = load_audio(lung_row["audio_path"], sr=None)
            n = min(len(heart_y), len(lung_y))
            heart_y_n, lung_y_n = heart_y[:n], lung_y[:n]
            combined = heart_y_n + lung_y_n
            noise = rng.normal(0.0, SYNTHETIC_NOISE_RMS_FRAC * _rms(combined), size=n)
            pairs.append(
                {
                    "heart": heart_y_n,
                    "lung": lung_y_n,
                    "mixed": combined + noise,
                    "sr": sr,
                    "heart_id": heart_row["Heart Sound ID"],
                    "lung_id": lung_row["Lung Sound ID"],
                }
            )
    return pairs


def _run_synthetic_report(label: str, separate_fn, pairs: list[dict]) -> str:
    """
    Evaluate a plain separate_fn(mixed, sr) -> (heart_est, lung_est) over the
    synthetic pairs and report mean SDR + Pearson correlation per source, to
    compare directly against the SSA paper's Table I (SDR/Correlation
    columns; no STOI here -- `pystoi` isn't in requirements.txt/`.venv` and
    this project didn't want to add a new dependency unasked-for, so STOI is
    a known gap relative to the paper's third metric).

    Prints only a short progress line; returns an HTML <section> for the
    combined results/baselines_report.html (see report_utils.py).
    """
    from metrics import evaluate_heart_lung
    from report_utils import df_to_html, section

    print(f"Running {label} on synthetic set ({len(pairs)} pairs)...")
    rows = []
    for pair in pairs:
        heart_est, lung_est = separate_fn(pair["mixed"], pair["sr"])
        m = evaluate_heart_lung(pair["heart"], pair["lung"], heart_est, lung_est)
        nh = min(len(pair["heart"]), len(heart_est))
        nl = min(len(pair["lung"]), len(lung_est))
        rows.append(
            {
                "sdr_heart": m["heart"]["sdr"],
                "sdr_lung": m["lung"]["sdr"],
                "corr_heart": float(np.corrcoef(pair["heart"][:nh], heart_est[:nh])[0, 1]),
                "corr_lung": float(np.corrcoef(pair["lung"][:nl], lung_est[:nl])[0, 1]),
            }
        )

    df = pd.DataFrame(rows)
    mean_df = df.mean().to_frame(name="mean").T
    return section(f"{label} (synthetic set)", f"{len(pairs)} pairs, mean over all pairs", df_to_html(mean_df, index_label=""))


def _run_and_report(label: str, fit_fn, full_mix_df: pd.DataFrame, valid_mix_df: pd.DataFrame, seed: int = 0) -> dict:
    """
    Run fit_fn's cross-validation on both the full Mix.csv and the
    additive-only subset (load_dataset.verify_additive_triplets()). Per
    TEEP2026_Sprint0_Review, the full-145-row headline is known to be
    diluted by 109 rows whose "mixed" file is acoustically unrelated to its
    named heart/lung sources on the current (GitHub, not yet Mendeley)
    dataset copy -- see README.md's Dataset section. Both fold-level
    (mean+/-std across folds) and row-level (mean/median pooled across every
    evaluated row) numbers are reported and explicitly labeled, since they
    are not interchangeable -- averaging within folds before taking std
    understates row-to-row spread (the review's statistics note).

    Prints only short progress lines; returns {"html": <section> for the
    combined results/baselines_report.html, "cv_summary_full"/
    "cv_summary_valid": the raw across-fold summaries, so __main__ can build
    a ΔSDR-vs-baseline table without re-running cross_validate}.
    """
    from eval_harness import cross_validate, summarize_pooled
    from report_utils import df_to_html, section

    print(f"Running {label}...")

    print(f"  full {len(full_mix_df)} rows...")
    results_full, fold_summary_full, cv_summary_full = cross_validate(fit_fn, n_folds=5, seed=seed, mix_df=full_mix_df)
    full_body = "\n".join([
        "<h4>Per-fold means</h4>",
        df_to_html(fold_summary_full.set_index(["fold", "source"]), index_label="fold / source"),
        "<h4>Across-fold mean &plusmn; std (fold-level dispersion)</h4>",
        df_to_html(cv_summary_full, index_label="source"),
        "<h4>Pooled mean/median/std (row-level dispersion, every evaluated row)</h4>",
        df_to_html(summarize_pooled(results_full), index_label="source"),
    ])

    print(f"  additive-only {len(valid_mix_df)} rows...")
    results_valid, fold_summary_valid, cv_summary_valid = cross_validate(
        fit_fn, n_folds=5, seed=seed, mix_df=valid_mix_df
    )
    valid_body = "\n".join([
        "<h4>Per-fold means</h4>",
        df_to_html(fold_summary_valid.set_index(["fold", "source"]), index_label="fold / source"),
        "<h4>Across-fold mean &plusmn; std (fold-level dispersion)</h4>",
        df_to_html(cv_summary_valid, index_label="source"),
        "<h4>Pooled mean/median/std (row-level dispersion, every evaluated row)</h4>",
        df_to_html(summarize_pooled(results_valid), index_label="source"),
    ])

    body = (
        f'<h3>Full {len(full_mix_df)} rows</h3>\n{full_body}\n'
        f'<h3>Additive-only {len(valid_mix_df)} rows (mixed &asymp; a&middot;(heart+lung))</h3>\n{valid_body}'
    )
    html = section(label, f"full {len(full_mix_df)} / additive-only {len(valid_mix_df)} rows", body)
    return {
        "html": html,
        "cv_summary_full": cv_summary_full,
        "cv_summary_valid": cv_summary_valid,
    }


def _delta_vs_raw_mixture_section(baseline_results: list, raw_cv_summary_valid: pd.DataFrame) -> str:
    """
    Delta-SDR table (additive-only subset only -- the full-145-row axis
    isn't interpretable, see the report's dek): each baseline's across-fold
    mean SDR minus Baseline 0's (raw mixture, no separation) mean SDR, per
    source. This is the number that actually answers "did separation help
    at all", since a baseline's raw SDR is meaningless without knowing what
    passthrough already scores on the same rows.
    """
    from report_utils import df_to_html, section

    rows = []
    for label, cv_summary_valid in baseline_results:
        row = {"baseline": label}
        for src in SOURCE_LABELS:
            row[f"{src}_sdr"] = cv_summary_valid.loc[src, ("sdr", "mean")]
            row[f"delta_{src}_sdr"] = cv_summary_valid.loc[src, ("sdr", "mean")] - raw_cv_summary_valid.loc[src, ("sdr", "mean")]
        rows.append(row)
    delta_df = pd.DataFrame(rows).set_index("baseline")

    return section(
        "ΔSDR vs. raw mixture (additive-only subset)",
        "across-fold mean SDR minus Baseline 0's (no separation)",
        df_to_html(delta_df, index_label="baseline", float_fmt="{:.2f}"),
    )


if __name__ == "__main__":
    from baseline.baseline1 import HEART_BAND, LUNG_BAND, bandpass_separate, fit_bandpass_baseline
    from baseline.baseline2 import ACTIVATION_ITERS, DICT_ITERS, K_HEART, K_LUNG, make_supervised_nmf_baseline
    from baseline.baseline3 import STANDARD_NMF_ITERS, make_standard_nmf_baseline
    from baseline.baseline4 import (
        SSA_CARDIAC_SPLIT_HZ,
        SSA_CORRELATION_THRESHOLD,
        SSA_EIGENVALUE_THRESHOLD_PCT,
        SSA_WINDOW_LENGTH,
        fit_ssa_baseline,
        mssa_separate,
    )
    from load_dataset import load_mix, verify_additive_triplets
    from report_utils import report_shell, results_dir, stat_tile, write_report

    print("Loading Mix.csv and checking additivity (mixed ~= a*(heart+lung))...")
    full_mix_df = load_mix()
    additivity = verify_additive_triplets(full_mix_df)
    valid_mix_df = full_mix_df[full_mix_df["Mixed Sound ID"].isin(additivity["valid_ids"])].reset_index(drop=True)
    print(f"Dataset: {len(full_mix_df)} rows total, {len(valid_mix_df)} pass the additivity check")

    baseline_specs = [
        ("Baseline 0 (raw mixture, no separation)", fit_raw_mixture_baseline),
        (f"Baseline 1 (bandpass): heart={HEART_BAND} Hz, lung={LUNG_BAND} Hz", fit_bandpass_baseline),
        (
            f"Baseline 2 (supervised NMF): Ki(heart)={K_HEART}, Kr(lung)={K_LUNG}, "
            f"dict_iters={DICT_ITERS}, activation_iters={ACTIVATION_ITERS}",
            make_supervised_nmf_baseline(seed=0),
        ),
        (
            f"Baseline 3 (standard NMF, no learned dictionary; ablation against Baseline 2): "
            f"k_total={K_HEART + K_LUNG}, iters={STANDARD_NMF_ITERS}",
            make_standard_nmf_baseline(seed=0),
        ),
        (
            f"Baseline 4 (MSSA, reproducing Han & Quan ICSPS 2025): L={SSA_WINDOW_LENGTH}, "
            f"split={SSA_CARDIAC_SPLIT_HZ} Hz, eigenvalue_threshold={SSA_EIGENVALUE_THRESHOLD_PCT}%, "
            f"correlation_threshold={SSA_CORRELATION_THRESHOLD}",
            fit_ssa_baseline,
        ),
    ]

    baseline_run_results = [
        (label, _run_and_report(label, fit_fn, full_mix_df, valid_mix_df)) for label, fit_fn in baseline_specs
    ]
    sections = [r["html"] for _label, r in baseline_run_results]

    raw_cv_summary_valid = baseline_run_results[0][1]["cv_summary_valid"]
    sections.append(
        _delta_vs_raw_mixture_section(
            [(label, r["cv_summary_valid"]) for label, r in baseline_run_results[1:]],
            raw_cv_summary_valid,
        )
    )

    print("Building SSA-paper-style synthetic evaluation set...")
    synthetic_pairs = build_synthetic_mixes(seed=SYNTHETIC_SEED)
    print(
        f"Synthetic set: {len(synthetic_pairs)} pairs "
        f"({SYNTHETIC_N_HEART} heart x {SYNTHETIC_N_LUNG} lung, all combinations), "
        f"{SYNTHETIC_NOISE_RMS_FRAC * 100:.0f}% RMS Gaussian noise added"
    )
    sections.append(_run_synthetic_report("Baseline 0 (raw mixture, no separation)", raw_mixture_separate, synthetic_pairs))
    sections.append(_run_synthetic_report("Baseline 1 (bandpass)", bandpass_separate, synthetic_pairs))
    sections.append(_run_synthetic_report("Baseline 4 (MSSA)", mssa_separate, synthetic_pairs))

    stat_tiles = "\n".join([
        stat_tile("Mix rows", str(len(full_mix_df)), "total"),
        stat_tile("Additive rows", str(len(valid_mix_df)), f"{100 * len(valid_mix_df) / len(full_mix_df):.0f}%"),
        stat_tile("Baselines run", "5", "0-4, + 3 synthetic"),
    ])

    html = report_shell(
        title="Baseline Separation Results",
        eyebrow="HLS-CMDS · separation baselines",
        heading="Baselines 0&ndash;4 + SSA-paper synthetic set",
        dek=(
            "Cross-validated SDR/SIR/SAR for each baseline on the full Mix.csv and the "
            "additive-only subset, plus a synthetic-set comparison against the SSA "
            "paper's own Table I (MSSA: cardiac SDR 26.4 dB / corr 99.2%, respiratory "
            "SDR 5.3 dB / corr 80.5%; Butterworth baseline: cardiac SDR 5.7 dB, "
            "respiratory SDR -5.7 dB). <strong>The full-145-row numbers are not "
            "interpretable as separation quality</strong> -- 109 rows have mixed != "
            "a*(heart+lung), so a large share of the mixture's own energy sits outside "
            "the span of both references and every method (including Baseline 0) "
            "inherits negative SAR from that non-additive residual. Only the "
            "additive-only 36-row numbers, and the ΔSDR-vs-Baseline-0 table below, "
            "should be read as separation quality."
        ),
        stat_tiles=stat_tiles,
        body="\n\n".join(sections),
        footer=(
            "<p><strong>Method.</strong> See <code>baseline/</code> / "
            "<code>code_description.md</code> for each baseline's parameters and "
            "interpretation notes. Baseline 0 added specifically so every other "
            "baseline's SDR can be read as a delta over doing nothing, not an "
            "absolute number.</p>"
        ),
    )

    report_path = write_report(results_dir() / "baselines_report.html", html)
    print(f"\nReport written to {report_path}")
