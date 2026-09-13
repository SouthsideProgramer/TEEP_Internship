"""
Oracle-mask ceiling check: how well *could* a mask-based separator do on
HLS-CMDS if it knew the answer?

Why this exists. self_learning.md records an informal probe, made while
Baseline 2 (supervised NMF) was found to barely match Baseline 1
(bandpass): an oracle mask built from one mixture row's own true
heart/lung spectra "only modestly beat no-separation on that row". If
true in general, it would mean the dataset's real acoustic mixtures
impose a hard ceiling on mask-based separation and NMF's gap is a dataset
property, not an implementation gap. report.tex flags that probe as not
reproducible (one row, no script). This module is the re-runnable check,
over the 36 additive-valid rows instead of one.

What is measured. For every row of the native additive subset
(load_dataset.verify_additive_triplets()), three oracle masks are built on
the same STFT grid Baseline 2 uses (N_FFT=512, HOP=256) from the row's
*true* heart and lung spectrograms -- cheating on purpose -- and applied
to the mixture's complex STFT:

    IRM   (ideal ratio mask)     |H| / (|H| + |L|)          -- Baseline 2's
                                                              exact mask form
    IWF   (ideal Wiener filter)  |H|^2 / (|H|^2 + |L|^2)
    IBM   (ideal binary mask)    1[|H| > |L|]

The lung estimate is the complementary mask on the same STFT. Every
oracle is then scored with the same BSS Eval call as every baseline
(metrics.evaluate_heart_lung), alongside Baseline 0 (mixture passthrough)
and Baseline 1 (bandpass) on the identical rows, so the ceiling and the
floor sit in one table. No folds: nothing here is fitted.

How to read it. Oracle SDR is the ceiling for any separator that acts as
a magnitude mask on this STFT grid (Baseline 2's family). The gap
"oracle minus Baseline 2" is what better dictionaries/activations could
at most recover; the gap "oracle minus Baseline 1" is what a mask-based
method has to earn before it is worth its cost. If the oracle itself sat
near Baseline 0, the self_learning.md hypothesis would hold; if it sits
far above, that hypothesis is refuted for the valid rows and the
Baseline 2 shortfall is the method's.

Usage:
    from oracle_mask_ceiling import run_oracle_ceiling

    results_df = run_oracle_ceiling()       # one row per (mixed_id, method, source)
"""
import librosa
import numpy as np
import pandas as pd

from baseline.baseline1 import bandpass_separate
from baselines import raw_mixture_separate
from load_dataset import load_audio, load_mix, verify_additive_triplets
from metrics import SOURCE_LABELS, evaluate_heart_lung

# Same grid as baseline2.py so the IRM is exactly that method's ceiling.
N_FFT = 512
HOP_LENGTH = 256
_EPS = 1e-12

ORACLE_LABELS = {
    "IRM": "Oracle IRM (|H|/(|H|+|L|))",
    "IWF": "Oracle Wiener (|H|^2/(|H|^2+|L|^2))",
    "IBM": "Oracle binary (|H|>|L|)",
}
REFERENCE_LABELS = {
    "B0": "Baseline 0 (raw mixture, no separation)",
    "B1": "Baseline 1 (bandpass)",
}


def oracle_masks(heart_ref: np.ndarray, lung_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Heart-side masks on the (N_FFT, HOP_LENGTH) grid from the true sources.
    The lung mask is 1 - mask in every case."""
    H = np.abs(librosa.stft(heart_ref, n_fft=N_FFT, hop_length=HOP_LENGTH))
    L = np.abs(librosa.stft(lung_ref, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return {
        "IRM": H / (H + L + _EPS),
        "IWF": H**2 / (H**2 + L**2 + _EPS),
        "IBM": (H > L).astype(np.float64),
    }


def apply_mask(mixed: np.ndarray, mask_heart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask the *mixture's* complex STFT (the only thing a real separator has) and invert."""
    S_mix = librosa.stft(mixed, n_fft=N_FFT, hop_length=HOP_LENGTH)
    heart_est = librosa.istft(mask_heart * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
    lung_est = librosa.istft((1.0 - mask_heart) * S_mix, hop_length=HOP_LENGTH, length=len(mixed))
    return heart_est, lung_est


def oracle_separate(mixed: np.ndarray, heart_ref: np.ndarray, lung_ref: np.ndarray, kind: str = "IRM"):
    """separate_fn-shaped helper that is allowed to see the references."""
    return apply_mask(mixed, oracle_masks(heart_ref, lung_ref)[kind])


def evaluate_row(row: pd.Series) -> list[dict]:
    """Every oracle and reference method on one Mix.csv row -> tidy rows."""
    heart_ref, sr = load_audio(row["heart_audio_path"], sr=None)
    lung_ref, _ = load_audio(row["lung_audio_path"], sr=None)
    mixed, _ = load_audio(row["mixed_audio_path"], sr=None)
    n = min(len(heart_ref), len(lung_ref), len(mixed))
    heart_ref, lung_ref, mixed = heart_ref[:n], lung_ref[:n], mixed[:n]

    estimates = {
        "B0": raw_mixture_separate(mixed, sr),
        "B1": bandpass_separate(mixed, sr),
    }
    masks = oracle_masks(heart_ref, lung_ref)
    for kind, mask in masks.items():
        estimates[kind] = apply_mask(mixed, mask)

    out = []
    for key, (heart_est, lung_est) in estimates.items():
        m = evaluate_heart_lung(heart_ref, lung_ref, heart_est, lung_est)
        for source in SOURCE_LABELS:
            out.append({
                "mixed_id": row["Mixed Sound ID"],
                "method": key,
                "source": source,
                "sdr": m[source]["sdr"],
                "sir": m[source]["sir"],
                "sar": m[source]["sar"],
            })
    return out


def run_oracle_ceiling(mix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Oracles + references over the native additive subset (default) or any Mix.csv rows."""
    if mix_df is None:
        mix_df = load_mix()
        valid_ids = verify_additive_triplets(mix_df)["valid_ids"]
        mix_df = mix_df[mix_df["Mixed Sound ID"].isin(valid_ids)].reset_index(drop=True)
    rows = []
    for _, row in mix_df.iterrows():
        rows.extend(evaluate_row(row))
    return pd.DataFrame(rows)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean / 95% CI / median per (method, source), same convention as
    first_sdr_table.py (t-based CI over rows, n = number of mixtures)."""
    from scipy import stats

    def ci95(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return float("nan")
        return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

    g = results_df.groupby(["method", "source"])
    summary = g.agg(n=("sdr", "size"), sdr_mean=("sdr", "mean"), sdr_ci95=("sdr", ci95), sdr_median=("sdr", "median"),
                    sir_mean=("sir", "mean"), sar_mean=("sar", "mean"))
    order = list(REFERENCE_LABELS) + list(ORACLE_LABELS)
    return summary.reindex(pd.MultiIndex.from_product([order, SOURCE_LABELS], names=["method", "source"]))


def headroom_table(summary: pd.DataFrame, baseline_sdr: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Oracle-IRM mean SDR minus each reference's mean SDR, per source.
    baseline_sdr may add already-published numbers (e.g. Baseline 2's
    native-additive SDR) as {"label": {"heart": x, "lung": y}} so the
    ceiling can be read against methods this script does not re-run."""
    rows = []
    irm = summary.loc["IRM", "sdr_mean"]
    refs = {REFERENCE_LABELS[k]: summary.loc[k, "sdr_mean"].to_dict() for k in REFERENCE_LABELS}
    if baseline_sdr:
        refs.update(baseline_sdr)
    for label, per_source in refs.items():
        rows.append({"reference": label, **{f"headroom_{s}_db": float(irm[s] - per_source[s]) for s in SOURCE_LABELS}})
    return pd.DataFrame(rows).set_index("reference")


if __name__ == "__main__":
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    # Baseline 2's native-additive SDR as published in report.tex
    # (tab:baselinevalid, 5-fold, mean across folds) -- the number the
    # oracle ceiling is read against; kept as data here rather than
    # re-running a 5-fold NMF fit.
    BASELINE2_NATIVE_SDR = {"Baseline 2 (supervised NMF), report tab:baselinevalid": {"heart": 3.78, "lung": -0.44}}

    print("Oracle-mask ceiling over the native additive subset...")
    results_df = run_oracle_ceiling()
    n_rows = results_df["mixed_id"].nunique()
    results_df.to_csv(results_dir() / "oracle_mask_ceiling.csv", index=False)

    summary = summarize(results_df)
    headroom = headroom_table(summary, BASELINE2_NATIVE_SDR)
    labels = {**REFERENCE_LABELS, **ORACLE_LABELS}
    summary_named = summary.rename(index=labels, level="method")

    print(summary_named[["n", "sdr_mean", "sdr_ci95", "sdr_median"]].round(2).to_string())
    print("\nOracle IRM headroom (dB) over each reference:")
    print(headroom.round(2).to_string())

    irm_h, irm_l = summary.loc[("IRM", "heart"), "sdr_mean"], summary.loc[("IRM", "lung"), "sdr_mean"]
    b0_h, b0_l = summary.loc[("B0", "heart"), "sdr_mean"], summary.loc[("B0", "lung"), "sdr_mean"]
    b1_h, b1_l = summary.loc[("B1", "heart"), "sdr_mean"], summary.loc[("B1", "lung"), "sdr_mean"]

    stat_tiles = "\n".join([
        stat_tile("Oracle IRM SDR, heart", f"{irm_h:.2f} dB", f"vs Baseline 0 {b0_h:.2f} / Baseline 1 {b1_h:.2f}"),
        stat_tile("Oracle IRM SDR, lung", f"{irm_l:.2f} dB", f"vs Baseline 0 {b0_l:.2f} / Baseline 1 {b1_l:.2f}"),
        stat_tile("Rows", str(n_rows), "native additive subset, no folds"),
    ])

    per_row = (
        results_df.pivot_table(index="mixed_id", columns=["method", "source"], values="sdr")
        .reindex(columns=pd.MultiIndex.from_product([list(REFERENCE_LABELS) + list(ORACLE_LABELS), SOURCE_LABELS]))
    )

    body = "\n\n".join([
        section(
            "Ceiling vs floor, pooled over the 36 rows",
            "mean, 95% t-CI and median SDR per (method, source); SIR/SAR means alongside",
            df_to_html(summary_named, index_label="method / source", float_fmt="{:.2f}"),
        ),
        section(
            "Headroom: oracle IRM minus each reference",
            "what a mask-based separator on this STFT grid could at most gain",
            df_to_html(headroom, index_label="reference", float_fmt="{:.2f}"),
        ),
        section(
            "Per-row SDR (dB)",
            "results/oracle_mask_ceiling.csv",
            df_to_html(per_row, index_label="mixed_id", float_fmt="{:.1f}"),
        ),
    ])

    html = report_shell(
        title="Oracle-Mask Ceiling",
        eyebrow="HLS-CMDS · native additive subset · separation ceiling",
        heading="How well could a mask-based separator do if it knew the answer?",
        dek=(
            "Three oracle masks built from each row's true heart/lung spectrograms (cheating on purpose) "
            "on Baseline 2's own STFT grid, scored with the same BSS Eval as every baseline, next to "
            "Baseline 0 (no separation) and Baseline 1 (bandpass) on the identical rows. Replaces the "
            "one-row informal probe recorded in self_learning.md."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer="<p><strong>Method.</strong> See <code>oracle_mask_ceiling.py</code>'s module docstring.</p>",
    )
    report_path = write_report(results_dir() / "oracle_mask_ceiling_report.html", html)
    print(f"\nReport written to {report_path}")
