"""
Sensitivity of the additive-triplet audit (load_dataset.verify_additive_triplets)
to the relaxations a reviewer would ask about.

The audit fits mixed ~= a * (heart + lung) with one least-squares scalar
gain and calls a row additive when the relative residual is below 1e-3.
109 of 145 rows fail. This module asks whether they fail for a reason the
strict model would miss rather than because the mixture is unrelated to
its named sources, by re-fitting every row under progressively looser
models and reporting how many rows change verdict:

  strict        the audit as published: one gain, a >= 0 not enforced but
                a < 0 never occurs on the additive cluster
  dc            remove the mean of every signal before fitting
  polarity      allow a < 0 explicitly (report how many rows pick it)
  shift         allow an integer time shift of the summed reference
                within +/- MAX_SHIFT samples, chosen by cross-correlation
  dc+shift      both
  two-gain      mixed ~= a_h * heart + a_l * lung (separate gains)
  two-gain+shift  separate gains, and separate shifts per source

For every model the residual of every row is kept, so the report can show
the residual histogram (bimodality) and the extreme residuals of the two
clusters, and the verdict is recomputed at thresholds 1e-2, 1e-3 and 1e-4.
Sample rates are also read from every file and compared to the documented
4000 Hz.

Outputs: results/additive_audit_sensitivity.csv (one row per Mix.csv row
x model), results/additive_audit_sensitivity_report.html,
results/plots/additive_residual_hist.png.

Usage:
    make additive-audit-sensitivity
"""
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from load_dataset import load_audio, load_mix
from report_utils import results_dir

MAX_SHIFT = 100
THRESHOLDS = (1e-2, 1e-3, 1e-4)
DOCUMENTED_SR = 4000


def _fit_gain(mixed, ref):
    denom = float(ref @ ref)
    a = float(mixed @ ref) / denom if denom > 0 else 0.0
    res = np.linalg.norm(mixed - a * ref) / np.linalg.norm(mixed)
    return a, float(res)


def _fit_two_gains(mixed, h, l):
    X = np.stack([h, l], axis=1)
    coef, *_ = np.linalg.lstsq(X, mixed, rcond=None)
    res = np.linalg.norm(mixed - X @ coef) / np.linalg.norm(mixed)
    return float(coef[0]), float(coef[1]), float(res)


def _best_shift(mixed, ref, max_shift):
    """Integer lag maximising cross-correlation within +/- max_shift."""
    c = correlate(mixed, ref, mode="full", method="fft")
    lags = correlation_lags(len(mixed), len(ref), mode="full")
    keep = np.abs(lags) <= max_shift
    return int(lags[keep][np.argmax(np.abs(c[keep]))])


def _apply_shift(x, lag):
    """Shift x by `lag` samples (positive = delay), zero-padded, same length."""
    y = np.zeros_like(x)
    if lag >= 0:
        y[lag:] = x[: len(x) - lag]
    else:
        y[: len(x) + lag] = x[-lag:]
    return y


def audit_row(mixed, heart, lung, max_shift=MAX_SHIFT) -> dict:
    n = min(len(mixed), len(heart), len(lung))
    mixed, heart, lung = mixed[:n], heart[:n], lung[:n]
    summed = heart + lung
    out = {}

    a, r = _fit_gain(mixed, summed)
    out["strict"] = {"gain": a, "residual": r}

    m0, s0 = mixed - mixed.mean(), summed - summed.mean()
    a, r = _fit_gain(m0, s0)
    out["dc"] = {"gain": a, "residual": r}

    # polarity: the LS gain already takes the sign that minimises the residual;
    # record it explicitly and also the residual if the sign were forced positive
    a_free, r_free = _fit_gain(mixed, summed)
    r_pos = float(np.linalg.norm(mixed - abs(a_free) * summed) / np.linalg.norm(mixed))
    out["polarity"] = {"gain": a_free, "residual": r_free, "residual_if_positive": r_pos, "negative_gain": a_free < 0}

    lag = _best_shift(mixed, summed, max_shift)
    a, r = _fit_gain(mixed, _apply_shift(summed, lag))
    out["shift"] = {"gain": a, "residual": r, "lag": lag}

    lag0 = _best_shift(m0, s0, max_shift)
    a, r = _fit_gain(m0, _apply_shift(s0, lag0))
    out["dc+shift"] = {"gain": a, "residual": r, "lag": lag0}

    ah, al, r = _fit_two_gains(mixed, heart, lung)
    out["two-gain"] = {"gain_heart": ah, "gain_lung": al, "residual": r}

    lh, ll = _best_shift(mixed, heart, max_shift), _best_shift(mixed, lung, max_shift)
    ah, al, r = _fit_two_gains(mixed, _apply_shift(heart, lh), _apply_shift(lung, ll))
    out["two-gain+shift"] = {"gain_heart": ah, "gain_lung": al, "residual": r, "lag_heart": lh, "lag_lung": ll}
    return out


def run(mix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    mix_df = mix_df if mix_df is not None else load_mix()
    rows = []
    for _, row in mix_df.iterrows():
        h, sr_h = load_audio(row["heart_audio_path"], sr=None)
        l, sr_l = load_audio(row["lung_audio_path"], sr=None)
        m, sr_m = load_audio(row["mixed_audio_path"], sr=None)
        srs = {p: sf.info(row[p]).samplerate for p in ("heart_audio_path", "lung_audio_path", "mixed_audio_path")}
        res = audit_row(m, h, l)
        for model, vals in res.items():
            rows.append({"mixed_id": row["Mixed Sound ID"], "model": model, **vals,
                         "sr_heart": srs["heart_audio_path"], "sr_lung": srs["lung_audio_path"], "sr_mixed": srs["mixed_audio_path"]})
    return pd.DataFrame(rows)


def verdict_table(df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    strict_ids = set(df[(df["model"] == "strict") & (df["residual"] < 1e-3)]["mixed_id"])
    out = []
    for model, g in df.groupby("model", sort=False):
        rec = {"model": model}
        for t in thresholds:
            ids = set(g[g["residual"] < t]["mixed_id"])
            rec[f"n_additive@{t:g}"] = len(ids)
            if t == 1e-3:
                rec["gained_vs_strict"] = len(ids - strict_ids)
                rec["lost_vs_strict"] = len(strict_ids - ids)
        r = g["residual"].to_numpy()
        lo, hi = r[r < 1e-2], r[r >= 1e-2]
        rec["cluster_low_min"], rec["cluster_low_max"] = (lo.min(), lo.max()) if len(lo) else (np.nan, np.nan)
        rec["cluster_high_min"], rec["cluster_high_max"] = (hi.min(), hi.max()) if len(hi) else (np.nan, np.nan)
        rec["gap_ratio"] = (hi.min() / lo.max()) if len(lo) and len(hi) else np.nan
        out.append(rec)
    return pd.DataFrame(out).set_index("model")


def plot_hist(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["strict", "dc+shift", "two-gain+shift"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    bins = np.logspace(-9, 1, 60)
    for ax, model in zip(axes, models):
        r = df[df["model"] == model]["residual"].clip(1e-9, None)
        ax.hist(r, bins=bins, color="#3b6ea5")
        for t in THRESHOLDS:
            ax.axvline(t, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(model)
        ax.set_xlabel("relative residual ||m - fit|| / ||m||")
    axes[0].set_ylabel("rows (of 145)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    from report_utils import df_to_html, image_figure, report_shell, section, stat_tile, write_report

    print("Re-fitting every Mix.csv row under seven additivity models ...")
    df = run()
    rd = results_dir()
    df.to_csv(rd / "additive_audit_sensitivity.csv", index=False)
    (rd / "plots").mkdir(exist_ok=True)
    plot_hist(df, rd / "plots" / "additive_residual_hist.png")

    vt = verdict_table(df)
    pd.set_option("display.width", 250)
    print(vt.round(6).to_string())
    neg = df[(df["model"] == "polarity") & df["negative_gain"]]
    print("\nrows whose LS gain is negative:", len(neg), "-- of which additive at 1e-3:", int((neg["residual"] < 1e-3).sum()))
    sh = df[df["model"] == "dc+shift"]
    print("dc+shift: lag distribution among rows additive at 1e-3:", sh[sh["residual"] < 1e-3]["lag"].value_counts().to_dict())
    print("dc+shift: |lag| among non-additive rows: median", sh[sh["residual"] >= 1e-3]["lag"].abs().median(), "max", sh[sh["residual"] >= 1e-3]["lag"].abs().max())
    sr_bad = df[(df["model"] == "strict") & ((df["sr_heart"] != DOCUMENTED_SR) | (df["sr_lung"] != DOCUMENTED_SR) | (df["sr_mixed"] != DOCUMENTED_SR))]
    print("rows with any file not at", DOCUMENTED_SR, "Hz:", len(sr_bad))

    tiles = "\n".join([
        stat_tile("Additive, strict @1e-3", str(int(vt.loc["strict", "n_additive@0.001"])), "of 145"),
        stat_tile("Additive, two gains + shift @1e-3", str(int(vt.loc["two-gain+shift", "n_additive@0.001"])), "the loosest model"),
        stat_tile("Gap between clusters (strict)", f"{vt.loc['strict', 'gap_ratio']:.0e}x", "min residual of the failing cluster / max of the passing one"),
        stat_tile("Files not at 4000 Hz", str(len(sr_bad)), "of 145 x 3"),
    ])
    body = "\n\n".join([
        section("Residual histograms", "three models; dotted lines at 1e-2, 1e-3, 1e-4",
                image_figure("plots/additive_residual_hist.png", "residual histograms")),
        section("Verdict under each model and threshold", "gained/lost = rows that change verdict vs the published audit at 1e-3",
                df_to_html(vt, index_label="model", float_fmt="{:.3g}")),
    ])
    html = report_shell(
        title="Additive Audit Sensitivity", eyebrow="HLS-CMDS · additive-triplet audit · robustness of the 36/145 verdict",
        heading="Do the 109 failing rows fail for a trivial reason?",
        dek="DC offset, polarity, up to +/-100 samples of misalignment, and separate per-source gains, each tried on all 145 rows, "
            "with the verdict recomputed at three thresholds.",
        stat_tiles=tiles, body=body,
        footer="<p><strong>Method.</strong> See <code>additive_audit_sensitivity.py</code>'s module docstring.</p>",
    )
    print(f"\nReport written to {write_report(rd / 'additive_audit_sensitivity_report.html', html)}")
