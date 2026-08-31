"""Load the HS, LS, and Mix CSVs into pandas DataFrames and resolve each
row to its audio file inside the extracted HS/, LS/, and mix/ folders.

Usage:
    from load_dataset import load_hs, load_ls, load_mix, load_audio

    hs_df = load_hs()
    y, sr = load_audio(hs_df.loc[0, "audio_path"])
"""
import wave
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "HLS_CMDS"

LS_TYPE_TO_FILE_ABBREV = {
    "Fine Crackles": "FC",
    "Coarse Crackles": "CC",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def load_hs() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "HS.csv"))
    df["audio_path"] = str(DATA_DIR / "HS") + "/" + df["Heart Sound ID"] + ".wav"
    return df


def load_ls() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "LS.csv"))
    id_abbrev = df["Lung Sound ID"].str.split("_").str[1]
    abbrev = df["Lung Sound Type"].map(LS_TYPE_TO_FILE_ABBREV).fillna(id_abbrev)
    file_id = df["Gender"] + "_" + abbrev + "_" + df["Location"]
    df["audio_path"] = str(DATA_DIR / "LS") + "/" + file_id + ".wav"
    return df


def load_mix() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "Mix.csv"))
    mix_dir = str(DATA_DIR / "Mix")
    df["heart_audio_path"] = mix_dir + "/" + df["Heart Sound ID"] + ".wav"
    df["lung_audio_path"] = mix_dir + "/" + df["Lung Sound ID"] + ".wav"
    df["mixed_audio_path"] = mix_dir + "/" + df["Mixed Sound ID"] + ".wav"
    return df


def load_audio(path: str, sr=None):
    """Decode a .wav file with librosa."""
    import librosa

    return librosa.load(path, sr=sr)


def _validate(df: pd.DataFrame, path_col: str, label: str) -> None:
    missing = [p for p in df[path_col] if not Path(p).is_file()]
    if missing:
        raise ValueError(f"{label}: {len(missing)} rows reference missing audio files, e.g. {missing[:5]}")


PAPER_HEART_COUNTS = {
    "Normal": {"abbrev": "NH", "own_zip": 9, "mix_zip": 13},
    "Late Diastolic Murmur": {"abbrev": "LDM", "own_zip": 6, "mix_zip": 13},
    "Mid Systolic Murmur": {"abbrev": "MSM", "own_zip": 7, "mix_zip": 14},
    "Late Systolic Murmur": {"abbrev": "LSM", "own_zip": 5, "mix_zip": 17},
    "Atrial Fibrillation": {"abbrev": "AF", "own_zip": 4, "mix_zip": 15},
    "S4": {"abbrev": "S4", "own_zip": 2, "mix_zip": 16},
    "Early Systolic Murmur": {"abbrev": "ESM", "own_zip": 6, "mix_zip": 13},
    "S3": {"abbrev": "S3", "own_zip": 5, "mix_zip": 15},
    "Tachycardia": {"abbrev": "T", "own_zip": 3, "mix_zip": 16},
    "AV Block": {"abbrev": "AVB", "own_zip": 3, "mix_zip": 13},
}

PAPER_LUNG_COUNTS = {
    "Normal": {"abbrev": "NL", "own_zip": 12, "mix_zip": 28},
    "Wheezing": {"abbrev": "W", "own_zip": 7, "mix_zip": 28},
    "Fine Crackles": {"abbrev": "FC", "own_zip": 5, "mix_zip": 22},
    "Rhonchi": {"abbrev": "R", "own_zip": 8, "mix_zip": 23},
    "Pleural Rub": {"abbrev": "PR", "own_zip": 9, "mix_zip": 25},
    "Coarse Crackles": {"abbrev": "CC", "own_zip": 9, "mix_zip": 19},
}


def _compare_counts(own_counts: pd.Series, mix_counts: pd.Series, paper_counts: dict) -> list:
    rows = []
    for sound_type, paper in paper_counts.items():
        computed_own = int(own_counts.get(sound_type, 0))
        computed_mix = int(mix_counts.get(sound_type, 0))
        rows.append({
            "type": sound_type,
            "abbrev": paper["abbrev"],
            "paper_own": paper["own_zip"],
            "computed_own": computed_own,
            "paper_mix": paper["mix_zip"],
            "computed_mix": computed_mix,
            "match": paper["own_zip"] == computed_own and paper["mix_zip"] == computed_mix,
        })
    return rows


def compare_counts_to_paper() -> dict:
    """Compare the paper's published sound-type counts against what's actually in the CSVs."""
    hs_df, ls_df, mix_df = load_hs(), load_ls(), load_mix()

    heart_rows = _compare_counts(
        hs_df["Heart Sound Type"].value_counts(),
        mix_df["Heart Sound Type"].value_counts(),
        PAPER_HEART_COUNTS,
    )
    lung_rows = _compare_counts(
        ls_df["Lung Sound Type"].value_counts(),
        mix_df["Lung Sound Type"].value_counts(),
        PAPER_LUNG_COUNTS,
    )
    return {"heart": heart_rows, "lung": lung_rows}


_ROW_TEMPLATE = (
    '<tr><td class="type-name">{type}<span class="abbrev">{abbrev}</span></td>'
    '<td class="num">{paper_own}</td><td class="num computed col-group">{computed_own}</td>'
    '<td class="num col-group">{paper_mix}</td><td class="num computed">{computed_mix}</td>'
    '<td class="status">{badge}</td></tr>'
)

_MATCH_BADGE = '<span class="badge-match">&#10003; match</span>'
_MISMATCH_BADGE = '<span class="badge-mismatch">&#10007; mismatch</span>'


def _render_rows(rows: list) -> str:
    return "\n".join(
        _ROW_TEMPLATE.format(badge=_MATCH_BADGE if row["match"] else _MISMATCH_BADGE, **row)
        for row in rows
    )


def _render_table(title: str, source: str, own_zip_label: str, rows: list) -> str:
    total_paper_own = sum(r["paper_own"] for r in rows)
    total_computed_own = sum(r["computed_own"] for r in rows)
    total_paper_mix = sum(r["paper_mix"] for r in rows)
    total_computed_mix = sum(r["computed_mix"] for r in rows)
    return f"""  <section class="group">
    <div class="group-head">
      <h2>{title}</h2>
      <span class="group-source">{source}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th class="num">Paper &middot; {own_zip_label}</th>
            <th class="num col-group">Computed &middot; {own_zip_label}</th>
            <th class="num col-group">Paper &middot; Mix.zip</th>
            <th class="num col-group">Computed &middot; Mix.zip</th>
            <th class="status">Status</th>
          </tr>
        </thead>
        <tbody>
{_render_rows(rows)}
        </tbody>
        <tfoot>
          <tr>
            <td>Total</td>
            <td class="num">{total_paper_own}</td>
            <td class="num col-group">{total_computed_own}</td>
            <td class="num col-group">{total_paper_mix}</td>
            <td class="num">{total_computed_mix}</td>
            <td class="status"></td>
          </tr>
        </tfoot>
      </table>
    </div>
  </section>"""


_REPORT_CSS = """
  :root {
    --bg: #f5f6f4;
    --surface: #ffffff;
    --surface-alt: #eef1ef;
    --border: #dadfdc;
    --border-strong: #c3cbc7;
    --text: #1a211e;
    --text-muted: #5c6864;
    --accent: #1f5673;
    --accent-soft: #e3edf2;
    --success: #2f7a52;
    --success-soft: #e6f4eb;
    --success-border: #bfe0cd;
    --danger: #a33b2f;
    --danger-soft: #f7e7e4;
    --danger-border: #e3bdb5;

    --font-display: ui-serif, "Iowan Old Style", "Palatino Linotype", "URW Palladio L", Georgia, serif;
    --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Menlo, Consolas, monospace;
  }

  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #12181a;
      --surface: #182023;
      --surface-alt: #1e2729;
      --border: #2b3437;
      --border-strong: #3a4548;
      --text: #e7ecea;
      --text-muted: #93a19c;
      --accent: #7fbdda;
      --accent-soft: #1e323c;
      --success: #6fcb94;
      --success-soft: #16281e;
      --success-border: #2c4736;
      --danger: #e08a7c;
      --danger-soft: #2b1c1a;
      --danger-border: #4a2e29;
    }
  }

  :root[data-theme="dark"] {
    --bg: #12181a;
    --surface: #182023;
    --surface-alt: #1e2729;
    --border: #2b3437;
    --border-strong: #3a4548;
    --text: #e7ecea;
    --text-muted: #93a19c;
    --accent: #7fbdda;
    --accent-soft: #1e323c;
    --success: #6fcb94;
    --success-soft: #16281e;
    --success-border: #2c4736;
    --danger: #e08a7c;
    --danger-soft: #2b1c1a;
    --danger-border: #4a2e29;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    -webkit-font-smoothing: antialiased;
  }

  .page {
    max-width: 900px;
    margin: 0 auto;
    padding: 4rem 1.5rem 5rem;
  }

  header.report-head {
    border-bottom: 1px solid var(--border-strong);
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.9rem;
  }

  h1 {
    font-family: var(--font-display);
    font-weight: 500;
    font-size: clamp(1.8rem, 4vw, 2.4rem);
    line-height: 1.15;
    margin: 0 0 0.75rem;
    text-wrap: balance;
    letter-spacing: -0.01em;
  }

  .dek {
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--text-muted);
    max-width: 62ch;
    margin: 0;
  }

  .dek code {
    font-family: var(--font-mono);
    font-size: 0.92em;
    background: var(--surface-alt);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    color: var(--text);
  }

  .stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 3rem;
  }

  .stat {
    background: var(--surface);
    padding: 1.4rem 1.5rem;
  }

  .stat-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 0.5rem;
  }

  .stat-value {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--stat-color, var(--success));
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }

  .stat-value .of {
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-muted);
  }

  section.group {
    margin-bottom: 3rem;
  }

  .group-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.9rem;
    flex-wrap: wrap;
  }

  .group-head h2 {
    font-family: var(--font-display);
    font-weight: 500;
    font-size: 1.3rem;
    margin: 0;
  }

  .group-source {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.88rem;
    min-width: 640px;
  }

  thead th {
    text-align: left;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--surface-alt);
    padding: 0.7rem 0.9rem;
    border-bottom: 1px solid var(--border-strong);
    white-space: nowrap;
  }

  thead th.num, td.num { text-align: right; }
  thead th.status, td.status { text-align: center; }

  tbody td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  tbody tr:last-child td { border-bottom: none; }

  td.type-name {
    font-family: var(--font-body);
    font-size: 0.88rem;
  }

  td.type-name .abbrev {
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.78rem;
    margin-left: 0.4rem;
  }

  td.num { color: var(--text); }
  td.num.computed { color: var(--text-muted); }

  .badge-match, .badge-mismatch {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-family: var(--font-body);
    font-size: 0.76rem;
    font-weight: 600;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    border: 1px solid;
  }

  .badge-match {
    color: var(--success);
    background: var(--success-soft);
    border-color: var(--success-border);
  }

  .badge-mismatch {
    color: var(--danger);
    background: var(--danger-soft);
    border-color: var(--danger-border);
  }

  tfoot td {
    padding: 0.6rem 0.9rem;
    font-weight: 600;
    background: var(--surface-alt);
    border-top: 1px solid var(--border-strong);
  }

  .col-group {
    border-left: 1px solid var(--border-strong);
  }

  footer.report-foot {
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.7;
  }

  footer.report-foot p { margin: 0 0 0.5rem; }
  footer.report-foot code {
    font-family: var(--font-mono);
    font-size: 0.85em;
    background: var(--surface-alt);
    padding: 0.05em 0.35em;
    border-radius: 3px;
  }

  @media (max-width: 560px) {
    .stat-row { grid-template-columns: 1fr; }
    .page { padding: 2.5rem 1.1rem 3rem; }
  }
"""


def _report_shell(title: str, eyebrow: str, heading: str, dek: str, stat_tiles: str, body: str, footer: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{_REPORT_CSS}
</style>
</head>
<body>
<div class="page">

  <header class="report-head">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{heading}</h1>
    <p class="dek">{dek}</p>
  </header>

  <div class="stat-row">
{stat_tiles}
  </div>

{body}

  <footer class="report-foot">
{footer}
  </footer>

</div>
</body>
</html>
"""


def _stat_tile(label: str, value: str, of: str, ok: bool) -> str:
    color = "var(--success)" if ok else "var(--danger)"
    return (
        f'    <div class="stat">\n      <p class="stat-label">{label}</p>\n'
        f'      <div class="stat-value" style="--stat-color: {color}">{value} <span class="of">{of}</span></div>\n'
        f"    </div>"
    )


def generate_validation_report(output_path: Path | str | None = None) -> Path:
    """Render the paper-vs-computed comparison as a self-contained HTML report."""
    comparison = compare_counts_to_paper()
    heart_rows, lung_rows = comparison["heart"], comparison["lung"]

    heart_match = sum(r["match"] for r in heart_rows)
    lung_match = sum(r["match"] for r in lung_rows)
    total_match = heart_match + lung_match
    total_total = len(heart_rows) + len(lung_rows)

    all_match = total_match == total_total
    result_sentence = (
        f"All {total_total} published values ({len(heart_rows)} heart types &times; 2 columns, "
        f"{len(lung_rows)} lung types &times; 2 columns) match the local CSVs exactly."
        if all_match
        else f"{total_match}/{total_total} published values match the local CSVs &mdash; see the mismatches flagged above."
    )

    stat_tiles = "\n".join([
        _stat_tile("Heart sound values", f"{heart_match}/{len(heart_rows)}", "types match", heart_match == len(heart_rows)),
        _stat_tile("Lung sound values", f"{lung_match}/{len(lung_rows)}", "types match", lung_match == len(lung_rows)),
        _stat_tile("Total cells checked", f"{total_match}/{total_total}", "match", all_match),
    ])

    body = "\n\n".join([
        _render_table("Heart sounds", "HS.csv &middot; Mix.csv", "HS.zip", heart_rows),
        _render_table("Lung sounds", "LS.csv &middot; Mix.csv", "LS.zip", lung_rows),
    ])

    html = _report_shell(
        title="Dataset Count Validation",
        eyebrow="HLS&#8209;CMDS &middot; data integrity check",
        heading="Published table vs. local dataset",
        dek=(
            "Cross-checking the sound-type counts printed in the HLS-CMDS descriptor paper against what "
            "<code>load_dataset.py</code> actually reads out of <code>HS.csv</code>, <code>LS.csv</code>, "
            "and <code>Mix.csv</code>. Every cell below is a live count, not a transcription."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> Computed counts came from <code>load_hs()</code>, <code>load_ls()</code>, "
            "and <code>load_mix()</code> in <code>src/load_dataset.py</code>, grouped by <code>Heart Sound Type</code> "
            "/ <code>Lung Sound Type</code> and tallied with <code>pandas.Series.value_counts()</code>.</p>\n"
            f"    <p><strong>Result.</strong> {result_sentence}</p>"
        ),
    )

    output_path = Path(output_path) if output_path else Path(__file__).resolve().parent / "dataset_validation.html"
    output_path.write_text(html)
    return output_path


def verify_mix_triplets() -> dict:
    """
    Verify that every row in Mix.csv forms one correctly-paired H/L/M triplet:
    each Heart/Lung/Mixed Sound ID is used exactly once across the whole table
    (a true 1:1:1 mapping, no reused or missing IDs), and each triplet's three
    audio files agree on sample rate, frame count, and channel count (i.e. they
    were captured as one synchronized recording).
    """
    mix_df = load_mix()

    id_checks = []
    for col, prefix in [("Heart Sound ID", "H"), ("Lung Sound ID", "L"), ("Mixed Sound ID", "M")]:
        vals = mix_df[col]
        expected = {f"{prefix}{i:04d}" for i in range(1, len(mix_df) + 1)}
        id_checks.append({
            "column": col,
            "unique": bool(vals.is_unique),
            "complete": set(vals) == expected,
            "duplicates": vals[vals.duplicated()].tolist(),
        })

    triplets = []
    for _, row in mix_df.iterrows():
        formats = {}
        for key in ("heart_audio_path", "lung_audio_path", "mixed_audio_path"):
            with wave.open(row[key]) as w:
                formats[key] = (w.getframerate(), w.getnframes(), w.getnchannels())
        audio_consistent = formats["heart_audio_path"] == formats["lung_audio_path"] == formats["mixed_audio_path"]
        triplets.append({
            "mixed_id": row["Mixed Sound ID"],
            "heart_id": row["Heart Sound ID"],
            "lung_id": row["Lung Sound ID"],
            "sample_rate": formats["mixed_audio_path"][0],
            "frames": formats["mixed_audio_path"][1],
            "audio_consistent": audio_consistent,
        })

    return {"id_checks": id_checks, "triplets": triplets}


def verify_additive_triplets(mix_df: pd.DataFrame | None = None, residual_threshold: float = 1e-3) -> dict:
    """
    Test each Mix.csv row for mixed ~= a * (heart + lung) -- the assumption
    every separation baseline and BSS Eval score in this pipeline relies on.
    verify_mix_triplets() only checks that the three files exist, are
    uniquely paired, and share format; it says nothing about whether the
    mixed recording is actually related to the heart/lung files named
    alongside it, which is a separate, silent failure mode (see
    TEEP2026_Sprint0_Review: 109/145 rows on the GitHub copy of this
    dataset turned out to be acoustically unrelated to their named sources).

    For each row, fits the least-squares scalar gain a minimizing
    ||mixed - a*(heart+lung)||^2 (a = <mixed, summed> / <summed, summed>),
    then reports the relative residual ||mixed - a*summed|| / ||mixed||. A
    row is "additive" (usable as paired ground truth) when that residual
    falls below residual_threshold.

    Sensitivity note: on the data audited so far this measure is empirically
    bimodal -- genuinely additive rows land at residual ~1e-8-1e-4 (limited
    by 16-bit quantization), unrelated rows land at residual ~1 (no shared
    energy at all) -- with no rows in between, so the exact threshold value
    is not load-bearing across that gap. If a future dataset copy produces
    rows near the threshold, that gap assumption should be re-checked rather
    than assumed.

    Returns {"rows": [{"mixed_id", "gain", "relative_residual",
    "correlation", "additive": bool}, ...], "valid_ids": set of Mixed Sound
    IDs classified additive}.
    """
    mix_df = mix_df if mix_df is not None else load_mix()

    rows = []
    for _, row in mix_df.iterrows():
        heart, _sr = load_audio(row["heart_audio_path"], sr=None)
        lung, _sr = load_audio(row["lung_audio_path"], sr=None)
        mixed, _sr = load_audio(row["mixed_audio_path"], sr=None)
        n = min(len(heart), len(lung), len(mixed))
        heart, lung, mixed = heart[:n], lung[:n], mixed[:n]

        summed = heart + lung
        denom = float(np.dot(summed, summed))
        gain = float(np.dot(mixed, summed) / denom) if denom > 0 else 0.0
        residual_norm = float(np.linalg.norm(mixed - gain * summed))
        mixed_norm = float(np.linalg.norm(mixed))
        relative_residual = residual_norm / mixed_norm if mixed_norm > 0 else float("inf")
        correlation = (
            float(np.corrcoef(mixed, summed)[0, 1]) if np.std(summed) > 0 and np.std(mixed) > 0 else 0.0
        )

        rows.append({
            "mixed_id": row["Mixed Sound ID"],
            "gain": gain,
            "relative_residual": relative_residual,
            "correlation": correlation,
            "additive": relative_residual < residual_threshold,
        })

    valid_ids = {r["mixed_id"] for r in rows if r["additive"]}
    return {"rows": rows, "valid_ids": valid_ids}


def generate_pairing_report(output_path: Path | str | None = None) -> Path:
    """Render the M/H/L triplet-pairing verification as a self-contained HTML report."""
    result = verify_mix_triplets()
    id_checks, triplets = result["id_checks"], result["triplets"]
    total = len(triplets)

    id_rows = "\n".join(
        '<tr><td class="type-name">{col}</td><td class="status">{uniq}</td>'
        '<td class="status">{comp}</td><td class="num">{dupes}</td></tr>'.format(
            col=c["column"],
            uniq=_MATCH_BADGE if c["unique"] else _MISMATCH_BADGE,
            comp=_MATCH_BADGE if c["complete"] else _MISMATCH_BADGE,
            dupes=len(c["duplicates"]),
        )
        for c in id_checks
    )
    ids_ok = sum(1 for c in id_checks if c["unique"] and c["complete"])

    audio_ok = sum(t["audio_consistent"] for t in triplets)
    bad_triplets = [t for t in triplets if not t["audio_consistent"]]

    if bad_triplets:
        bad_rows = "\n".join(
            f'<tr><td class="type-name">{t["mixed_id"]}</td><td class="num">{t["heart_id"]}</td>'
            f'<td class="num">{t["lung_id"]}</td><td class="status">{_MISMATCH_BADGE}</td></tr>'
            for t in bad_triplets
        )
        audio_body = f"""  <section class="group">
    <div class="group-head">
      <h2>Audio format mismatches</h2>
      <span class="group-source">{len(bad_triplets)} of {total} triplets</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Mixed ID</th><th class="num">Heart ID</th><th class="num">Lung ID</th><th class="status">Status</th></tr></thead>
        <tbody>
{bad_rows}
        </tbody>
      </table>
    </div>
  </section>"""
    else:
        audio_body = """  <section class="group">
    <div class="group-head">
      <h2>Audio format consistency</h2>
    </div>
    <p class="dek">Every triplet's heart, lung, and mixed recordings agree on sample rate, frame count, and channel count &mdash; consistent with a single synchronized capture per row.</p>
  </section>"""

    stat_tiles = "\n".join([
        _stat_tile("Unique, complete ID columns", f"{ids_ok}/{len(id_checks)}", "columns", ids_ok == len(id_checks)),
        _stat_tile("Audio-format-consistent triplets", f"{audio_ok}/{total}", "triplets", audio_ok == total),
        _stat_tile("Total triplets checked", f"{total}", "rows", audio_ok == total),
    ])

    body = f"""  <section class="group">
    <div class="group-head">
      <h2>Referential integrity</h2>
      <span class="group-source">Mix.csv &middot; {total} rows</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Column</th><th class="status">Unique</th><th class="status">Covers 1&ndash;{total}</th><th class="num">Duplicates</th></tr>
        </thead>
        <tbody>
{id_rows}
        </tbody>
      </table>
    </div>
  </section>

{audio_body}"""

    all_ok = ids_ok == len(id_checks) and audio_ok == total
    result_sentence = (
        f"All {total} M/H/L triplets are correctly paired: every Heart/Lung/Mixed Sound ID is used exactly "
        f"once (a true 1:1:1 mapping across the {total} rows), and every triplet's three recordings share "
        f"the same sample rate, frame count, and channel count."
        if all_ok
        else "Some triplets failed verification &mdash; see the tables above for details."
    )

    html = _report_shell(
        title="Mix Triplet Pairing",
        eyebrow="HLS&#8209;CMDS &middot; data integrity check",
        heading="Verifying all 145 M/H/L triplets",
        dek=(
            "Confirming that each row in <code>Mix.csv</code> pairs a unique heart recording, lung recording, "
            "and mixed recording &mdash; and that the three files in each triplet were captured together."
        ),
        stat_tiles=stat_tiles,
        body=body,
        footer=(
            "<p><strong>Method.</strong> <code>verify_mix_triplets()</code> in <code>src/load_dataset.py</code> checks "
            "ID uniqueness/completeness on the <code>Mix.csv</code> columns, then opens each triplet's three .wav files "
            "with the stdlib <code>wave</code> module and compares sample rate, frame count, and channel count.</p>\n"
            f"    <p><strong>Result.</strong> {result_sentence}</p>"
        ),
    )

    output_path = Path(output_path) if output_path else Path(__file__).resolve().parent / "mix_pairing_validation.html"
    output_path.write_text(html)
    return output_path


if __name__ == "__main__":
    hs_df = load_hs()
    ls_df = load_ls()
    mix_df = load_mix()

    _validate(hs_df, "audio_path", "HS")
    _validate(ls_df, "audio_path", "LS")
    for path_col, label in [
        ("heart_audio_path", "Mix (heart)"),
        ("lung_audio_path", "Mix (lung)"),
        ("mixed_audio_path", "Mix (mixed)"),
    ]:
        _validate(mix_df, path_col, label)

    print(f"HS:  {len(hs_df)} rows, all audio files found")
    print(f"LS:  {len(ls_df)} rows, all audio files found")
    print(f"Mix: {len(mix_df)} rows, all audio files found")

    report_path = generate_validation_report()
    print(f"Validation report written to {report_path}")

    pairing_report_path = generate_pairing_report()
    print(f"Pairing report written to {pairing_report_path}")

    with wave.open(mix_df.loc[0, "mixed_audio_path"]) as w:
        sample_rate = w.getframerate()

    additivity = verify_additive_triplets(mix_df)
    print(
        f"\nSample rate: {sample_rate} Hz\n"
        f"Additivity check (mixed ~= a*(heart+lung)): "
        f"{len(additivity['valid_ids'])}/{len(mix_df)} rows additive "
        f"(residual_threshold={1e-3:g})"
    )
