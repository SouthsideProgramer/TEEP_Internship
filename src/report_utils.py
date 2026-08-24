"""
Shared HTML report building blocks for the results/ folder.

Every analysis script under src/ (baselines.py, eval_harness.py, split.py,
metrics.py, statistics/audio_quality.py, visualization/*.py) used to print
its results as long text tables straight to the terminal. That got hard to
read once the tables piled up, so each script's __main__ now prints only
short progress lines and instead renders its results into a self-contained
HTML file under results/ (repo root), using this module. Same visual style
as load_dataset.py's dataset_validation.html / mix_pairing_validation.html.

Usage:
    from report_utils import df_to_html, report_shell, section, stat_tile, results_dir, write_report

    body = section("Per-fold means", "5 folds", df_to_html(fold_summary, index_label="fold"))
    html = report_shell(title=..., eyebrow=..., heading=..., dek=..., stat_tiles=stat_tiles, body=body, footer=footer)
    write_report(results_dir() / "my_report.html", html)
"""
import math
from pathlib import Path

import pandas as pd

REPORT_CSS = """
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
    max-width: 980px;
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
    max-width: 68ch;
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

  .group-body h3, .group-body h4 {
    font-family: var(--font-body);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .group-body h3 { font-size: 0.85rem; margin: 1.6rem 0 0.6rem; }
  .group-body h4 { font-size: 0.95rem; margin: 2rem 0 0.8rem; color: var(--text); text-transform: none; letter-spacing: 0; }
  .group-body h3:first-child, .group-body h4:first-child { margin-top: 0; }

  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    margin-bottom: 0.5rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.86rem;
    min-width: 480px;
  }

  thead th {
    text-align: right;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--surface-alt);
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border-strong);
    white-space: nowrap;
  }

  thead th:first-child, td.row-label { text-align: left; }

  tbody td {
    padding: 0.5rem 0.8rem;
    text-align: right;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  tbody tr:last-child td { border-bottom: none; }

  td.row-label {
    font-family: var(--font-body);
    font-size: 0.86rem;
    color: var(--text);
  }

  .note {
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.6;
  }

  .mono-block {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }

  figure.plot {
    margin: 0 0 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
  }

  figure.plot img {
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }

  figure.plot figcaption {
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--text-muted);
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


def results_dir() -> Path:
    """repo_root/results, created if missing. Stable regardless of caller's cwd or location."""
    d = Path(__file__).resolve().parent.parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_cell(v, float_fmt: str) -> str:
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "—" if math.isnan(v) else float_fmt.format(v)
    return str(v)


def df_to_html(df: pd.DataFrame, index_label: str = "", float_fmt: str = "{:.3f}") -> str:
    """Render a (possibly MultiIndex-column / MultiIndex-row) DataFrame as an HTML table."""
    if isinstance(df.columns, pd.MultiIndex):
        columns = [" ".join(str(c) for c in col if str(c)) for col in df.columns]
    else:
        columns = [str(c) for c in df.columns]

    header_cells = "".join(f"<th>{c}</th>" for c in columns)

    body_rows = []
    for idx, row in zip(df.index, df.itertuples(index=False, name=None)):
        idx_str = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        cells = "".join(f"<td>{_fmt_cell(v, float_fmt)}</td>" for v in row)
        body_rows.append(f"<tr><td class=\"row-label\">{idx_str}</td>{cells}</tr>")

    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"<th>{index_label}</th>{header_cells}"
        "</tr></thead><tbody>"
        f"{''.join(body_rows)}"
        "</tbody></table></div>"
    )


def section(title: str, source: str, body_html: str) -> str:
    source_html = f'<span class="group-source">{source}</span>' if source else ""
    return f"""  <section class="group">
    <div class="group-head">
      <h2>{title}</h2>
      {source_html}
    </div>
    <div class="group-body">
{body_html}
    </div>
  </section>"""


def stat_tile(label: str, value: str, of: str = "", ok: bool = True) -> str:
    color = "var(--success)" if ok else "var(--danger)"
    of_html = f'<span class="of">{of}</span>' if of else ""
    return (
        f'    <div class="stat">\n      <p class="stat-label">{label}</p>\n'
        f'      <div class="stat-value" style="--stat-color: {color}">{value} {of_html}</div>\n'
        f"    </div>"
    )


def image_figure(src: str, caption: str = "") -> str:
    cap_html = f"<figcaption>{caption}</figcaption>" if caption else ""
    alt = caption or "plot"
    return f'<figure class="plot"><img src="{src}" alt="{alt}">{cap_html}</figure>'


def report_shell(title: str, eyebrow: str, heading: str, dek: str, stat_tiles: str, body: str, footer: str) -> str:
    stat_row_html = f'  <div class="stat-row">\n{stat_tiles}\n  </div>\n\n' if stat_tiles.strip() else ""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{REPORT_CSS}
</style>
</head>
<body>
<div class="page">

  <header class="report-head">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{heading}</h1>
    <p class="dek">{dek}</p>
  </header>

{stat_row_html}{body}

  <footer class="report-foot">
{footer}
  </footer>

</div>
</body>
</html>
"""


def write_report(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def write_cv_report(path: Path, title: str, heading: str, fold_summary: pd.DataFrame, cv_summary: pd.DataFrame) -> Path:
    """Small standalone per-fold + across-fold CV report, for quick single-baseline `make baselineN` runs."""
    body = "\n\n".join([
        section("Per-fold means", "", df_to_html(fold_summary.set_index(["fold", "source"]), index_label="fold / source")),
        section("Across-fold mean ± std", "", df_to_html(cv_summary, index_label="source")),
    ])
    html = report_shell(
        title=title,
        eyebrow="HLS-CMDS · quick baseline run",
        heading=heading,
        dek="Single-baseline cross-validation run. See results/baselines_report.html for the full comparison across all baselines.",
        stat_tiles="",
        body=body,
        footer="",
    )
    return write_report(path, html)
