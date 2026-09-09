"""
The server-vs-board comparison as a figure, for the report and the paper.

FORM. Two panels, because the comparison has two halves that a single axis
would have to fake. The top panel is the finding -- the 276x throughput gap
decomposed into the part that is clock rate and the part that is architecture
-- and the bottom panel is the measurement it rests on, so the abstract ratio
stays anchored to real numbers.

WHY A LOG AXIS ON THE TOP PANEL. The decomposition is multiplicative:
276 = 56 x 4.9, not 56 + 4.9. On a log axis multiplication becomes addition,
so the two contributions compose as adjacent segments of one bar with no
distortion -- the segment lengths are literally log(56) and log(4.9). A linear
stacked bar would be arithmetically wrong for this data, and a grouped bar
would lose the "these two multiply to the whole" relationship entirely.

WHY NOT ONE PANEL OF RAW TIMES. Server and board differ by 276x in us/sample.
On a linear axis the server bar is 0.4 % of the board's and vanishes; on a log
axis two lone bars invite the reader to eyeball a ratio off a log scale, which
people do badly. Cycles per sample (bottom panel) is the normalised measure
where a linear axis is honest, and the ratio the reader should take away is
stated in the top panel rather than left to be estimated.

Usage:
    python plot_server_vs_board.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_utils import results_dir  # noqa: E402
from server_vs_board import (  # noqa: E402
    BOARD_CLOCK_HZ,
    BOARD_US_PER_SAMPLE,
    BOARD_US_PER_SAMPLE_SOFTFLOAT,
    SERVER_CLOCK_HZ,
    build_rows,
    measure,
)

# Categorical slots 1 and 2 from the skill's reference palette, validated for
# this pair: CVD dE 24.7 (protan) / 32.7 (tritan), normal-vision dE 33.6.
SERVER_COLOR = "#2a78d6"
BOARD_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"


def _style(ax):
    """Recessive chrome: hairline solid grid, no top/right spine, no dashes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.6)


def make_figure(clock_ratio, arch_ratio, server_cycles, board_cycles, out_stem):
    total = clock_ratio * arch_ratio
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.75}
    )
    fig.patch.set_facecolor(SURFACE)

    # --- Panel 1: the multiplicative decomposition, on a log axis -----------
    _style(ax1)
    ax1.set_xscale("log")
    ax1.set_xlim(1, total * 1.35)
    ax1.set_ylim(-0.32, 0.72)  # room for the segment labels above, tight below

    bar_h = 0.26
    # Segment 1: 1x -> clock_ratio. Segment 2: clock_ratio -> total.
    ax1.barh(0, clock_ratio - 1, left=1, height=bar_h,
             color=SERVER_COLOR, edgecolor=SURFACE, linewidth=2.0, zorder=3)
    ax1.barh(0, total - clock_ratio, left=clock_ratio, height=bar_h,
             color=BOARD_COLOR, edgecolor=SURFACE, linewidth=2.0, zorder=3)

    mid1 = np.sqrt(1 * clock_ratio)          # geometric midpoint = visual centre on log
    mid2 = np.sqrt(clock_ratio * total)
    ax1.text(mid1, 0.30, f"clock rate\n{clock_ratio:.0f}×", ha="center", va="bottom",
             fontsize=9, color=INK, linespacing=1.35)
    ax1.text(mid2, 0.30, f"architecture\n{arch_ratio:.1f}×", ha="center", va="bottom",
             fontsize=9, color=INK, linespacing=1.35)
    ax1.text(total * 1.12, 0, f"{total:.0f}×", ha="left", va="center",
             fontsize=11, color=INK, fontweight="bold")

    ax1.set_yticks([])
    # Ticks derived from the measured values, never hardcoded -- the server
    # side varies a few percent run to run, and a tick labelled with a stale
    # total sitting beside a bar that ends elsewhere is worse than no tick.
    ax1.set_xticks([1, 10, clock_ratio, total])
    ax1.set_xticklabels(["1×", "10×", f"{clock_ratio:.0f}×", f"{total:.0f}×"])
    ax1.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title(f"How the board's {total:.0f}× slowdown decomposes",
                  fontsize=10.5, color=INK, loc="left", pad=10)
    ax1.set_xlabel("slower than the server (log scale; the two factors multiply)",
                   fontsize=8.5, color=INK_MUTED, labelpad=6)

    # --- Panel 2: the normalised measurement, linear ------------------------
    _style(ax2)
    labels = ["Server\nCore i9-9900K, 3.6 GHz", "Board\nCortex-M4F, 64 MHz"]
    values = [server_cycles, board_cycles]
    colors = [SERVER_COLOR, BOARD_COLOR]
    ypos = [1, 0]

    ax2.barh(ypos, values, height=0.34, color=colors,
             edgecolor=SURFACE, linewidth=2.0, zorder=3)
    for y, v in zip(ypos, values):
        ax2.text(v + board_cycles * 0.025, y, f"{v:.0f}", va="center", ha="left",
                 fontsize=10, color=INK, fontweight="bold")

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=9, color=INK)
    ax2.set_xlim(0, board_cycles * 1.18)
    ax2.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Cycles per sample — the same work, normalised for clock",
                  fontsize=10.5, color=INK, loc="left", pad=10)
    ax2.set_xlabel("clock cycles per output sample (one causal pass per band, both platforms)",
                   fontsize=8.5, color=INK_MUTED, labelpad=6)

    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    # Median over several independent measure() calls, not one. A single call
    # already takes the median of its own repetitions, but the *run* itself
    # varies: seven runs gave 45.1-47.3 cycles/sample, tight except for one
    # outlier, and letting whichever run happened to be last set the number in
    # a published figure is how a 251x and a 277x both end up quoted for the
    # same measurement.
    N_RUNS = 5
    print(f"Re-timing the server ({N_RUNS} runs), then plotting...")
    per_run = []
    for i in range(N_RUNS):
        m = measure()
        rows = {r["platform"]: r for r in build_rows(m)}
        per_run.append(rows["Server, CPU time only"]["cycles/sample"])
        print(f"  run {i + 1}: {per_run[-1]:.1f} cycles/sample")

    server_cycles = float(np.median(per_run))
    board_cycles = rows["Board (Nano 33 BLE Sense, 64 MHz)"]["cycles/sample"]
    print(f"  median {server_cycles:.1f}, spread "
          f"{100 * (max(per_run) - min(per_run)) / server_cycles:.0f} %")
    clock_ratio = SERVER_CLOCK_HZ / BOARD_CLOCK_HZ
    arch_ratio = board_cycles / server_cycles

    out_dir = results_dir() / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / "server_vs_board")
    make_figure(clock_ratio, arch_ratio, server_cycles, board_cycles, stem)

    print(f"  server {server_cycles:.1f} cycles/sample, board {board_cycles:.1f}")
    print(f"  {clock_ratio:.0f}x clock x {arch_ratio:.1f}x architecture "
          f"= {clock_ratio * arch_ratio:.0f}x total")
    print(f"\nWrote {stem}.png and {stem}.pdf")
