"""
Server versus microcontroller, on identical work.

WHY THIS IS NOT JUST TWO NUMBERS DIVIDED. latency.py already times Baseline 1
on this workstation and firmware/ has measured it on a Nano 33 BLE Sense, but
those two figures are not comparable as they stand, for two reasons that both
favour the board if ignored:

  1. DIFFERENT WORK. latency.py times `bandpass_separate`, which calls
     scipy.signal.sosfiltfilt -- forwards then backwards, two passes per band.
     The firmware runs one causal pass per band (see report Sec. "Embedded
     Deployment"), so the desktop figure is for roughly twice the arithmetic.
     This module therefore re-times the desktop doing *exactly* what the board
     does, rather than halving the existing number and hoping the constant
     factors cancel.

  2. DIFFERENT CLOCKS. 3.6 GHz against 64 MHz is a 56x clock ratio before any
     architectural difference is considered. A raw wall-clock ratio conflates
     "this processor is faster" with "this processor does more per cycle",
     which are different claims and only the second is interesting. Cycles per
     sample separates them.

WHAT THE BOARD FIGURE IS. Measured, not modelled: three real HLS-CMDS
mixtures replayed from flash, timed on-device around the int16 scaling and
both band cascades with serial I/O excluded, after the FPU fix (before it,
the same board reported 62.4 us/sample because every float operation was
going through software-float calls). Recorded here as a constant with its
provenance rather than re-derived, since reproducing it needs the hardware.

WHAT THIS COMPARISON IS FOR. Not to declare a winner -- a workstation is
obviously faster in absolute terms, and nobody deploys an i9 to a chest
strap. It is to establish what the port actually costs in throughput, and
whether that cost is explained by clock rate alone or by the architecture
doing genuinely less per cycle, which is the question a deployment decision
turns on.

Usage:
    python server_vs_board.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402
from latency import time_calls  # noqa: E402
from load_dataset import load_mix  # noqa: E402

SAMPLE_RATE = 4000

# --- The board's measured figures -------------------------------------------
# firmware/, Nano 33 BLE Sense (nRF52840, Cortex-M4F at 64 MHz), CMSIS-DSP
# arm_biquad_cascade_df2T_f32, three additive mixtures, median of the three.
BOARD_NAME = "Nano 33 BLE Sense"
BOARD_CLOCK_HZ = 64e6
BOARD_US_PER_SAMPLE = 3.474
BOARD_MS_PER_CLIP = 208.5
BOARD_PROVENANCE = (
    "measured on silicon, 3 clips, median; results/firmware_on_device_report.html"
)

# Same part before the -mfloat-abi build fix, kept because the comparison it
# supports -- what a build flag was worth -- is one of this project's findings.
BOARD_US_PER_SAMPLE_SOFTFLOAT = 62.45

SERVER_CLOCK_HZ = 3.6e9  # Intel i9-9900K base clock; boost is higher, so this
                         # flatters the server's per-cycle figure if anything.


# Same design call baseline.common._bandpass makes -- order, normalisation and
# output form all taken from there rather than restated, so the two cannot
# drift apart. Only sosfiltfilt -> sosfilt differs, which is the whole point.
FILTER_ORDER = 4


def _sos(low: float, high: float, sr: int):
    from scipy.signal import butter

    nyq = sr / 2.0
    return butter(FILTER_ORDER, [low / nyq, high / nyq], btype="bandpass", output="sos")


def _causal_band_split(mixed: np.ndarray, sr: int):
    """Exactly the firmware's arithmetic: one causal pass per band, no filtfilt."""
    heart = sosfilt(_sos(*HEART_BAND, sr), mixed)
    lung = sosfilt(_sos(*LUNG_BAND, sr), mixed)
    return heart, lung


def _zero_phase_band_split(mixed: np.ndarray, sr: int):
    """What the Python baseline actually does, for the like-for-unlike row."""
    from baseline.baseline1 import bandpass_separate

    return bandpass_separate(mixed, sr)


def measure(n_trials: int = 7) -> dict:
    mix = load_mix()
    row = mix.iloc[0]
    mixed, sr = sf.read(row["mixed_audio_path"], dtype="int16")
    mixed = mixed.astype(np.float64) / 32768.0
    n = len(mixed)

    causal = time_calls(lambda: _causal_band_split(mixed, sr), n_trials=n_trials)
    zero_phase = time_calls(lambda: _zero_phase_band_split(mixed, sr), n_trials=n_trials)

    load1 = os.getloadavg()[0]
    return {
        "n_samples": n,
        "sample_rate": sr,
        "causal": causal,
        "zero_phase": zero_phase,
        "loadavg_1": load1,
        "n_cpus": os.cpu_count(),
    }


def build_rows(m: dict) -> list[dict]:
    n = m["n_samples"]
    audio_s = n / m["sample_rate"]

    def per_sample_us(ms):
        return ms * 1000.0 / n

    def cycles_per_sample(us, clock_hz):
        return us * 1e-6 * clock_hz

    server_wall_ms = m["causal"]["wall_median_s"] * 1000.0
    server_cpu_ms = m["causal"]["cpu_median_s"] * 1000.0
    server_us_wall = per_sample_us(server_wall_ms)
    server_us_cpu = per_sample_us(server_cpu_ms)

    return [
        {
            "platform": "Server (i9-9900K, 3.6 GHz)",
            "work": "causal, 1 pass/band -- same as the board",
            "ms/clip": server_wall_ms,
            "us/sample": server_us_wall,
            "cycles/sample": cycles_per_sample(server_us_wall, SERVER_CLOCK_HZ),
            "x real-time": audio_s * 1000.0 / server_wall_ms,
        },
        {
            "platform": "Server, CPU time only",
            "work": "causal, 1 pass/band; excludes scheduler contention",
            "ms/clip": server_cpu_ms,
            "us/sample": server_us_cpu,
            "cycles/sample": cycles_per_sample(server_us_cpu, SERVER_CLOCK_HZ),
            "x real-time": audio_s * 1000.0 / server_cpu_ms,
        },
        {
            "platform": f"Board ({BOARD_NAME}, 64 MHz)",
            "work": "causal, 1 pass/band",
            "ms/clip": BOARD_MS_PER_CLIP,
            "us/sample": BOARD_US_PER_SAMPLE,
            "cycles/sample": cycles_per_sample(BOARD_US_PER_SAMPLE, BOARD_CLOCK_HZ),
            "x real-time": audio_s * 1000.0 / BOARD_MS_PER_CLIP,
        },
        {
            "platform": "Board, before the FPU fix",
            "work": "causal, 1 pass/band, software float",
            "ms/clip": BOARD_US_PER_SAMPLE_SOFTFLOAT * n / 1000.0,
            "us/sample": BOARD_US_PER_SAMPLE_SOFTFLOAT,
            "cycles/sample": cycles_per_sample(BOARD_US_PER_SAMPLE_SOFTFLOAT, BOARD_CLOCK_HZ),
            "x real-time": audio_s * 1e6 / (BOARD_US_PER_SAMPLE_SOFTFLOAT * n),
        },
    ]


if __name__ == "__main__":
    import pandas as pd

    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print("Timing the server on exactly the board's workload (causal, one pass per band)...")
    m = measure()
    print(f"  load average {m['loadavg_1']:.1f} on {m['n_cpus']} CPUs\n")

    df = pd.DataFrame(build_rows(m)).set_index("platform")
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))

    server_cpu_us = df.loc["Server, CPU time only", "us/sample"]
    board_us = df.loc[f"Board ({BOARD_NAME}, 64 MHz)", "us/sample"]
    slowdown = board_us / server_cpu_us
    clock_ratio = SERVER_CLOCK_HZ / BOARD_CLOCK_HZ
    per_cycle = df.loc[f"Board ({BOARD_NAME}, 64 MHz)", "cycles/sample"] / \
        df.loc["Server, CPU time only", "cycles/sample"]

    print(f"\nBoard is {slowdown:.1f}x slower per sample than the server's CPU time,")
    print(f"against a {clock_ratio:.0f}x clock disadvantage.")
    print(f"Per clock cycle the board spends {per_cycle:.2f}x what the server does.")

    # The zero-phase row is what latency.py reports; kept to show the confound
    # this module exists to remove.
    zp_us = m["zero_phase"]["wall_median_s"] * 1000.0 * 1000.0 / m["n_samples"]
    print(f"\n(For reference: the server doing sosfiltfilt -- what latency.py times -- "
          f"is {zp_us:.3f} us/sample, {zp_us / df.loc['Server (i9-9900K, 3.6 GHz)', 'us/sample']:.1f}x "
          f"the causal figure, which is the confound this module removes.)")

    display = df.copy()
    for c in ["ms/clip", "us/sample", "cycles/sample", "x real-time"]:
        display[c] = display[c].map(lambda v: f"{v:,.2f}")

    stat_tiles = "\n".join([
        stat_tile("Board vs server", f"{slowdown:.0f}x", "slower per sample (CPU time)"),
        stat_tile("Clock disadvantage", f"{clock_ratio:.0f}x", "64 MHz vs 3.6 GHz"),
        stat_tile("Per clock cycle", f"{per_cycle:.2f}x", "board / server work per cycle",
                  ok=per_cycle < 2.0),
    ])

    body = "\n".join([
        section("Identical workload on both platforms", "server_vs_board.py",
                df_to_html(display, index_label="platform")
                + "<p>Every row filters the same 60&nbsp;000-sample mixture through the same two "
                  "Butterworth cascades. The server rows were re-timed for this comparison rather "
                  "than taken from <code>latency.py</code>, which times "
                  "<code>sosfiltfilt</code> &mdash; two passes per band, roughly twice the "
                  "arithmetic the firmware performs.</p>"),
        section("Reading the ratio", "what the numbers do and do not say",
                f"<p>The board is <strong>{slowdown:.0f}x</strong> slower per sample than the "
                f"server's CPU time, against a <strong>{clock_ratio:.0f}x</strong> clock "
                f"disadvantage. Dividing one by the other is the architecture-fair figure: per "
                f"clock cycle the board spends <strong>{per_cycle:.2f}x</strong> what the server "
                "does on the same work.</p>"
                "<p>That is the result worth carrying into a deployment argument. A biquad cascade "
                "is inherently serial &mdash; each output sample depends on the previous two &mdash; "
                "so the server's width, out-of-order execution and cache hierarchy have little to "
                "work with, and most of its advantage reduces to clock rate. A microcontroller is "
                "not giving up much per cycle here; it is simply clocked slower, and 72x real-time "
                "headroom means that is affordable.</p>"
                "<p>The last row is the same silicon before the <code>-mfloat-abi</code> build fix "
                "(report Sec. &ldquo;A plausible measurement that was wrong by 18x&rdquo;), kept "
                "because it shows what the comparison would have concluded from a figure that "
                "looked entirely reasonable at the time.</p>"),
    ])

    html = report_shell(
        title="Server vs Board",
        eyebrow="HLS-CMDS · same workload, two platforms",
        heading="Baseline 1 on a workstation and on a microcontroller",
        dek=("A like-for-like throughput comparison: the server re-timed doing exactly the "
             "firmware's causal single-pass band split, against the board's measured on-silicon "
             "figure, normalised per clock cycle so that clock rate and architecture can be told "
             "apart."),
        stat_tiles=stat_tiles,
        body=body,
        footer=(f"<p><strong>Method.</strong> Server rows: median of 7 timed repetitions after one "
                f"untimed warm-up, load average {m['loadavg_1']:.1f} on {m['n_cpus']} CPUs at "
                f"measurement time. Board row: {BOARD_PROVENANCE}.</p>"),
    )
    path = write_report(results_dir() / "server_vs_board_report.html", html)
    print(f"\nReport written to {path}")
