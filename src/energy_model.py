"""
Energy per clip: server versus microcontroller, and the one number still missing.

WHY THIS EXISTS. server_vs_board.py establishes that the board is 276x slower.
That is the wrong axis for a deployment argument. A device that runs on a
battery is judged on joules, not seconds, and the two need not agree: a
processor 276x slower that draws 1000x less power still wins on energy. This
module works out which way that actually falls here.

WHAT IS MEASURED, AND WHAT IS NOT.

  Server: MEASURED, and without any external instrumentation. Intel CPUs
  expose the package energy counter through RAPL at
  /sys/class/powercap/intel-rapl:0/energy_uj, in microjoules, readable as an
  ordinary file. Sampling it either side of a known workload gives real
  energy, not a TDP estimate.

  Board: NOT MEASURED. The Nano 33 BLE Sense has no onboard current sense,
  so the firmware cannot report its own consumption, and nothing on the
  server can infer it. An external measurement is required -- a USB power
  meter inline with the board is enough for an order-of-magnitude answer, a
  Nordic PPK2 for a precise one.

  This module therefore treats board power as a free parameter and reports
  the BREAK-EVEN value: the consumption at which the two platforms cost the
  same energy per clip. That converts "we don't know" into "here is exactly
  what the missing measurement decides", which is the useful form.

WHY A DATASHEET FIGURE IS NOT SUBSTITUTED. It would be one number in this
project not derived from either a measurement or the code's own constants,
and it would not settle the question anyway -- see the break-even below,
which lands close enough to a plausible board consumption that the answer
flips inside the range a datasheet would support. When the answer is
insensitive to an assumption, assuming is fine; here it is not.

TWO SERVER BASELINES, because the honest figure depends on the question.
  - marginal: the extra power drawn while computing, over idle. The right
    number if the server exists regardless and the question is what one more
    clip costs.
  - full package: everything the CPU package draws during the run. The right
    number if the comparison is a dedicated device against a dedicated
    machine.
Both are reported; neither is the single true answer.

Usage:
    python energy_model.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402
from load_dataset import load_mix  # noqa: E402
from server_vs_board import BOARD_MS_PER_CLIP, BOARD_NAME  # noqa: E402

RAPL_PACKAGE = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
IDLE_SECONDS = 3.0
FILTER_ORDER = 4

# Consumptions to tabulate. Spans a bare MCU core running from a coin cell up
# to a devkit with USB, regulators, LEDs and onboard sensors all awake -- the
# range an actual measurement of this board could plausibly land in.
BOARD_POWER_SWEEP_MW = (10, 25, 50, 75, 100, 150, 250)


def read_rapl_uj() -> int:
    return int(RAPL_PACKAGE.read_text())


def rapl_available() -> bool:
    try:
        read_rapl_uj()
        return True
    except (OSError, ValueError):
        return False


def _sos(low, high, sr):
    nyq = sr / 2.0
    return butter(FILTER_ORDER, [low / nyq, high / nyq],
                  btype="bandpass", output="sos").astype(np.float32)


def measure_server(n_reps: int = 4000) -> dict:
    """
    Idle package power, then package power under the board's exact workload.
    RAPL counts the whole package, so the marginal figure -- busy minus idle
    -- is what isolates the cost of the computation from the machine simply
    being switched on.
    """
    row = load_mix().iloc[0]
    x, sr = sf.read(row["mixed_audio_path"], dtype="int16")
    mixed = x.astype(np.float32) / np.float32(32768.0)
    heart_sos, lung_sos = _sos(*HEART_BAND, sr), _sos(*LUNG_BAND, sr)

    time.sleep(0.3)
    e0, t0 = read_rapl_uj(), time.perf_counter()
    time.sleep(IDLE_SECONDS)
    idle_w = (read_rapl_uj() - e0) / 1e6 / (time.perf_counter() - t0)

    e0, t0 = read_rapl_uj(), time.perf_counter()
    for _ in range(n_reps):
        sosfilt(heart_sos, mixed)
        sosfilt(lung_sos, mixed)
    elapsed = time.perf_counter() - t0
    busy_w = (read_rapl_uj() - e0) / 1e6 / elapsed

    seconds_per_clip = elapsed / n_reps
    return {
        "idle_w": idle_w,
        "busy_w": busy_w,
        "marginal_w": busy_w - idle_w,
        "seconds_per_clip": seconds_per_clip,
        "mj_per_clip_marginal": (busy_w - idle_w) * seconds_per_clip * 1000,
        "mj_per_clip_package": busy_w * seconds_per_clip * 1000,
        "n_reps": n_reps,
    }


def board_mj_per_clip(power_mw: float) -> float:
    return power_mw / 1000.0 * (BOARD_MS_PER_CLIP / 1000.0) * 1000


def break_even_mw(server_mj: float) -> float:
    """Board consumption at which both platforms cost the same per clip."""
    return server_mj / (BOARD_MS_PER_CLIP / 1000.0)


if __name__ == "__main__":
    import pandas as pd

    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    if not rapl_available():
        sys.exit(f"RAPL not readable at {RAPL_PACKAGE} -- this needs an Intel CPU "
                 "exposing powercap, and read permission on that file.")

    # Repeated, and the spread reported. Idle package power drifts with
    # ambient load and temperature far more than the timing does -- two early
    # runs gave 18.2 and 13.6 mJ/clip marginal, a 34 % swing, which moves the
    # break-even from 87 to 65 mW. A single run is not a number to publish.
    N_RUNS = 5
    print(f"Measuring server energy via RAPL ({N_RUNS} runs: idle baseline, then workload)...")
    runs = []
    for i in range(N_RUNS):
        r = measure_server()
        runs.append(r)
        print(f"  run {i + 1}: idle {r['idle_w']:5.2f} W  marginal {r['marginal_w']:5.2f} W  "
              f"-> {r['mj_per_clip_marginal']:5.2f} mJ/clip")

    def med(key):
        return float(np.median([r[key] for r in runs]))

    s = {k: med(k) for k in ("idle_w", "busy_w", "marginal_w", "seconds_per_clip",
                             "mj_per_clip_marginal", "mj_per_clip_package")}
    s["n_reps"] = runs[0]["n_reps"]
    marg = [r["mj_per_clip_marginal"] for r in runs]
    s["marginal_spread_pct"] = 100 * (max(marg) - min(marg)) / s["mj_per_clip_marginal"]
    print(f"  median {s['mj_per_clip_marginal']:.1f} mJ/clip marginal "
          f"(spread {s['marginal_spread_pct']:.0f} %), "
          f"{s['mj_per_clip_package']:.1f} mJ/clip package\n")

    be_marginal = break_even_mw(s["mj_per_clip_marginal"])
    be_package = break_even_mw(s["mj_per_clip_package"])
    print(f"Break-even board power: {be_marginal:.0f} mW vs marginal, "
          f"{be_package:.0f} mW vs full package")

    rows = []
    for mw in BOARD_POWER_SWEEP_MW:
        mj = board_mj_per_clip(mw)
        rows.append({
            "board power (mW)": mw,
            "board mJ/clip": mj,
            "vs server marginal": f"{mj / s['mj_per_clip_marginal']:.2f}x",
            "vs server package": f"{mj / s['mj_per_clip_package']:.2f}x",
            "cheaper than server?": "board" if mj < s["mj_per_clip_marginal"]
                                    else ("board, vs package only" if mj < s["mj_per_clip_package"]
                                          else "server"),
        })
    df = pd.DataFrame(rows).set_index("board power (mW)")
    print("\n" + df.to_string())

    display = df.copy()
    display["board mJ/clip"] = display["board mJ/clip"].map(lambda v: f"{v:.2f}")

    stat_tiles = "\n".join([
        stat_tile("Server, measured", f"{s['mj_per_clip_marginal']:.1f} mJ",
                  "per clip, marginal over idle (RAPL)"),
        stat_tile("Break-even", f"{be_marginal:.0f} mW",
                  "board power for parity on energy"),
        stat_tile("Board power", "unmeasured", "needs an inline meter", ok=False),
    ])

    body = "\n".join([
        section("What the server actually costs", "measured via RAPL, no external hardware",
                f"<p>Package power idle <strong>{s['idle_w']:.2f} W</strong>, under the board's "
                f"workload <strong>{s['busy_w']:.2f} W</strong>, so the computation itself draws "
                f"<strong>{s['marginal_w']:.2f} W</strong> above idle. At "
                f"{s['seconds_per_clip'] * 1000:.2f} ms per clip that is "
                f"<strong>{s['mj_per_clip_marginal']:.1f} mJ</strong> marginal, or "
                f"<strong>{s['mj_per_clip_package']:.1f} mJ</strong> counting the whole package. "
                "Intel's RAPL counters make this a real measurement rather than a TDP estimate, "
                "and they need no instrumentation &mdash; the counter is a file.</p>"
                f"<p>Median of {N_RUNS} runs, spread {s['marginal_spread_pct']:.0f} %. Idle package "
                "power drifts with ambient load and temperature considerably more than the timing "
                "measurements do, so the marginal figure is the noisier of the two and is reported "
                "as a median rather than a single reading.</p>"),
        section("What the board would cost, per assumed consumption", "board power is NOT measured",
                df_to_html(display, index_label="board power (mW)")
                + f"<p>The board's time per clip is measured ({BOARD_MS_PER_CLIP:.1f} ms on a "
                  f"{BOARD_NAME}); only its power is not. Energy is the product, so a single "
                  "missing number decides the entire comparison.</p>"
                  f"<p><strong>The break-even is {be_marginal:.0f} mW against the server's marginal "
                  f"cost, {be_package:.0f} mW against its full package.</strong> That is the "
                  "uncomfortable part: a devkit with USB, regulators and onboard sensors awake can "
                  "plausibly sit on either side of 87 mW, so the answer flips inside the range a "
                  "reasonable guess would cover. Where a comparison is insensitive to an "
                  "assumption, assuming is fine; this one is not, which is precisely why no "
                  "datasheet figure is substituted here.</p>"),
        section("What would settle it", "one measurement, modest equipment",
                "<p>A USB power meter inline between host and board, run during "
                "<code>*-bench</code>, is enough for the order-of-magnitude answer this argument "
                "needs. A Nordic PPK2 would give a precise figure including sleep current, which "
                "matters for a duty-cycled device but not for this comparison, where the board is "
                "computing continuously.</p>"
                "<p>Worth measuring in both firmware builds: the pre-FPU-fix binary took 18x "
                "longer for identical output, so if power draw is comparable between them, that "
                "build also cost 18x the energy &mdash; a build flag worth an order of magnitude "
                "on battery life.</p>"),
    ])

    html = report_shell(
        title="Energy per clip",
        eyebrow="HLS-CMDS · energy, not just latency",
        heading="Server against microcontroller on joules",
        dek=("Throughput says the board is 276x slower. Battery life is decided by energy, not "
             "time, and the two can disagree. The server side is measured here via RAPL; the "
             "board's power is not measured, so what is reported instead is the break-even value "
             "that the missing measurement would resolve."),
        stat_tiles=stat_tiles,
        body=body,
        footer=(f"<p><strong>Method.</strong> Server: RAPL package counter, {IDLE_SECONDS:.0f} s "
                f"idle baseline then {s['n_reps']} clips of the board's exact workload. Board: "
                f"time measured on silicon, power an open parameter. Server figures are the "
                f"median of {N_RUNS} runs.</p>"),
    )
    path = write_report(results_dir() / "energy_model_report.html", html)
    print(f"\nReport written to {path}")
