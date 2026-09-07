"""
Measure what the firmware's causal filter costs against the Python baseline's
zero-phase one -- the one deliberate numerical difference between them.

Baseline 1 (src/baseline/baseline1.py) applies the Butterworth bands with
scipy.signal.sosfiltfilt: forward then backward, zero phase, but it needs the
whole 15 s clip in memory and cannot run while audio is arriving. The firmware
runs the same coefficients through scipy.signal.sosfilt's equivalent -- one
causal pass, O(1) memory, streaming. That is the right trade for an MCU, but it
is a real change to the output, so this quantifies it on the same 145 Mix rows
and the same BSS Eval metrics the rest of the project reports.

Nothing here runs on the board; it exists so firmware/README.md can state the
cost as a measured number rather than an assurance.

Usage:
    python firmware/tools/causal_vs_zerophase.py     # -> results/firmware_causal_vs_zerophase.html
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.signal import sosfilt, sosfiltfilt

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402
from gen_filter_coeffs import design_bandpass as _sos  # noqa: E402  (one design call, one place)
from metrics import evaluate_dataset  # noqa: E402
from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report  # noqa: E402


def make_separator(filter_fn):
    """separate_fn(mixed, sr) -> (heart_est, lung_est), per the metrics.py contract."""
    def separate(mixed, sr):
        return (filter_fn(_sos(*HEART_BAND, sr), mixed),
                filter_fn(_sos(*LUNG_BAND, sr), mixed))
    return separate


def main():
    print("Baseline 1 with sosfiltfilt (Python baseline, zero-phase) ...")
    zero_phase = evaluate_dataset(make_separator(sosfiltfilt))
    print("Baseline 1 with sosfilt (firmware, causal single-pass) ...")
    causal = evaluate_dataset(make_separator(sosfilt))

    zero_phase["filter"] = "zero-phase (sosfiltfilt)"
    causal["filter"] = "causal (sosfilt)"
    both = pd.concat([zero_phase, causal], ignore_index=True)

    summary = both.groupby(["source", "filter"])[["sdr", "sir", "sar"]].mean()
    delta = (causal.groupby("source")[["sdr", "sir", "sar"]].mean()
             - zero_phase.groupby("source")[["sdr", "sir", "sar"]].mean())
    delta.index = [f"{s} (causal - zero-phase)" for s in delta.index]

    print("\nMean over 145 Mix rows:")
    print(summary.round(3).to_string())
    print("\nDelta:")
    print(delta.round(3).to_string())

    tiles = "\n".join([
        stat_tile("Heart SDR delta", f"{delta.loc['heart (causal - zero-phase)', 'sdr']:+.2f} dB",
                  "causal vs zero-phase"),
        stat_tile("Lung SDR delta", f"{delta.loc['lung (causal - zero-phase)', 'sdr']:+.2f} dB",
                  "causal vs zero-phase"),
        stat_tile("Rows", f"{len(both) // 2}", "Mix.csv triplets, both filters"),
    ])

    body = "\n".join([
        section("Mean SDR / SIR / SAR by source and filter", "firmware/tools/causal_vs_zerophase.py",
                df_to_html(summary, index_label="source / filter")),
        section("What the firmware gives up", "",
                df_to_html(delta, index_label="")
                + f"""
<p><b>SDR is essentially unchanged</b> ({delta.loc['heart (causal - zero-phase)', 'sdr']:+.2f} dB heart,
{delta.loc['lung (causal - zero-phase)', 'sdr']:+.2f} dB lung). On this dataset the causal filter costs
nothing on the headline metric.</p>
<p><b>SIR drops slightly</b> ({delta.loc['heart (causal - zero-phase)', 'sir']:+.2f} dB heart,
{delta.loc['lung (causal - zero-phase)', 'sir']:+.2f} dB lung), which is what the halved effective filter
order predicts: <code>sosfiltfilt</code> applies the same cascade twice, so its stopbands are twice as
deep in dB, and rejection of the other source is exactly what stopband depth buys.</p>
<p><b>SAR rises sharply</b> ({delta.loc['heart (causal - zero-phase)', 'sar']:+.2f} dB heart,
{delta.loc['lung (causal - zero-phase)', 'sar']:+.2f} dB lung) &mdash; and this one is a property of the
metric, not a quality gain. <code>mir_eval.separation._project</code> fits a <em>causal</em> 512-tap
distortion filter (&ldquo;delays between 0 and flen-1&rdquo;). A causal IIR's phase response lies inside
that subspace and is absorbed as allowed distortion; a zero-phase filter's acausal response does not fit
it, so part of the zero-phase output is scored as artifacts. Do not report this as the firmware separating
more cleanly.</p>
<p>Both filters sit at roughly &minus;12 to &minus;14 dB SDR. Baseline 1 is the project's zero-training
sanity floor, not a working separator; these numbers say what the <em>port</em> costs, not that the method
is good.</p>
<p>What the firmware buys in exchange: no clip buffer. Filter state is a few hundred bytes regardless of
recording length, against 120 kB for a buffered 15 s int16 clip plus a 240 kB float32 working copy for
<code>filtfilt</code> &mdash; on a part with 256 kB of RAM in total.</p>"""),
    ])

    out = results_dir() / "firmware_causal_vs_zerophase.html"
    write_report(out, report_shell(
        title="Causal vs zero-phase bandpass",
        eyebrow="Firmware",
        heading="What the MCU's causal filter costs against Baseline 1",
        dek="Same Butterworth coefficients, same 145 Mix rows, same BSS Eval metrics -- "
            "the only change is sosfiltfilt (Python baseline) vs sosfilt (firmware).",
        stat_tiles=tiles,
        body=body,
        footer=f"<p>Generated {date.today().isoformat()} by firmware/tools/causal_vs_zerophase.py</p>",
    ))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
