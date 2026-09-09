"""
Drive the dataset-replay bench firmware and score what the board sends back.

The board has real HLS-CMDS mixtures in flash and filters them with the same
coefficients the Python baseline uses, on bit-identical input samples. So this
script can do two things no microphone capture could:

  1. Diff the board's output against scipy sample by sample. Any deviation
     beyond the float32 quantisation floor is a bug in the CMSIS-DSP / ESP-DSP
     path, not a difference in what was recorded.
  2. Score the board's output with the same BSS Eval metrics as the rest of the
     project, against the dataset's own isolated heart/lung references -- so the
     on-device SDR sits in the same table as everything in results/.

It also collects the board's own timing, measured around the filtering only,
which is the figure that replaces src/latency.py's desktop numbers.

Two ways in. Directly off the board, if this machine has both the board and the
dataset; or from a file captured by tools/capture_raw.py on a machine that has
only the board -- scoring needs the HLS-CMDS references, flashing does not.

Usage:
    python firmware/tools/run_on_device.py --port /dev/ttyACM0
    python firmware/tools/run_on_device.py --from-file capture.bin
"""
import argparse
import struct
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import sosfilt

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402
from gen_filter_coeffs import design_bandpass  # noqa: E402  (one design call, one place)
from load_dataset import load_mix  # noqa: E402
from metrics import evaluate_heart_lung  # noqa: E402
from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report  # noqa: E402

CLIP_MAGIC = b"HLSC"
FRAME_MAGIC = b"HLS1"
FOOTER_MAGIC = b"HLSE"
DONE_MAGIC = b"HLSD"

CLIP_HEADER = struct.Struct("<4s12sII")
FRAME_HEADER = struct.Struct("<4sHHI")
CLIP_FOOTER = struct.Struct("<4sII")
DONE = struct.Struct("<4sI")

# The board computes in float32; the scipy reference here is float64. The gap
# between them is not a constant -- it scales with signal amplitude, clip
# length and the cascade's Q, because an IIR accumulates rounding through its
# own state. A fixed absolute tolerance therefore only holds for the amplitude
# it was calibrated on: 1e-5 was fine for clips peaking near 0.05, and failed
# the moment additive (peak-normalised, full-scale) clips were embedded, on a
# kernel that had already passed its on-board self-test.
#
# So the floor is measured per clip instead of assumed: run the same filter
# over the same samples in float32 with scipy itself, and the deviation that
# produces is what any float32 implementation is entitled to. The board is
# then held to a small multiple of it. Observed board/floor ratios on real
# hardware are 0.5-1.6 (the board is as accurate as scipy-in-float32, and
# sometimes more), while a sign error, a wrong coefficient or a dropped
# section lands 3+ orders of magnitude higher -- so 4x separates them with
# room to spare in both directions.
FLOOR_MARGIN = 4.0
# Guards a silent or near-silent clip, where the measured floor would be ~0.
FLOOR_MIN = 1.0e-7


def float32_floor(sos, mixed, ref64):
    """
    How far a correct float32 implementation lands from the float64 reference,
    for this exact signal and filter -- measured by running scipy itself in
    float32 rather than assumed from a constant.
    """
    ref32 = sosfilt(sos.astype(np.float32), mixed.astype(np.float32))
    return max(float(np.max(np.abs(ref32.astype(np.float64) - ref64))), FLOOR_MIN)


def read_exact(port, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = port.read(n - len(buf))
        if not chunk:
            raise TimeoutError(f"serial timeout after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


class FileSource:
    """Adapts a captured .bin to the same read() contract as a serial port."""

    def __init__(self, path):
        self._data = Path(path).read_bytes()
        start = self._data.find(CLIP_MAGIC)
        if start < 0:
            raise ValueError(f"{path} contains no clip record -- was the capture truncated?")
        self._pos = start

    def read(self, n):
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


def wait_for_banner(port, echo=True):
    """Consume the firmware's '#' banner up to its prompt, then start the run."""
    while True:
        line = port.readline()
        if not line:
            raise TimeoutError("no banner from the board -- is the *-bench firmware flashed?")
        text = line.decode("ascii", "replace").strip()
        if echo and text.startswith("#"):
            print(f"  {text}")
        if "send 's'" in text:
            return


def receive(port):
    """Parse one full run: clip records until the done record. Returns a list of dicts."""
    clips = []
    while True:
        magic = read_exact(port, 4)
        if magic == DONE_MAGIC:
            read_exact(port, DONE.size - 4)
            return clips
        if magic != CLIP_MAGIC:
            raise ValueError(f"expected a clip or done record, got {magic!r}")

        _, clip_id, n_samples, sample_rate = CLIP_HEADER.unpack(
            magic + read_exact(port, CLIP_HEADER.size - 4))
        clip_id = clip_id.rstrip(b"\x00").decode("ascii")
        heart_parts, lung_parts = [], []
        print(f"  receiving {clip_id} ({n_samples} samples) ...")

        while True:
            magic = read_exact(port, 4)
            if magic == FOOTER_MAGIC:
                _, echoed_n, compute_micros = CLIP_FOOTER.unpack(
                    magic + read_exact(port, CLIP_FOOTER.size - 4))
                if echoed_n != n_samples:
                    raise ValueError(f"{clip_id}: footer says {echoed_n} samples, header said {n_samples}")
                break
            if magic != FRAME_MAGIC:
                raise ValueError(f"{clip_id}: expected a frame or footer, got {magic!r}")
            _, n, _seq, _dropped = FRAME_HEADER.unpack(magic + read_exact(port, FRAME_HEADER.size - 4))
            both = np.frombuffer(read_exact(port, n * 2 * 4), dtype="<f4")
            heart_parts.append(both[:n])
            lung_parts.append(both[n:])

        heart = np.concatenate(heart_parts)
        lung = np.concatenate(lung_parts)
        if len(heart) != n_samples:
            raise ValueError(f"{clip_id}: received {len(heart)} samples, header said {n_samples}")
        clips.append({"clip_id": clip_id, "n_samples": n_samples, "sample_rate": sample_rate,
                      "compute_micros": compute_micros, "heart": heart, "lung": lung})
    return clips


def score(clips, mix_df):
    """Per clip: board-vs-scipy deviation, BSS Eval for both, and the board's timing."""
    rows = []
    for clip in clips:
        row = mix_df[mix_df["Mixed Sound ID"] == clip["clip_id"]]
        if row.empty:
            sys.exit(f"{clip['clip_id']} is not in Mix.csv -- was hls_clips.h generated from this dataset?")
        row = row.iloc[0]
        sr = clip["sample_rate"]

        # Exactly what the board filtered: the same int16 samples, scaled the same way.
        mixed = sf.read(row["mixed_audio_path"], dtype="int16")[0][: clip["n_samples"]] / 32768.0
        heart_ref = sf.read(row["heart_audio_path"], dtype="int16")[0][: clip["n_samples"]] / 32768.0
        lung_ref = sf.read(row["lung_audio_path"], dtype="int16")[0][: clip["n_samples"]] / 32768.0

        heart_sos = design_bandpass(*HEART_BAND, sr)
        lung_sos = design_bandpass(*LUNG_BAND, sr)
        py_heart = sosfilt(heart_sos, mixed)
        py_lung = sosfilt(lung_sos, mixed)

        heart_err = float(np.max(np.abs(clip["heart"] - py_heart)))
        lung_err = float(np.max(np.abs(clip["lung"] - py_lung)))
        heart_floor = float32_floor(heart_sos, mixed, py_heart)
        lung_floor = float32_floor(lung_sos, mixed, py_lung)

        board_m = evaluate_heart_lung(heart_ref, lung_ref, clip["heart"], clip["lung"])
        py_m = evaluate_heart_lung(heart_ref, lung_ref, py_heart, py_lung)

        audio_seconds = clip["n_samples"] / sr
        rows.append({
            "clip": clip["clip_id"],
            "heart type": row["Heart Sound Type"],
            "lung type": row["Lung Sound Type"],
            "max |board - scipy| heart": heart_err,
            "max |board - scipy| lung": lung_err,
            "err / f32 floor heart": heart_err / heart_floor,
            "err / f32 floor lung": lung_err / lung_floor,
            "board heart SDR": board_m["heart"]["sdr"],
            "scipy heart SDR": py_m["heart"]["sdr"],
            "board lung SDR": board_m["lung"]["sdr"],
            "scipy lung SDR": py_m["lung"]["sdr"],
            "compute ms": clip["compute_micros"] / 1000.0,
            "us/sample": clip["compute_micros"] / clip["n_samples"],
            "real-time factor": audio_seconds * 1e6 / clip["compute_micros"],
        })
    return pd.DataFrame(rows).set_index("clip")


def build_report(df, port_name, out_path=None):
    worst = max(df["max |board - scipy| heart"].max(), df["max |board - scipy| lung"].max())
    worst_ratio = max(df["err / f32 floor heart"].max(), df["err / f32 floor lung"].max())
    ok = worst_ratio <= FLOOR_MARGIN
    rtf = df["real-time factor"].min()

    tiles = "\n".join([
        stat_tile("Board vs scipy", f"{worst_ratio:.2f}x",
                  f"of the float32 floor (max abs {worst:.1e}), limit {FLOOR_MARGIN:.0f}x", ok=ok),
        stat_tile("Slowest real-time factor", f"{rtf:.0f}x", "compute only, no I/O", ok=rtf > 1.0),
        stat_tile("Cost per sample", f"{df['us/sample'].max():.2f} us", "worst clip"),
        stat_tile("Clips", f"{len(df)}", "replayed from flash"),
    ])

    accuracy = df[["max |board - scipy| heart", "max |board - scipy| lung",
                   "err / f32 floor heart", "err / f32 floor lung",
                   "board heart SDR", "scipy heart SDR", "board lung SDR", "scipy lung SDR"]]
    timing = df[["compute ms", "us/sample", "real-time factor"]]

    body = "\n".join([
        section("Board output vs scipy, and BSS Eval against the dataset's own references",
                "firmware/tools/run_on_device.py",
                df_to_html(accuracy, index_label="clip", float_fmt="{:.4g}")
                + "<p>The board filtered the same int16 samples the Python baseline filters, so these "
                  "two SDR columns should agree to the last printed digit. They are not independent "
                  "measurements of the same thing &mdash; they are a check that the port is faithful. "
                  "The deviation columns are the real test, read as a multiple of the float32 "
                  "quantisation floor rather than in absolute terms: the floor is measured per clip "
                  "by running the same filter through scipy in float32, because it scales with "
                  "amplitude and clip length. A ratio near 1 means the board is as accurate as "
                  f"scipy-in-float32; above {FLOOR_MARGIN:.0f}x means the vendor DSP kernel is not "
                  "computing what scipy computes.</p>"),
        section("On-device timing", "measured on the board, filtering only",
                df_to_html(timing, index_label="clip", float_fmt="{:.3f}")
                + "<p>Timed around the int16 scaling and both band cascades, with serial I/O excluded. "
                  "These are the figures that replace <code>src/latency.py</code>'s desktop numbers for "
                  "this method; the desktop ones measure a different machine doing a different amount "
                  "of work.</p>"),
    ])

    out = Path(out_path) if out_path else results_dir() / "firmware_on_device_report.html"
    write_report(out, report_shell(
        title="On-device band split",
        eyebrow="Firmware",
        heading="HLS-CMDS mixtures replayed on the board",
        dek="No microphone: real dataset clips in flash, filtered on the MCU, streamed back and "
            "scored against scipy and against the dataset's isolated references.",
        stat_tiles=tiles,
        body=body,
        footer=f"<p>Generated {date.today().isoformat()} from {port_name} "
               f"by firmware/tools/run_on_device.py</p>",
    ))
    return out, ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="serial device, e.g. /dev/ttyACM0")
    source.add_argument("--from-file", dest="from_file",
                        help="score a capture written by tools/capture_raw.py, no board needed")
    p.add_argument("--baud", type=int, default=115200, help="ignored on native-USB boards")
    p.add_argument("--timeout", type=float, default=30.0, help="per-read serial timeout, seconds")
    p.add_argument("--out", default=None,
                   help="HTML report path (default: results/firmware_on_device_report.html)")
    args = p.parse_args()

    if args.from_file:
        source_name = args.from_file
        print(f"reading {source_name} ...")
        clips = receive(FileSource(args.from_file))
    else:
        source_name = args.port
        try:
            import serial  # noqa: PLC0415  (optional dependency, only the serial tools need it)
        except ImportError:
            sys.exit("pyserial is required: pip install -r requirements.txt")

        with serial.Serial(args.port, args.baud, timeout=args.timeout) as port:
            print(f"waiting for the bench firmware on {args.port} ...")
            port.reset_input_buffer()  # drop whatever the board printed before we connected
            wait_for_banner(port)
            port.write(b"s")
            port.flush()
            clips = receive(port)

    print(f"\nreceived {len(clips)} clip(s); scoring against scipy and the dataset references ...")
    df = score(clips, load_mix())
    print()
    print(df.to_string(float_format=lambda v: f"{v:.4g}"))

    out, ok = build_report(df, source_name, args.out)
    print(f"\nwrote {out}")
    if not ok:
        print(f"FAIL: board output deviates from scipy by more than {FLOOR_MARGIN:.0f}x the float32 "
              "quantisation floor -- the vendor DSP kernel is not computing what scipy computes")
        return 1
    print("PASS: board output matches scipy within the float32 quantisation floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
