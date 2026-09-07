"""
Capture the binary frames the board firmware streams and write them as WAVs.

The band split comes off the MCU as 4000 Hz float32 heart/lung pairs
(~32 kB/s), framed by include/hls_stream.h -- too fast for any text format, so
this decodes the frames on the host. Output lands in results/ next to
everything else the project generates.

The frame header carries a sequence number and a cumulative dropped-sample
count; both are checked here and reported, because a capture with gaps is not
comparable to the Python baseline's numbers and should not quietly be used.

Usage:
    python firmware/tools/capture_serial.py --port /dev/ttyACM0 --seconds 15
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_DIR = Path(__file__).resolve().parents[2]
BASELINE_SR = 4000
MAGIC = b"HLS1"
HEADER = struct.Struct("<4sHHI")  # magic, n_samples, seq, dropped


def resync(port):
    """
    Skip the firmware's '#' banner lines and land on the first frame header.
    Scans byte by byte for MAGIC rather than assuming the banner ends cleanly.
    """
    window = b""
    while True:
        byte = port.read(1)
        if not byte:
            raise TimeoutError("no data from the board -- is it running the streaming firmware?")
        window = (window + byte)[-4:]
        if window == MAGIC:
            return


def read_exact(port, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = port.read(n - len(buf))
        if not chunk:
            raise TimeoutError(f"serial timeout after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


def capture(port, seconds):
    heart_chunks, lung_chunks = [], []
    gaps = 0
    dropped = 0
    expected_seq = None
    deadline = time.monotonic() + seconds

    resync(port)
    first = True
    while time.monotonic() < deadline:
        if first:
            # resync() already consumed the magic of this frame
            rest = read_exact(port, HEADER.size - len(MAGIC))
            magic, n, seq, dropped = HEADER.unpack(MAGIC + rest)
            first = False
        else:
            magic, n, seq, dropped = HEADER.unpack(read_exact(port, HEADER.size))
            if magic != MAGIC:
                resync(port)
                gaps += 1
                first = True
                continue

        payload = read_exact(port, n * 2 * 4)
        both = np.frombuffer(payload, dtype="<f4")
        heart_chunks.append(both[:n])
        lung_chunks.append(both[n:])

        if expected_seq is not None and seq != expected_seq:
            gaps += 1
        expected_seq = (seq + 1) & 0xFFFF

    return (np.concatenate(heart_chunks) if heart_chunks else np.zeros(0, "f4"),
            np.concatenate(lung_chunks) if lung_chunks else np.zeros(0, "f4"),
            gaps, dropped)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200, help="ignored on native-USB boards")
    p.add_argument("--seconds", type=float, default=15.0, help="capture length (default: one clip's worth)")
    p.add_argument("--out", default=str(REPO_DIR / "results" / "firmware_capture"),
                   help="output prefix; _heart.wav and _lung.wav are appended")
    args = p.parse_args()

    try:
        import serial  # noqa: PLC0415  (optional dependency, only this tool needs it)
    except ImportError:
        sys.exit("pyserial is required: pip install -r requirements.txt")

    with serial.Serial(args.port, args.baud, timeout=2.0) as port:
        print(f"capturing {args.seconds:g} s from {args.port} ...")
        heart, lung, gaps, dropped = capture(port, args.seconds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(f"{out}_heart.wav", heart, BASELINE_SR, subtype="PCM_16")
    sf.write(f"{out}_lung.wav", lung, BASELINE_SR, subtype="PCM_16")

    print(f"heart: {len(heart)} samples ({len(heart) / BASELINE_SR:.2f} s)  peak {np.abs(heart).max():.4f}")
    print(f"lung : {len(lung)} samples ({len(lung) / BASELINE_SR:.2f} s)  peak {np.abs(lung).max():.4f}")
    print(f"wrote {out}_heart.wav and {out}_lung.wav")
    if gaps or dropped:
        print(f"WARNING: {gaps} frame gap(s), {dropped} mic samples dropped on the board -- "
              "this capture is not gap-free and is not comparable to the Python baseline")
        return 1
    print("no frame gaps, no dropped mic samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
