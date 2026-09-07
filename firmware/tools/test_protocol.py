"""
Exercise the board<->host protocol end to end without a board.

A framing or struct-layout mistake is invisible until hardware is plugged in,
and then it looks like a hardware problem. This emulates the bench firmware's
byte stream over a pty, runs the real tools/run_on_device.py against it, and
checks both that a faithful stream is accepted and that a corrupted one is
rejected -- so the first real board session is spent on the board, not on the
parser.

What this does NOT cover: the C side actually writing those bytes. That is
pinned separately by the HLS_STATIC_ASSERT size checks in include/hls_stream.h,
which this script mirrors below.

Usage:
    python firmware/tools/test_protocol.py      # -> exit 0 if both cases behave
"""
import os
import pty
import re
import struct
import select
import subprocess
import sys
import termios
import tty
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402
from gen_filter_coeffs import design_bandpass  # noqa: E402
from load_dataset import load_mix  # noqa: E402

CLIPS_HEADER = REPO_DIR / "firmware" / "lib" / "hls_filter" / "hls_clips.h"

CLIP_HEADER = struct.Struct("<4s12sII")
FRAME_HEADER = struct.Struct("<4sHHI")
CLIP_FOOTER = struct.Struct("<4sII")
DONE = struct.Struct("<4sI")

# Must match the HLS_STATIC_ASSERT sizes in include/hls_stream.h.
EXPECTED_SIZES = {"frame": 12, "clip": 24, "footer": 12, "done": 8}
BLOCK = 256


def check_struct_sizes():
    actual = {"frame": FRAME_HEADER.size, "clip": CLIP_HEADER.size,
              "footer": CLIP_FOOTER.size, "done": DONE.size}
    if actual != EXPECTED_SIZES:
        print(f"FAIL: host struct sizes {actual} != the C layout {EXPECTED_SIZES}")
        return False
    print(f"ok    host struct sizes match include/hls_stream.h: {actual}")
    return True


def embedded_clip_ids():
    text = CLIPS_HEADER.read_text()
    ids = re.findall(r'\{"([^"]+)",\s*HLS_CLIP_\d+_SAMPLES', text)
    if not ids:
        sys.exit(f"no clips found in {CLIPS_HEADER} -- run `make firmware-clips` first")
    return ids


def emulate(fd, clip_ids, corrupt=False):
    """Write the bench firmware's exact byte stream: banner, then records."""
    try:
        _emulate(fd, clip_ids, corrupt)
    except OSError:
        pass  # host closed the pty first (e.g. the corrupted-stream case aborting)


def _emulate(fd, clip_ids, corrupt):
    mix_df = load_mix()
    os.write(fd, b"# hls_filter dataset-replay bench\n")
    os.write(fd, b"# backend: emulated\n")

    # The firmware repeats its prompt once a second until answered; mirror that,
    # otherwise this test would pass against a firmware that prompts only once.
    prompt = b"# send 's' to stream filtered output, any other byte for timing only\n"
    while True:
        os.write(fd, prompt)
        if select.select([fd], [], [], 0.25)[0] and os.read(fd, 1) == b"s":
            break

    for clip_id in clip_ids:
        row = mix_df[mix_df["Mixed Sound ID"] == clip_id].iloc[0]
        mixed = sf.read(row["mixed_audio_path"], dtype="int16")[0] / 32768.0
        n = len(mixed)
        heart = sosfilt(design_bandpass(*HEART_BAND, 4000), mixed).astype("<f4")
        lung = sosfilt(design_bandpass(*LUNG_BAND, 4000), mixed).astype("<f4")
        if corrupt:
            heart = heart * 1.05  # a plausible-looking gain error, well above tolerance

        os.write(fd, CLIP_HEADER.pack(b"HLSC", clip_id.encode(), n, 4000))
        for seq, off in enumerate(range(0, n, BLOCK)):
            chunk = min(BLOCK, n - off)
            os.write(fd, FRAME_HEADER.pack(b"HLS1", chunk, seq & 0xFFFF, 0))
            os.write(fd, heart[off:off + chunk].tobytes())
            os.write(fd, lung[off:off + chunk].tobytes())
        # Deliberately not a plausible timing: nothing here measures anything, and
        # emulator output must never be mistaken for a board measurement.
        os.write(fd, CLIP_FOOTER.pack(b"HLSE", n, 111_111))

    os.write(fd, DONE.pack(b"HLSD", len(clip_ids)))
    os.write(fd, b"\n# done\n")


def run_case(clip_ids, corrupt, expect_rc):
    master, slave = pty.openpty()
    # A pty defaults to canonical mode with echo and NL translation, which both
    # mangles binary payloads and holds the host's start byte until a newline.
    # Raw mode makes it behave like the USB CDC link it is standing in for.
    for fd in (master, slave):
        tty.setraw(fd, termios.TCSANOW)
    worker = threading.Thread(target=emulate, args=(master, clip_ids, corrupt), daemon=True)
    worker.start()

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "run_on_device.py"),
             "--port", os.ttyname(slave), "--timeout", "30",
             "--out", str(Path(tmp) / "report.html")],
            capture_output=True, text=True, timeout=600)
    worker.join(timeout=5)
    os.close(master)
    os.close(slave)

    label = "corrupted stream rejected" if corrupt else "faithful stream accepted"
    if proc.returncode != expect_rc:
        print(f"FAIL: {label} -- exit {proc.returncode}, expected {expect_rc}")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return False
    print(f"ok    {label} (exit {proc.returncode})")
    return True


def run_offline_case(clip_ids):
    """
    The handover path: capture_raw.py on the machine with the board, then
    run_on_device.py --from-file wherever the dataset lives. Worth testing
    end to end because it is the path that will actually be used, and its two
    halves run on different machines.
    """
    master, slave = pty.openpty()
    for fd in (master, slave):
        tty.setraw(fd, termios.TCSANOW)
    worker = threading.Thread(target=emulate, args=(master, clip_ids, False), daemon=True)
    worker.start()

    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / "capture.bin"
        grab = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "capture_raw.py"),
             "--port", os.ttyname(slave), "--out", str(capture), "--idle", "2"],
            capture_output=True, text=True, timeout=600)
        worker.join(timeout=5)
        os.close(master)
        os.close(slave)

        if grab.returncode != 0 or not capture.exists():
            print(f"FAIL: capture_raw.py exited {grab.returncode}")
            print(grab.stdout[-1000:], grab.stderr[-1000:])
            return False

        score = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "run_on_device.py"),
             "--from-file", str(capture), "--out", str(Path(tmp) / "report.html")],
            capture_output=True, text=True, timeout=600)
        if score.returncode != 0:
            print(f"FAIL: offline scoring exited {score.returncode}")
            print(score.stdout[-2000:], score.stderr[-2000:])
            return False

    print(f"ok    capture_raw.py -> run_on_device.py --from-file "
          f"({capture.name}, {len(clip_ids)} clips)")
    return True


def main():
    clip_ids = embedded_clip_ids()
    print(f"protocol check against {len(clip_ids)} embedded clip(s): {', '.join(clip_ids)}")
    results = [
        check_struct_sizes(),
        run_case(clip_ids, corrupt=False, expect_rc=0),
        run_case(clip_ids, corrupt=True, expect_rc=1),
        run_offline_case(clip_ids),
    ]
    ok = all(results)
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
