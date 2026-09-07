"""
Dump the bench firmware's raw serial stream to a file. pyserial only.

This is the one tool meant to run on the machine that has the board. It does no
analysis, so it needs no numpy, scipy, pandas, mir_eval or dataset -- copy the
file it writes back to wherever the dataset lives and score it there with:

    python firmware/tools/run_on_device.py --from-file capture.bin

That split exists because scoring needs the HLS-CMDS references to compare
against, and dragging the dataset and a full scientific Python stack onto a
laptop just to hold a USB cable is the wrong trade.

Stopping condition is silence: the firmware prints its text summary after the
last binary record and then goes quiet, so the capture ends once nothing has
arrived for --idle seconds. run_on_device.py validates the structure afterwards
and fails loudly on a truncated file, so a short capture cannot pass unnoticed.

Usage:
    python firmware/tools/capture_raw.py --port /dev/ttyACM0 --out capture.bin
"""
import argparse
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyACM0 or COM3")
    p.add_argument("--out", default="capture.bin", help="output file (default: capture.bin)")
    p.add_argument("--baud", type=int, default=115200, help="ignored on native-USB boards")
    p.add_argument("--idle", type=float, default=5.0, help="stop after this many seconds of silence")
    p.add_argument("--max-seconds", type=float, default=600.0, help="hard cap on capture length")
    args = p.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial is required on this machine: pip install pyserial")

    out = Path(args.out)
    started = time.monotonic()
    last_data = started
    total = 0
    prompted = False

    with serial.Serial(args.port, args.baud, timeout=0.5) as port, out.open("wb") as fp:
        port.reset_input_buffer()  # drop whatever the board printed before we connected
        print(f"capturing from {args.port} -> {out} (stops after {args.idle:g}s of silence)")

        while True:
            chunk = port.read(4096)
            now = time.monotonic()

            if chunk:
                fp.write(chunk)
                total += len(chunk)
                last_data = now
                print(f"\r  {total / 1024:.0f} kB", end="", flush=True)
                # The firmware repeats its prompt until answered; answer it once.
                if not prompted and b"send 's'" in chunk:
                    port.write(b"s")
                    port.flush()
                    prompted = True
            elif prompted and now - last_data >= args.idle:
                break
            elif not prompted and now - started >= args.idle * 2:
                sys.exit(f"\nno prompt from the board after {args.idle * 2:g}s -- "
                         "is a *-bench firmware flashed?")

            if now - started >= args.max_seconds:
                print(f"\nstopped at the {args.max_seconds:g}s cap")
                break

    print(f"\nwrote {out} ({total / 1024:.0f} kB)")
    print("score it where the dataset lives:")
    print(f"    python firmware/tools/run_on_device.py --from-file {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
