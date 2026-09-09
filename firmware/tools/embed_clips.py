"""
Embed dataset clips into the firmware as flash-resident int16 arrays.

The board has no microphone in this workflow: it replays real HLS-CMDS
mixtures, so the samples it filters are bit-identical to the ones the Python
baseline filters. That makes the on-device output directly comparable to
scipy -- no transducer, no resampling, no gain to calibrate away.

What gets embedded is the *mixture* (M####.wav), the separator's input. The
isolated heart/lung references stay on the host, where tools/run_on_device.py
uses them to score the board's output with the same BSS Eval metrics as the
rest of the project.

ADDITIVE-ONLY BY DEFAULT. Only 36 of Mix.csv's 145 rows have a mixture that is
actually a*(heart+lung) of its named sources (load_dataset.verify_additive_
triplets; see the dataset audit in report/). On the other 109 the listed
heart/lung files do not describe the mixed signal at all, so an SDR scored
against them measures nothing about separation quality. The board-vs-scipy
comparison is unaffected either way -- it is a numerical diff, not a
separation metric -- but the earlier default of `mix_df.head(count)` embedded
M0001-M0003, all three of them non-additive, which made run_on_device.py's SDR
column unquotable. Selection is now drawn from the additive subset so every
number the bench prints means something. `--any` restores the old behaviour
for a deliberate board-vs-scipy-only run.

The valid-ID list is computed from verify_additive_triplets() rather than
hardcoded, so it follows the dataset instead of drifting from it. That costs a
pass over all 145 mixtures (~30 s) each time this runs.

Flash cost is 2 bytes per sample: a 15 s clip at 4000 Hz is 120 kB, so the
default of 3 clips uses 360 kB. The Nano 33 BLE Sense has ~890 kB free after
the firmware, which is the binding constraint -- the ESP32-S3 has room for far
more.

Usage:
    python firmware/tools/embed_clips.py                     # first 3 additive rows
    python firmware/tools/embed_clips.py --count 5
    python firmware/tools/embed_clips.py --ids M0111 M0112
    python firmware/tools/embed_clips.py --any --count 3     # ignore additivity
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import soundfile as sf

FIRMWARE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = FIRMWARE_DIR.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from load_dataset import load_mix, verify_additive_triplets  # noqa: E402  (needs sys.path above)

OUT_HEADER = FIRMWARE_DIR / "lib" / "hls_filter" / "hls_clips.h"
BASELINE_SR = 4000
VALUES_PER_LINE = 16

# Nano 33 BLE Sense is the tighter target: 983 040 B of flash, ~85 kB of which
# is the firmware itself. Warn before a build that will not fit.
NANO33_FLASH_BYTES = 983_040
NANO33_FIRMWARE_BYTES = 92_000  # the -bench build, rounded up


def additive_ids(mix_df):
    """Mixed Sound IDs whose mixture really is a*(heart+lung) of its named sources."""
    result = verify_additive_triplets(mix_df)
    return {row["mixed_id"] for row in result["rows"] if row["additive"]}


def select_rows(mix_df, count, ids, allow_non_additive):
    valid = None if allow_non_additive else additive_ids(mix_df)

    if ids:
        missing = [i for i in ids if i not in set(mix_df["Mixed Sound ID"])]
        if missing:
            sys.exit(f"unknown Mixed Sound ID(s): {', '.join(missing)}")
        if valid is not None:
            non_additive = [i for i in ids if i not in valid]
            if non_additive:
                sys.exit(f"not additive, so any SDR scored on them is meaningless: "
                         f"{', '.join(non_additive)}. Pass --any if that is intended.")
        return mix_df[mix_df["Mixed Sound ID"].isin(ids)].set_index("Mixed Sound ID").loc[ids].reset_index()

    if valid is None:
        return mix_df.head(count)

    rows = mix_df[mix_df["Mixed Sound ID"].isin(valid)]
    if len(rows) < count:
        sys.exit(f"only {len(rows)} additive row(s) available, asked for {count}")
    return rows.head(count)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=3, help="how many Mix.csv rows to embed (default: 3)")
    p.add_argument("--ids", nargs="+", default=None, help="explicit Mixed Sound IDs, overrides --count")
    p.add_argument("--any", dest="allow_non_additive", action="store_true",
                   help="allow non-additive rows (board-vs-scipy only; their SDR means nothing)")
    args = p.parse_args()

    rows = select_rows(load_mix(), args.count, args.ids, args.allow_non_additive)

    clips = []
    for _, row in rows.iterrows():
        samples, sr = sf.read(row["mixed_audio_path"], dtype="int16", always_2d=False)
        if sr != BASELINE_SR:
            sys.exit(f"{row['Mixed Sound ID']} is {sr} Hz, expected {BASELINE_SR}")
        clips.append((row["Mixed Sound ID"], samples, row))

    total_bytes = sum(len(s) * 2 for _, s, _ in clips)
    free = NANO33_FLASH_BYTES - NANO33_FIRMWARE_BYTES
    if total_bytes > free:
        sys.exit(f"{total_bytes} B of clip data will not fit the Nano 33's ~{free} B of free flash; "
                 f"use fewer clips")

    parts = [f"""/*
 * GENERATED by firmware/tools/embed_clips.py -- do not edit by hand.
 *
 * Generated : {date.today().isoformat()}
 * Clips     : {len(clips)} HLS-CMDS mixture(s), {total_bytes} B of flash
 * Additive  : {"yes -- SDR scored on these is meaningful"
                if not args.allow_non_additive else
                "NO (--any) -- board-vs-scipy is still valid, SDR is NOT"}
 *
 * Raw int16 PCM exactly as stored in the .wav files. The firmware scales by
 * 1/32768 to reach the [-1, 1) range, which reproduces librosa.load()'s output
 * bit for bit -- verified, so the board and the Python baseline filter the same
 * numbers and their outputs are directly comparable.
 */
#ifndef HLS_CLIPS_H
#define HLS_CLIPS_H

#include <stdint.h>

#define HLS_CLIP_COUNT {len(clips)}u
#define HLS_CLIP_SR_HZ {BASELINE_SR}u
#define HLS_CLIP_ID_LEN 12u

typedef struct {{
    char id[HLS_CLIP_ID_LEN];
    const int16_t *samples;
    uint32_t n_samples;
}} hls_clip_t;
"""]

    for i, (clip_id, samples, row) in enumerate(clips):
        parts.append(f"\n/* {clip_id}: heart={row['Heart Sound Type']}, lung={row['Lung Sound Type']}, "
                     f"location={row['Location']}, {len(samples)} samples "
                     f"({len(samples) / BASELINE_SR:g} s) */")
        parts.append(f"static const int16_t HLS_CLIP_{i}_SAMPLES[{len(samples)}] = {{")
        for j in range(0, len(samples), VALUES_PER_LINE):
            parts.append("    " + ",".join(f"{v:d}" for v in samples[j:j + VALUES_PER_LINE]) + ",")
        parts.append("};")

    parts.append(f"\nstatic const hls_clip_t HLS_CLIPS[HLS_CLIP_COUNT] = {{")
    for i, (clip_id, samples, _) in enumerate(clips):
        parts.append(f'    {{"{clip_id}", HLS_CLIP_{i}_SAMPLES, {len(samples)}u}},')
    parts.append("};\n\n#endif /* HLS_CLIPS_H */")

    OUT_HEADER.write_text("\n".join(parts) + "\n")

    print(f"wrote {OUT_HEADER.relative_to(REPO_DIR)}")
    for clip_id, samples, row in clips:
        print(f"  {clip_id}  {len(samples):6d} samples  {len(samples) * 2 / 1024:6.1f} kB  "
              f"{row['Heart Sound Type']} / {row['Lung Sound Type']}")
    print(f"  total {total_bytes / 1024:.1f} kB of flash "
          f"({100 * total_bytes / free:.0f} % of the Nano 33's free flash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
