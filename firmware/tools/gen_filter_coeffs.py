"""
Design the firmware's IIR filters with scipy on the server and emit them as
hardcoded C headers -- the MCU never runs a filter-design routine.

Two stages are generated:

1. Decimation (mic rate -> baseline rate). The Nano 33 PDM mic and the
   ESP32-S3 I2S mic both run at 16 kHz; every Python number in this project
   was computed on the dataset's 4000 Hz recordings, so the firmware
   anti-alias-filters and decimates 16 kHz -> 4000 Hz before the band split.
   This stage has no counterpart in the Python baseline (the dataset is
   already at 4000 Hz), so it is verified against a synthetic signal.

2. Heart / lung bandpass, order 4 Butterworth, bands imported from
   src/baseline/common.py so the firmware cannot silently drift from
   Baseline 1.

Also written: golden vectors (raw float32) so the C implementation can be
diffed against scipy on the host before any board is involved. The reference
is scipy.signal.sosfilt -- causal, one-directional -- NOT the sosfiltfilt the
Python baseline uses. That is a deliberate difference, see firmware/README.md.

Usage:
    python firmware/tools/gen_filter_coeffs.py            # regenerate everything
    python firmware/tools/gen_filter_coeffs.py --check    # fail if outputs are stale
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import butter, sosfilt

FIRMWARE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = FIRMWARE_DIR.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from baseline.common import HEART_BAND, LUNG_BAND  # noqa: E402  (needs sys.path above)

BASELINE_SR = 4000          # dataset / Python-baseline sample rate
MIC_SR = 16000              # PDM (Nano 33) and I2S (ESP32-S3) capture rate
DECIM_FACTOR = MIC_SR // BASELINE_SR
DECIM_CUTOFF_HZ = 1400.0    # anti-alias, ~53 dB down at 3 kHz (the worst fold-back)
DECIM_ORDER = 8
BAND_ORDER = 4              # matches baseline/common.py::_bandpass

GOLDEN_SOURCE_WAV = REPO_DIR / "HLS_CMDS" / "Mix" / "H0001.wav"
GOLDEN_SAMPLES = 8000       # 2 s at 4000 Hz -- covers filter transient + steady state
EMBEDDED_SAMPLES = 1024     # subset compiled into the board self-test firmware (~12 kB of flash)
GOLDEN_DIR = FIRMWARE_DIR / "golden"
COEFF_HEADER = FIRMWARE_DIR / "lib" / "hls_filter" / "hls_filter_coeffs.h"
EMBEDDED_HEADER = FIRMWARE_DIR / "lib" / "hls_filter" / "hls_golden_embedded.h"


def design_bandpass(low_hz, high_hz, sr, order=BAND_ORDER):
    """Same design call as baseline/common.py::_bandpass -- only the *application* differs."""
    nyquist = sr / 2
    low_norm = max(low_hz / nyquist, 1e-6)
    high_norm = min(high_hz / nyquist, 1 - 1e-6)
    return butter(order, [low_norm, high_norm], btype="bandpass", output="sos")


def design_decimator(sr, cutoff_hz, order=DECIM_ORDER):
    return butter(order, cutoff_hz / (sr / 2), btype="lowpass", output="sos")


def _fmt_sos(sos):
    """scipy's 6-column SOS -> the 5-column {b0,b1,b2,a1,a2} layout the C core expects (a0 == 1)."""
    rows = []
    for b0, b1, b2, a0, a1, a2 in sos:
        assert abs(a0 - 1.0) < 1e-12, f"unnormalised section (a0={a0})"
        rows.append((b0, b1, b2, a1, a2))
    return rows


def _c_vector(name, values):
    """A plain float array, wrapped at 6 values per line to keep the header readable."""
    lines = [f"#define {name}_LEN {len(values)}u",
             f"static const float {name}[{name}_LEN] = {{"]
    for i in range(0, len(values), 6):
        lines.append("    " + " ".join(f"{v: .9e}f," for v in values[i:i + 6]))
    lines.append("};")
    return "\n".join(lines)


def _c_array(name, sos, comment):
    rows = _fmt_sos(sos)
    lines = [f"/* {comment} */", f"#define {name}_SECTIONS {len(rows)}u",
             f"static const float {name}_SOS[{name}_SECTIONS * 5] = {{"]
    for b0, b1, b2, a1, a2 in rows:
        vals = ", ".join(f"{v: .9e}f" for v in (b0, b1, b2, a1, a2))
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)


def quantisation_floor(sos, x):
    """
    How far scipy itself moves when the coefficients and data are rounded to
    float32 -- the error the C float32 core cannot go below, and the number
    the native test's tolerance is set from.
    """
    ref = sosfilt(sos, x.astype(np.float64))
    f32 = sosfilt(sos.astype(np.float32), x.astype(np.float32))
    return float(np.max(np.abs(ref - f32)))


def synthetic_mic_signal(n, sr, seed=20260907):
    """
    Deterministic 16 kHz stand-in for a mic frame: in-band tones, an
    out-of-band tone that must be rejected before decimation (3.1 kHz folds
    to 900 Hz, right inside the lung band), and low-level noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = (0.30 * np.sin(2 * np.pi * 60.0 * t)      # heart band
         + 0.20 * np.sin(2 * np.pi * 450.0 * t)   # lung band
         + 0.15 * np.sin(2 * np.pi * 3100.0 * t)  # aliases onto 900 Hz if not filtered
         + 0.02 * rng.standard_normal(n))
    return x.astype(np.float64)


def write_golden(name, arr):
    path = GOLDEN_DIR / f"{name}.f32"
    path.write_bytes(np.asarray(arr, dtype="<f4").tobytes())
    return path


def build(check_only=False):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    heart_sos = design_bandpass(*HEART_BAND, BASELINE_SR)
    lung_sos = design_bandpass(*LUNG_BAND, BASELINE_SR)
    decim_sos = design_decimator(MIC_SR, DECIM_CUTOFF_HZ)

    audio, sr = sf.read(GOLDEN_SOURCE_WAV, dtype="float32", always_2d=False)
    assert sr == BASELINE_SR, f"{GOLDEN_SOURCE_WAV} is {sr} Hz, expected {BASELINE_SR}"
    x = audio[:GOLDEN_SAMPLES].astype(np.float64)

    mic = synthetic_mic_signal(GOLDEN_SAMPLES * DECIM_FACTOR, MIC_SR)

    heart = sosfilt(heart_sos, x)
    lung = sosfilt(lung_sos, x)
    decim = sosfilt(decim_sos, mic)[::DECIM_FACTOR]

    floors = {
        "heart": quantisation_floor(heart_sos, x),
        "lung": quantisation_floor(lung_sos, x),
        "decim": quantisation_floor(decim_sos, mic),
    }

    header = f"""/*
 * GENERATED by firmware/tools/gen_filter_coeffs.py -- do not edit by hand.
 *
 * Generated : {date.today().isoformat()}
 * scipy     : {scipy.__version__}
 * Bands     : heart {HEART_BAND[0]:g}-{HEART_BAND[1]:g} Hz, lung {LUNG_BAND[0]:g}-{LUNG_BAND[1]:g} Hz
 *             (imported from src/baseline/common.py -- change them there, then rerun this tool)
 * Design    : Butterworth order {BAND_ORDER} bandpass @ {BASELINE_SR} Hz,
 *             Butterworth order {DECIM_ORDER} lowpass @ {DECIM_CUTOFF_HZ:g} Hz for {MIC_SR} -> {BASELINE_SR} Hz decimation
 *
 * Layout is scipy's second-order sections with a0 divided out:
 *   {{b0, b1, b2, a1, a2}} per section, y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 * which is exactly what ESP-DSP's dsps_biquad_f32 expects. The CMSIS-DSP
 * backend negates a1/a2 at init (see hls_filter.c) because CMSIS uses the
 * opposite denominator sign.
 */
#ifndef HLS_FILTER_COEFFS_H
#define HLS_FILTER_COEFFS_H

#define HLS_BASELINE_SR_HZ {BASELINE_SR}u
#define HLS_MIC_SR_HZ      {MIC_SR}u
#define HLS_DECIM_FACTOR   {DECIM_FACTOR}u

#define HLS_HEART_LOW_HZ   {HEART_BAND[0]:.1f}f
#define HLS_HEART_HIGH_HZ  {HEART_BAND[1]:.1f}f
#define HLS_LUNG_LOW_HZ    {LUNG_BAND[0]:.1f}f
#define HLS_LUNG_HIGH_HZ   {LUNG_BAND[1]:.1f}f

{_c_array("HLS_HEART", heart_sos, f"heart bandpass {HEART_BAND[0]:g}-{HEART_BAND[1]:g} Hz @ {BASELINE_SR} Hz")}

{_c_array("HLS_LUNG", lung_sos, f"lung bandpass {LUNG_BAND[0]:g}-{LUNG_BAND[1]:g} Hz @ {BASELINE_SR} Hz")}

{_c_array("HLS_DECIM", decim_sos, f"anti-alias lowpass {DECIM_CUTOFF_HZ:g} Hz @ {MIC_SR} Hz, decimate by {DECIM_FACTOR}")}

#endif /* HLS_FILTER_COEFFS_H */
"""

    if check_only:
        current = COEFF_HEADER.read_text() if COEFF_HEADER.exists() else ""
        # the Generated: line changes every day; compare everything else
        strip = lambda s: "\n".join(l for l in s.splitlines() if not l.startswith(" * Generated :"))
        if strip(current) != strip(header):
            print("STALE: hls_filter_coeffs.h does not match the current design", file=sys.stderr)
            return 1
        print("hls_filter_coeffs.h is up to date")
        return 0

    COEFF_HEADER.write_text(header)
    EMBEDDED_HEADER.write_text(f"""/*
 * GENERATED by firmware/tools/gen_filter_coeffs.py -- do not edit by hand.
 *
 * A {EMBEDDED_SAMPLES}-sample slice of the golden vectors, compiled into the board
 * self-test firmware (src/main_selftest.cpp, envs *-selftest). The `native`
 * environment already pins the portable C path to scipy on the host; this
 * header is what lets the CMSIS-DSP and ESP-DSP paths be checked against the
 * same scipy reference on real silicon, the first time a board is plugged in.
 *
 * Source: {GOLDEN_SOURCE_WAV.name} samples [0:{EMBEDDED_SAMPLES}], scipy {scipy.__version__}, {date.today().isoformat()}
 */
#ifndef HLS_GOLDEN_EMBEDDED_H
#define HLS_GOLDEN_EMBEDDED_H

{_c_vector("HLS_GOLDEN_INPUT", x[:EMBEDDED_SAMPLES])}

{_c_vector("HLS_GOLDEN_HEART", heart[:EMBEDDED_SAMPLES])}

{_c_vector("HLS_GOLDEN_LUNG", lung[:EMBEDDED_SAMPLES])}

#endif /* HLS_GOLDEN_EMBEDDED_H */
""")
    write_golden("bandpass_input", x)
    write_golden("bandpass_heart", heart)
    write_golden("bandpass_lung", lung)
    write_golden("decim_input", mic)
    write_golden("decim_output", decim)
    (GOLDEN_DIR / "manifest.txt").write_text(
        f"""generated      {date.today().isoformat()}
scipy          {scipy.__version__}
numpy          {np.__version__}
reference      scipy.signal.sosfilt (causal, single-pass) in float64
bandpass src   {GOLDEN_SOURCE_WAV.relative_to(REPO_DIR)} samples [0:{GOLDEN_SAMPLES}] @ {BASELINE_SR} Hz
decim src      synthetic_mic_signal(n={GOLDEN_SAMPLES * DECIM_FACTOR}, sr={MIC_SR}, seed=20260907)
float32 floor  heart {floors['heart']:.3e}  lung {floors['lung']:.3e}  decim {floors['decim']:.3e}
""")

    print(f"wrote {EMBEDDED_HEADER.relative_to(REPO_DIR)}  ({EMBEDDED_SAMPLES} samples x 3 vectors)")
    print(f"wrote {COEFF_HEADER.relative_to(REPO_DIR)}"
          f"  ({len(heart_sos)}/{len(lung_sos)}/{len(decim_sos)} sections heart/lung/decim)")
    print(f"wrote 5 golden vectors to {GOLDEN_DIR.relative_to(REPO_DIR)}/")
    print("float32 quantisation floor (scipy f64 vs scipy f32) -- the C core cannot beat this:")
    for k, v in floors.items():
        print(f"  {k:6s} {v:.3e}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify the committed header matches the design")
    sys.exit(build(check_only=p.parse_args().check))
