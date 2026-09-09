# `firmware/` — heart/lung band split on the MCU

The embedded port of **Baseline 1** (`src/baseline/baseline1.py`): a fixed
Butterworth band split, heart 20–200 Hz and lung 150–1000 Hz, running on the
board instead of on the server.

Everything numerical lives in `lib/hls_filter/` and is shared by all targets.
Board-specific code does acquisition and transport only.

**[`pipeline.md`](pipeline.md) has the flow charts** — where each piece runs, the
runtime signal path, and the verification chain. This file has the reasoning.

## Two input paths, and why the default has no microphone

| | Input | What it is for |
|---|---|---|
| **`*-bench` (default)** | real HLS-CMDS mixtures compiled into flash | measuring the port |
| `nano33ble` / `esp32s3` | live PDM / I2S microphone | a working device later |

The bench path is the one that produces numbers. Because the board filters the
**same int16 samples** the Python baseline filters, its output can be diffed
against scipy sample by sample, and its SDR dropped straight into the tables in
`results/`. A microphone cannot do that: different transducer, different gain,
no ground truth to score against — a live capture can show the firmware runs,
never that it computes the right thing.

The mic firmwares still build and are kept for the eventual real device, but
nothing in the current measurement workflow uses them.

---

## The filtering decision: causal streaming, not `filtfilt`

The Python baseline calls `scipy.signal.sosfiltfilt` — it runs the cascade
forwards, then backwards over the reversed signal. That cancels the filter's
phase response exactly (zero phase), but it can only start once the **entire**
recording exists.

The firmware runs the equivalent of `scipy.signal.sosfilt`: one causal pass,
sample- or block-at-a-time, state carried across calls.

**Why not buffer the clip and reproduce `filtfilt` exactly.** A 15 s clip at
4000 Hz is 60 000 samples. As raw `int16` that is 120 kB — 46 % of the Nano 33
BLE Sense's 256 kB of RAM before any processing. `filtfilt` then needs a
float working copy of the same signal: 240 kB in `float32`, on top of the
capture buffer and on top of what mbed-OS already occupies. That does not fit.
Even where it would fit (ESP32-S3 with PSRAM), it forfeits the point of the
exercise — no output can be produced until the recording ends, so it is
buffered batch processing wearing a microcontroller.

**What the causal filter costs.** Two things, both real:

1. **Phase distortion.** The output carries the filter's phase response and a
   non-zero group delay. Waveform-shape comparisons against the Python
   baseline's output are not like-for-like.
2. **Half the effective order.** `sosfiltfilt` applies the same cascade twice,
   so its stopbands are twice as deep in dB. One pass of the order-4 design is
   a genuinely weaker filter than what Baseline 1 reports.

### What it actually costs — measured

`make firmware-causal-check` runs both filters over the same 145 `Mix.csv` rows
with the same coefficients and the same BSS Eval metrics as the rest of the
project; only `sosfiltfilt` → `sosfilt` changes. Mean over the 145 rows
(`results/firmware_causal_vs_zerophase.html`):

| | SDR | SIR | SAR |
|---|---|---|---|
| heart, causal − zero-phase | **+0.18 dB** | −0.13 dB | +5.47 dB |
| lung, causal − zero-phase | **−0.04 dB** | −0.50 dB | +7.30 dB |

- **SDR is unchanged within ±0.2 dB.** On this dataset the causal filter costs
  nothing on the headline metric — better than the theory suggested.
- **SIR drops slightly**, as the halved effective stopband depth predicts.
  Rejecting the other source is exactly what stopband depth buys.
- **The large SAR gain is a metric artifact, not a quality gain.**
  `mir_eval.separation._project` fits a *causal* 512-tap distortion filter
  ("delays between 0 and flen-1"). A causal IIR's phase response lies inside
  that subspace and is absorbed as allowed distortion; a zero-phase filter's
  acausal response does not fit, so part of it is scored as artifacts. Do not
  report this as the firmware separating more cleanly.

Both filters sit around −12 to −14 dB SDR. Baseline 1 is this project's
zero-training sanity floor, not a working separator — these numbers say what
the *port* costs, not that the method is good.

**This is the one intentional divergence from the Python baseline. Firmware SDR
numbers are not interchangeable with Baseline 1's.**

---

## Signal path

### Bench path — dataset replay (the measurement path)

```
HLS-CMDS mixture, int16 @ 4000 Hz, in flash   (tools/embed_clips.py)
  ▼
× 1/32768   ← reproduces librosa.load() bit for bit (verified)
  ├─► heart bandpass  20–200 Hz   (Butterworth order 4, 4 biquads)
  └─► lung  bandpass 150–1000 Hz  (Butterworth order 4, 4 biquads)
  ▼
binary frames over USB serial → tools/run_on_device.py
  ▼
diff vs scipy + BSS Eval vs the dataset's isolated H####/L#### references
  → results/firmware_on_device_report.html
```

**Only additive rows are embedded.** 109 of `Mix.csv`'s 145 mixtures are not
`a·(heart+lung)` of their named sources at all (Section *Mixture Validity* of
the main report), so an SDR scored against those references measures nothing.
`embed_clips.py` draws from the 36 valid rows and refuses a non-additive
`--ids` unless `--any` is passed; the generated `hls_clips.h` records which
mode produced it. The board-vs-scipy diff is a numerical comparison and stays
valid either way — it is the SDR column that additivity governs.

**No decimation here** — the dataset is already at 4000 Hz. Only the two
bandpass cascades run, which is exactly what Baseline 1 does.

### Mic path — live capture (built, not currently used)

```
mic @ 16 kHz (int16)
  │   PDM (Nano 33, onboard)  /  I2S (ESP32-S3, external MEMS mic)
  ▼
anti-alias lowpass — Butterworth order 8 @ 1400 Hz
  ▼
decimate ÷4
  ▼
4000 Hz
  ├─► heart bandpass  20–200 Hz
  └─► lung  bandpass 150–1000 Hz
  ▼
binary frames over USB serial → tools/capture_serial.py → WAVs in results/
```

The decimation stage exists because the mics run at 16 kHz and the dataset —
and therefore every baseline, metric and report in this repo — is at 4000 Hz.
The 1400 Hz corner puts ~53 dB of attenuation on 3 kHz content, which is what
would otherwise fold onto 1 kHz and land inside the lung band. It has no
counterpart in the Python pipeline, so it is verified against a synthetic
signal rather than against a baseline.

---

## Layout

```
firmware/
├── pipeline.md               flow charts: signal path, verification chain, environments
├── platformio.ini            7 environments (see below)
├── lib/hls_filter/           the shared, board-independent core
│   ├── hls_filter.h/.c         biquad cascade, decimator, full pipeline
│   ├── hls_filter_coeffs.h     GENERATED — hardcoded SOS coefficients
│   ├── hls_golden_embedded.h   GENERATED — 1024-sample scipy reference for the on-board self-test
│   └── hls_clips.h             GENERATED — dataset mixtures as flash int16 arrays (committed)
├── include/hls_stream.h      serial record formats, sizes pinned by static asserts
├── src/
│   ├── main_native.c           host test: C vs scipy on the golden vectors
│   ├── main_selftest.cpp       on-board test: same check, on real silicon
│   ├── main_bench.cpp          dataset replay + timing + streaming  ← the measurement firmware
│   ├── main_nano33ble.cpp      PDM mic acquisition + streaming
│   └── main_esp32s3.cpp        I2S mic acquisition + streaming
├── tools/
│   ├── gen_filter_coeffs.py    scipy filter design → C headers + golden vectors
│   ├── embed_clips.py          dataset mixtures → flash-resident int16 arrays
│   ├── cmsis_dsp.py            PlatformIO hook that puts CMSIS-DSP on the build
│   ├── capture_raw.py          board machine: dump the serial stream to a file (pyserial only)
│   ├── run_on_device.py        host: receive/score, diff vs scipy, BSS Eval, timing → results/
│   ├── test_protocol.py        host: exercise the wire protocol against an emulated board
│   ├── capture_serial.py       host: decode mic-path frames → results/*.wav
│   └── causal_vs_zerophase.py  host: measure the cost of the causal filter
└── golden/                   GENERATED (gitignored) — raw float32 scipy reference vectors
```

## Coefficients are designed on the server, never on the MCU

`tools/gen_filter_coeffs.py` runs `scipy.signal.butter` and writes the
second-order sections into `lib/hls_filter/hls_filter_coeffs.h` as hardcoded
`float` literals. The MCU has no filter-design code and does no runtime design
work. The bands are **imported from `src/baseline/common.py`**, so the firmware
cannot silently drift from Baseline 1; change them there and re-run:

```sh
make firmware-coeffs
```

The tool also emits the golden vectors used by both tests, and reports the
`float32` quantisation floor — the accuracy the `float32` MCU arithmetic cannot
beat, which is what the test tolerances are set from.

## DSP backends

One API, three implementations, selected by a compile-time macro so all targets
consume the same generated coefficients:

| Backend | Macro | Kernel |
|---|---|---|
| Nano 33 BLE Sense (Cortex-M4F) | `HLS_FILTER_BACKEND_CMSIS` | CMSIS-DSP `arm_biquad_cascade_df2T_f32` |
| ESP32-S3 (Xtensa LX7) | `HLS_FILTER_BACKEND_ESPDSP` | ESP-DSP `dsps_biquad_f32` |
| host / native test | *(none)* | portable C, transposed direct form II |

CMSIS writes the recursion as `… + a1·y[n-1]`, scipy and ESP-DSP as
`… − a1·y[n-1]`; `hls_biquad_init` negates `a1`/`a2` for the CMSIS path so the
generated header stays backend-independent.

CMSIS-DSP is not shipped with the Arduino mbed core, and the registry's
`arduino-libraries/Arduino_CMSIS-DSP` package is an empty stub (`library.properties`
and a build script, no sources). `tools/cmsis_dsp.py` pulls the official
`platformio/framework-cmsis-dsp` package instead and compiles only the one
kernel actually called. ESP-DSP needs no such handling — it ships inside the
arduino-esp32 SDK, already on the include path and already linked.

---

## Verification

### On the server, no hardware (`make firmware-test`)

Builds the `native` environment (x86) and diffs the C filter against
`scipy.signal.sosfilt` on the golden vectors:

```
[1] bandpass vs scipy.signal.sosfilt
  heart 20-200 Hz              PASS  n=8000   max|err|=5.821e-08
  lung 150-1000 Hz             PASS  n=8000   max|err|=1.630e-08
[2] decimation 16 kHz -> 4 kHz vs scipy
  lowpass + drop 3 of 4        PASS  n=8000   max|err|=6.557e-07
[3] streaming state: block size must not change the output
  block size 1 / 7 / 64 / 333 == one call     PASS  (bit-identical)
[4] full pipeline (int16 mic -> heart/lung) == decimator + bandpass
  decimated signal / heart band               PASS  (bit-identical)
```

The bandpass errors are **exactly** the `float32` quantisation floor the
generator reports (`golden/manifest.txt`: heart 5.821e-08, lung 1.640e-08) —
i.e. the C core is doing the same arithmetic scipy would do in `float32`, not
merely something close. Test [3] is what catches broken streaming state: the
same input split into blocks of 1, 7, 64 and 333 samples must give a
bit-identical result to a single call.

### On the server, the wire protocol (`make firmware-protocol-test`)

A framing or struct-layout mistake is invisible until a board is plugged in, and
then it looks like a hardware problem. This emulates the bench firmware's byte
stream over a pty and runs the real `tools/run_on_device.py` against it, checking
both that a faithful stream is accepted and that a corrupted one is rejected. The
record sizes are pinned on the C side by `HLS_STATIC_ASSERT` in
`include/hls_stream.h` and mirrored in the test, so a layout change fails the
build or the test rather than the board session.

### On the board (`pio run -e nano33ble-selftest -t upload`)

A build that links is not a build that computes the right thing, and the CMSIS
and ESP-DSP kernels have not run here — this server has no boards attached. The
`*-selftest` environments replay a 1024-sample slice of the same scipy
reference through the vendor kernels and print the deviation over serial. Run
this first on each board; expect `max|err|` at the same ~6e-8 / ~2e-8 floor.
The report repeats every 3 seconds, so a serial monitor opened late still sees
it — the ESP32-S3 does not wait for the host before printing.

### Current status

| Checked | Status |
|---|---|
| C filter vs scipy, portable backend | ✅ error equals the float32 floor |
| Streaming state, 4 block sizes | ✅ bit-identical |
| Wire protocol, both directions | ✅ good stream accepted, corrupted stream rejected |
| All 7 environments compile | ✅ |
| CMSIS-DSP numerics, **on a Nano 33 BLE** | ✅ `5.3e-8` heart / `1.5e-8` lung on the golden slice |
| **On-device SDR and timing, Nano 33 BLE** | ✅ matches scipy, 4.0x real-time — see below |
| ESP-DSP numerics + bench, ESP32-S3 | ⬜ board not yet run |

### First hardware run (Nano 33 BLE Sense)

Three additive mixtures replayed from flash, streamed back, scored on the host:

| clip | max abs err (heart / lung) | × float32 floor | board SDR = scipy SDR (heart / lung) | µs/sample |
|---|---|---|---|---|
| M0087 | 9.4e-06 / 6.3e-07 | 1.57 / 1.37 | 4.918 / 3.686 | 62.45 |
| M0111 | 5.0e-06 / 3.6e-07 | 0.54 / 0.51 | 18.91 / −12.43 | 62.42 |
| M0112 | 1.0e-05 / 6.9e-07 | 0.94 / 0.85 | 4.556 / 1.197 | 62.43 |

Every SDR agrees with scipy to the last printed digit, which is the point of
the exercise: the port is faithful, not merely close.

**Timing: 62.4 µs/sample, 4.0x faster than real time.** A 15 s clip filters in
3.75 s of compute, so a 64 MHz Cortex-M4F has 4x headroom to run this band
split as a live stream. This replaces `src/latency.py`'s desktop figure for
Baseline 1 — that one measured a different machine doing a different amount of
work.

**On reading the deviation columns**: they are scored as a multiple of the
float32 quantisation floor, measured per clip, not against a fixed absolute
number. An earlier absolute `1e-5` tolerance passed the quiet clips it was
calibrated on and then failed these full-scale ones on a kernel that had
already passed its own self-test — the floor scales with amplitude, and the
36 additive rows are peak-normalised. Running scipy itself in float32 on
M0112 deviates by `1.11e-05`, *more* than the board's `1.05e-05`.

Flash / RAM at build time (3 embedded clips = 352 kB):

| Environment | RAM | Flash |
|---|---|---|
| `nano33ble` | 55 776 B / 262 144 (21.3 %) | 84 804 B / 983 040 (8.6 %) |
| `nano33ble-bench` | 45 992 B (17.5 %) | 438 860 B (44.6 %) |
| `esp32s3` | 28 192 B / 327 680 (8.6 %) | 283 749 B / 3 342 336 (8.5 %) |
| `esp32s3-bench` | 21 960 B (6.7 %) | 625 805 B (18.7 %) |

Nothing in those RAM figures scales with recording length — the filters are
streaming, and the embedded clips live in flash, not RAM. Filter state is a few
hundred bytes whether the recording is 15 seconds or an hour. The Nano 33's
flash is what limits how many clips can be embedded: ~890 kB free, 117 kB per
15 s clip, so about 7 is the practical ceiling (`make firmware-clips CLIPS=n`).

---

## Commands

Server only, no board:

```sh
make firmware-coeffs           # redesign filters with scipy, regenerate C headers + golden vectors
make firmware-clips            # embed dataset mixtures into flash (CLIPS=3 by default)
make firmware-test             # build native (x86) + diff against scipy
make firmware-protocol-test    # exercise the wire protocol against an emulated board
make firmware-build            # compile all six board environments
make firmware                  # all of the above, in order
make firmware-causal-check     # measure causal vs zero-phase on the 145 Mix rows
```

With a board attached — **in this order**:

```sh
cd firmware
pio run -e nano33ble-selftest -t upload && pio device monitor   # 1. verify the DSP kernel
pio run -e nano33ble-bench -t upload                            # 2. flash the measurement firmware
cd .. && make firmware-on-device PORT=/dev/ttyACM0              # 3. replay, score, time
```

Step 1 first, always: if the vendor DSP kernel is wrong, every number step 3
produces is wrong in a way that looks like a filter-design problem.

Step 3 writes `results/firmware_on_device_report.html` and exits non-zero if the
board's output deviates from scipy by more than 1e-5. Without the host script,
`pio device monitor` and any key other than `s` gives a plain timing report.

### When the board is on a different machine from the dataset

Scoring needs the HLS-CMDS references; flashing does not. So the machine holding
the USB cable only needs PlatformIO and `pyserial`:

```sh
# on the machine with the board — no dataset, no scientific Python stack
pio run -e nano33ble-bench -t upload
python firmware/tools/capture_raw.py --port /dev/ttyACM0 --out capture.bin

# copy capture.bin back, then wherever the dataset lives:
python firmware/tools/run_on_device.py --from-file capture.bin
```

`capture_raw.py` imports nothing but `pyserial` and the standard library. A
truncated capture is rejected by the scoring step rather than silently scored.
`hls_clips.h` is generated but **committed**, so the machine that builds the
bench firmware needs neither the dataset nor a Python environment — a checkout
and PlatformIO are enough. Regenerate and re-commit it only when changing which
clips are embedded (`make firmware-clips CLIPS=n`).

The mic path, if you want it later:

```sh
pio run -e nano33ble -t upload
python tools/capture_serial.py --port /dev/ttyACM0 --seconds 15   # -> results/*.wav
```

### ESP32-S3 wiring

`src/main_esp32s3.cpp` assumes a 24-bit I2S MEMS mic (INMP441 / ICS-43434
class) on BCLK = 4, WS = 5, DIN = 6. Change `I2S_PIN_*` at the top of that file
for a different board or an analogue front end.

---

## Known differences from the Python pipeline

1. **Causal, single-pass filtering** instead of zero-phase `sosfiltfilt` —
   deliberate; quantified by `make firmware-causal-check`.
2. **`float32` arithmetic** instead of `float64`. Measured at the quantisation
   floor (~6e-8 absolute, ~1e-5 relative to signal RMS); negligible next to
   point 1.
3. **A decimation stage with no Python counterpart** — *mic path only*, because
   the mics run at 16 kHz and the dataset is at 4000 Hz. The bench path does not
   decimate and has no such stage.

On the bench path that is the complete list: same coefficients, same input
samples, same sample rate. Points 1 and 2 are the only reasons an on-device
number can differ from a Python one, and both are quantified.

The mic path adds a fourth difference that cannot be quantified: **a different
acquisition front end.** The dataset was recorded with a digital stethoscope on
a clinical manikin; these boards use a MEMS microphone with its own gain,
bandwidth and noise floor. SDR figures from a live capture are not comparable to
the dataset numbers in `results/` without re-establishing a reference — which is
the whole reason the measurement workflow replays the dataset instead.
