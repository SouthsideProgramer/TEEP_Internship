# Firmware pipeline

Flow charts for `firmware/` — what runs where, what the signal goes through, and
how each stage is verified. Prose and rationale live in [README.md](README.md);
this file is the map.

The board has **no microphone in the measurement workflow**. It replays real
HLS-CMDS mixtures compiled into its flash, so the samples it filters are
bit-identical to the ones the Python baseline filters. That is what makes the
on-device output diffable against scipy and scoreable with the same BSS Eval
metrics as everything in `results/`.

---

## 1. Where each piece runs

Three machines, three phases. Nothing is designed on the MCU, and nothing is
measured on the host.

```mermaid
flowchart TB
    subgraph server["SERVER — design time"]
        direction TB
        bands["src/baseline/common.py<br/>HEART_BAND 20–200 Hz<br/>LUNG_BAND 150–1000 Hz"]
        gen["tools/gen_filter_coeffs.py<br/>scipy.signal.butter"]
        coeffs["lib/hls_filter/hls_filter_coeffs.h<br/>hardcoded SOS, 4+4+4 sections"]
        golden["golden/*.f32<br/>scipy.signal.sosfilt reference"]
        embed["lib/hls_filter/hls_golden_embedded.h<br/>1024-sample slice"]
        clips["tools/embed_clips.py<br/>→ hls_clips.h<br/>HLS-CMDS mixtures as int16"]
        bands --> gen
        gen --> coeffs
        gen --> golden
        gen --> embed
    end

    subgraph build["SERVER — build time"]
        direction TB
        native["env:native<br/>portable C, x86"]
        boards["env:nano33ble / esp32s3<br/>mic firmware"]
        selft["env:*-selftest<br/>golden check on silicon"]
        bench["env:*-bench<br/>dataset replay"]
    end

    subgraph board["BOARD — run time"]
        direction TB
        fw["replay clips → band split<br/>time the filtering<br/>stream the output"]
    end

    subgraph host["HOST — scoring time"]
        direction TB
        grab["tools/capture_raw.py<br/>pyserial only, no dataset needed<br/>(optional: board on another machine)"]
        run["tools/run_on_device.py<br/>--port or --from-file"]
        rep["results/firmware_on_device_report.html<br/>diff vs scipy · BSS Eval · timing"]
        grab -->|"capture.bin"| run
        run --> rep
    end

    coeffs --> native & boards & selft & bench
    golden --> native
    embed --> selft
    clips --> bench
    bench --> fw
    selft -.->|"verify kernels first"| fw
    fw -->|"USB serial, binary records"| grab
    fw -->|"USB serial, binary records"| run
    gtruth["dataset H####/L####<br/>isolated references"] --> run
```

Two edges carry the whole argument. `bands →` means the firmware **imports** its
bands from Baseline 1's module rather than restating them, so the two cannot
drift apart. `clips →` means the board's input is the dataset itself, so the
board's output is comparable to Python's by construction.

---

## 2. Runtime signal path

### Bench path — dataset replay (the measurement path)

Streaming throughout: state is carried across calls, and output is written out
per block, so RAM never scales with clip length. The clips themselves sit in
flash, not RAM.

```mermaid
flowchart TB
    flash["HLS-CMDS mixture in flash<br/>int16, 4000 Hz, 60 000 samples<br/>117 kB per 15 s clip"]
    scale["× 1/32768 → float32<br/>reproduces librosa.load bit for bit"]
    heart["Heart bandpass 20–200 Hz<br/>Butterworth order 4, 4 biquads"]
    lung["Lung bandpass 150–1000 Hz<br/>Butterworth order 4, 4 biquads"]
    timed["timed region:<br/>scaling + both cascades<br/>serial I/O excluded"]
    frame["binary records<br/>clip header · data frames · footer with compute_micros"]
    hostp["tools/run_on_device.py"]

    flash --> scale
    scale --> heart --> frame
    scale --> lung --> frame
    frame --> hostp
    timed -.-> scale
```

**No decimation on this path** — the dataset is already at 4000 Hz, so only the
two bandpass cascades run. That is exactly what Baseline 1 does.

### Mic path — live capture (built, not currently used)

```mermaid
flowchart TB
    mic["MEMS microphone<br/>16 000 Hz"]
    acq["Acquisition<br/>Nano 33: PDM ISR → 2048-sample ring<br/>ESP32-S3: I2S DMA, 4 × 256"]
    scale2["int16 → × 1/32768 → float32"]
    lp["anti-alias lowpass<br/>Butterworth order 8 @ 1400 Hz<br/>~53 dB down at 3 kHz"]
    drop["phase counter mod 4<br/>keep 1 of every 4 samples"]
    base["4000 Hz"]
    split["heart / lung bandpass<br/>same two cascades"]
    cap["tools/capture_serial.py → WAVs"]

    mic --> acq --> scale2 --> lp --> drop --> base --> split --> cap
```

The decimator exists because the mics run at 16 kHz while the dataset is at
4000 Hz; without it, content near 3 kHz folds onto 1 kHz, inside the lung band.
It has no Python counterpart, so it is verified against a synthetic signal.

### One biquad section

All three cascades are the same kernel with different coefficients. The
generated header stores scipy's sign convention; the CMSIS backend negates
`a1`/`a2` at init because CMSIS writes the recursion the other way round.

```mermaid
flowchart LR
    x["x[n]"] --> s1["section 1"] --> s2["section 2"] --> s3["section 3"] --> s4["section 4"] --> y["y[n]"]
    st["state: 2 floats per section<br/>carried across calls —<br/>this is what makes it streaming"] -.-> s1 & s2 & s3 & s4
```

---

## 3. Verification chain

A build that links is not a build that computes the right thing. Three checks
run on the server with no hardware; two more need a board.

```mermaid
flowchart TB
    scipy["scipy.signal.sosfilt<br/>float64, causal single pass"]

    subgraph nohw["No hardware — runs on the server"]
        direction TB
        t1["make firmware-test<br/>golden/*.f32 → src/main_native.c<br/>bandpass · decimation · block sizes · pipeline"]
        t2["make firmware-protocol-test<br/>emulated board over a pty<br/>direct scoring + capture-then-score"]
        t3["HLS_STATIC_ASSERT in hls_stream.h<br/>record sizes pinned at compile time"]
    end

    subgraph hw["Needs a board"]
        direction TB
        t4["*-selftest firmware<br/>1024-sample reference in flash<br/>→ CMSIS-DSP / ESP-DSP kernels"]
        t5["*-bench firmware + run_on_device.py<br/>full clips → diff vs scipy<br/>+ BSS Eval + timing"]
    end

    scipy --> t1
    scipy --> t4
    scipy --> t5
    t1 --> v1["PASS — error equals the<br/>float32 quantisation floor exactly"]
    t2 --> v2["PASS — good stream accepted,<br/>corrupted stream rejected"]
    t4 --> v4["not yet run"]
    t5 --> v5["not yet run"]
```

The host result is stronger than "close enough": the measured error **equals**
the float32 quantisation floor the generator reports, so the C core performs the
same arithmetic scipy would in `float32` rather than merely something similar.

Order matters on hardware. `*-selftest` before `*-bench`: if the vendor DSP
kernel is wrong, every number the bench produces is wrong in a way that looks
like a filter-design problem.

---

## 4. Build environments

```mermaid
flowchart LR
    core["lib/hls_filter/<br/>hls_filter.c<br/>one API"]

    core -->|"no macro"| p["portable C<br/>env:native"]
    core -->|"HLS_FILTER_BACKEND_CMSIS"| c["arm_biquad_cascade_df2T_f32<br/>Nano 33 BLE Sense"]
    core -->|"HLS_FILTER_BACKEND_ESPDSP"| e["dsps_biquad_f32<br/>ESP32-S3"]
```

| Environment | `src/` entry point | Backend | Purpose |
|---|---|---|---|
| `native` | `main_native.c` | portable C | diff against scipy on the host |
| `nano33ble-bench` | `main_bench.cpp` | CMSIS-DSP | **replay dataset, measure** |
| `esp32s3-bench` | `main_bench.cpp` | ESP-DSP | **replay dataset, measure** |
| `nano33ble-selftest` | `main_selftest.cpp` | CMSIS-DSP | verify the kernel on silicon |
| `esp32s3-selftest` | `main_selftest.cpp` | ESP-DSP | verify the kernel on silicon |
| `nano33ble` | `main_nano33ble.cpp` | CMSIS-DSP | PDM capture (future device) |
| `esp32s3` | `main_esp32s3.cpp` | ESP-DSP | I2S capture (future device) |

---

## 5. The one deliberate divergence

```mermaid
flowchart TB
    sos["Same Butterworth SOS coefficients,<br/>same input samples"]
    sos --> py["Python — sosfiltfilt<br/>forward + backward<br/>zero phase<br/>needs the whole clip in RAM"]
    sos --> fw["Firmware — sosfilt<br/>one causal pass<br/>phase response retained<br/>state is a few hundred bytes"]

    py --> cost["Measured over 145 Mix.csv rows:<br/>SDR unchanged ±0.2 dB<br/>SIR −0.13 / −0.50 dB<br/>SAR gap is a mir_eval artifact,<br/>not a quality gain"]
    fw --> cost
```

`make firmware-causal-check` produces that comparison —
`results/firmware_causal_vs_zerophase.html`. The SAR caveat matters: mir_eval's
`_project` fits a *causal* 512-tap distortion filter, which absorbs a causal
IIR's phase response but not a zero-phase filter's acausal one. See
[README.md](README.md#what-it-actually-costs--measured).

On the bench path this is one of only two differences from the Python baseline —
the other being `float32` versus `float64` arithmetic, measured at the
quantisation floor. Same coefficients, same samples, same sample rate.
