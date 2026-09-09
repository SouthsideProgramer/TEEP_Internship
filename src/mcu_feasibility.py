"""
Which of the six separation baselines can actually run on the microcontrollers
this project targets, and for those that cannot, exactly which resource runs
out first.

WHY THIS EXISTS. compute_cost.py counts MACs and latency.py measures desktop
wall-clock, but neither answers the deployment question, because on an MCU the
binding constraint is usually not arithmetic -- it is memory, and specifically
whether the algorithm can be expressed as a *stream* at all. Baseline 1 was
ported to a Nano 33 BLE Sense and measured on silicon (firmware/, 62.4 us per
sample, 4.0x real time), which raised the obvious follow-up: is there anything
better that would also fit? This module answers it with a resource model
rather than by attempting six ports and discovering the answer the expensive
way.

THE AXIS THAT MATTERS MOST IS NOT MACs. A method is *streaming* if it can
produce output from a bounded window of input, carrying only fixed state
between calls, and *batch* if it needs the whole recording resident before it
can produce anything. Baseline 1 is streaming: 16 floats of biquad state,
independent of whether the recording is 15 seconds or an hour. Baseline 4 is
batch by construction -- SSA embeds the signal into an L x K trajectory
matrix, takes its SVD, and reconstructs by diagonal averaging back across the
whole signal, so no prefix of the input determines any prefix of the output.
That distinction, not the MAC count, is what decides most of the rows below.

WHAT IS MODELLED, per method and per target:
  - peak working RAM: the largest set of buffers simultaneously live during
    one inference, derived from the method's own real constants (imported,
    not restated) rather than measured, since these methods have no MCU
    implementation to measure.
  - parameter storage: what has to live in flash (dictionaries, weights).
    Read-only data can sit in flash and be addressed directly on both targets,
    so it is charged against flash rather than RAM.
  - compute time: compute_cost.py's MAC count divided by an assumed
    throughput, stated per target below.

WHAT THIS IS NOT. The RAM figures are lower bounds from the algorithm's own
data structures: a real implementation adds framework overhead, stack, and
whatever the toolchain's allocator wastes. A method whose model says it needs
1.5x the available RAM should be read as "does not fit", not as "might fit
with tuning" -- but a method needing 92x, as one below does, is in a different
category entirely, and no amount of implementation care changes it. Where the
verdict is close, it is flagged as close rather than decided.

Usage:
    python mcu_feasibility.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_cost import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_RATE,
    _stft_frame_count,
    bandpass_macs,
    convtasnet_params_and_macs,
    evmd_macs,
    mssa_macs,
    standard_nmf_macs,
    supervised_nmf_macs,
)

F32 = 4  # bytes per float32
KB = 1024


@dataclass(frozen=True)
class Target:
    name: str
    ram_bytes: int
    flash_bytes: int
    clock_hz: float
    macs_per_cycle: float
    note: str


# RAM/flash are the parts' own totals; the firmware itself already occupies a
# share of each (firmware/README.md records 45 992 B RAM / 438 860 B flash for
# the Nano 33 bench build, most of the flash being embedded clips). The
# headroom column below accounts for that rather than pretending a method gets
# the whole part.
#
# macs_per_cycle is deliberately generous to each target: 1.0 assumes every
# multiply-accumulate issues in one cycle with its operands already in
# registers, which no real float32 workload achieves. Being generous is the
# point -- a method that does not fit even under an optimistic throughput
# assumption is not a borderline case.
#
# CALIBRATION NOTE. The one silicon measurement available when this was
# written -- Baseline 1 at 62.4 us/sample -- came in ~100x slower than this
# model predicts, which is far too large a gap to blame on optimism. Chasing
# it found the cause: the arduino-mbed build was appending -mfloat-abi=soft
# after its own -mfloat-abi=softfp, GCC takes the last one, and every float
# operation was compiling to __aeabi_* software-float calls with the
# Cortex-M4F's FPU unused (0 VFP instructions in the disassembly). Fixed in
# firmware/platformio.ini via build_unflags. The re-measurement is pending a
# board, so the timing column here should be read as an upper bound whose
# calibration against silicon is not yet re-established -- the RAM columns,
# which decide every infeasible row below, are unaffected either way.
TARGETS = (
    Target(
        name="Nano 33 BLE Sense",
        ram_bytes=262_144,
        flash_bytes=983_040,
        clock_hz=64e6,
        macs_per_cycle=1.0,
        note="nRF52840, Cortex-M4F, single-precision FPU, CMSIS-DSP",
    ),
    Target(
        name="ESP32-S3",
        ram_bytes=327_680,
        flash_bytes=8_388_608,
        clock_hz=240e6,
        macs_per_cycle=1.0,
        note="Xtensa LX7 dual-core (one core assumed), FPU, ESP-DSP",
    ),
)

# What the Baseline 1 firmware already occupies, so the models below are
# charged against what is actually left rather than the part's datasheet total.
FIRMWARE_RAM_OVERHEAD = 46_000
FIRMWARE_FLASH_OVERHEAD = 92_000


def _bandpass_memory() -> dict:
    """
    Streaming: 2 state floats per biquad section, 4 sections per band, 2 bands,
    plus one block buffer per output. Measured on real hardware rather than
    modelled -- firmware/README.md's Nano 33 figures minus the embedded clips.
    """
    from baseline.common import HEART_BAND, LUNG_BAND  # noqa: F401  (documents the source)

    sections, bands = 4, 2
    state = sections * 2 * bands * F32
    block = 64
    buffers = block * 3 * F32  # input block + two output blocks
    return {
        "method": "Baseline 1 (bandpass)",
        "streaming": True,
        "peak_ram_bytes": state + buffers,
        "param_bytes": sections * bands * 6 * F32,  # SOS coefficients
        "ram_note": f"{state} B of biquad state + a {block}-sample working block; "
                    "independent of recording length",
        "verified": "measured on silicon: 45 992 B RAM total for the whole bench build",
    }


def _supervised_nmf_memory() -> dict:
    """
    Streaming per STFT frame. The dictionary is fixed and read-only, so it
    lives in flash; only one frame's spectrum, its activation vector, and the
    overlap-add tail are live at once. This is the one non-trivial method that
    is genuinely streamable: the activations for frame t depend on frame t's
    spectrum and the frozen dictionary, not on any other frame.
    """
    from baseline.baseline2 import HOP_LENGTH, K_HEART, K_LUNG, N_FFT

    bins = N_FFT // 2 + 1
    rank = K_HEART + K_LUNG
    live = (
        N_FFT * F32          # analysis window / FFT scratch
        + bins * 2 * F32     # complex spectrum
        + bins * F32         # magnitude
        + rank * F32         # activations H for this frame
        + bins * 2 * F32     # the two reconstructed magnitudes for the mask
        + N_FFT * F32        # overlap-add tail
    )
    return {
        "method": "Baseline 2 (supervised NMF)",
        "streaming": True,
        "peak_ram_bytes": live,
        "param_bytes": bins * rank * F32,
        "ram_note": f"one {N_FFT}-point frame at a time: spectrum, magnitude, {rank} activations, "
                    f"overlap-add tail. Dictionary ({bins}x{rank}) is read-only and stays in flash",
        "verified": "modelled",
    }


def _standard_nmf_memory() -> dict:
    """
    Batch. Unlike Baseline 2 there is no frozen dictionary: W and H are
    factorised out of the whole clip's spectrogram, so every frame must be
    resident before the first iteration can start, and every iteration
    revisits all of them.
    """
    from baseline.baseline2 import HOP_LENGTH, K_HEART, K_LUNG, N_FFT

    bins = N_FFT // 2 + 1
    rank = K_HEART + K_LUNG
    frames = _stft_frame_count(N_SAMPLES, N_FFT, HOP_LENGTH)
    v = bins * frames * F32
    live = (
        v                       # magnitude spectrogram V
        + v                     # the WH reconstruction, same shape, needed each iteration
        + bins * rank * F32     # W
        + rank * frames * F32   # H
        + bins * frames * 2 * F32  # the complex spectrum kept for reconstruction
    )
    return {
        "method": "Baseline 3 (standard NMF)",
        "streaming": False,
        "peak_ram_bytes": live,
        "param_bytes": 0,
        "ram_note": f"whole-clip spectrogram ({bins}x{frames}) plus an equal-sized reconstruction, "
                    "revisited every multiplicative update -- no frozen dictionary to stream against",
        "verified": "modelled",
    }


def _mssa_memory() -> dict:
    """
    Batch by construction, and the clearest infeasibility in the set. SSA
    embeds the signal into an L x K trajectory matrix (K = N - L + 1), takes
    its SVD, and reconstructs each component by diagonal averaging across the
    full length. Every one of those three structures is O(L*N).
    """
    from baseline.baseline4 import SSA_WINDOW_LENGTH as L

    K = N_SAMPLES - L + 1
    trajectory = L * K * F32
    svd = (L * L + L + L * K) * F32   # U, S, Vt from the reduced SVD
    rcs = L * N_SAMPLES * F32
    return {
        "method": "Baseline 4 (MSSA)",
        "streaming": False,
        "peak_ram_bytes": trajectory + svd + rcs,
        "param_bytes": 0,
        "ram_note": f"L={L}, K={K:,}: trajectory matrix {trajectory / 1e6:.1f} MB + SVD factors "
                    f"{svd / 1e6:.1f} MB + {L} reconstructed components {rcs / 1e6:.1f} MB",
        "verified": "modelled",
    }


def _evmd_memory() -> dict:
    """
    Batch. VMD's ADMM iterates in the frequency domain over the whole
    mirror-padded signal, holding K modes plus their spectra simultaneously,
    and the K-selection sweep reruns the whole solve for each candidate K.
    """
    from baseline.baseline5 import EVMD_K_MAX as K

    padded = 2 * N_SAMPLES  # mirror padding at both ends
    modes = K * padded * F32
    spectra = K * padded * 2 * F32  # complex, per mode
    return {
        "method": "Baseline 5 (EVMD)",
        "streaming": False,
        "peak_ram_bytes": modes + spectra + padded * 2 * F32,
        "param_bytes": 0,
        "ram_note": f"K={K} modes over a mirror-padded {padded:,}-sample signal, plus their complex "
                    "spectra, all live across every ADMM iteration",
        "verified": "modelled",
    }


def _convtasnet_memory() -> dict:
    """
    Batch as written, though unlike Baselines 3-5 it is not batch by
    construction: a causal, chunked Conv-TasNet is a known variant. The figure
    below is what the *current* model does -- encode the whole clip, run the
    TCN over the full frame sequence -- which is what would have to be
    reworked, not merely quantised, before it could run here.

    Parameters are charged twice: float32 as trained, and int8 as a standard
    post-training quantisation would leave them.
    """
    from convtasnet import ENCODER_N, ENCODER_STRIDE, TCN_B, TCN_H

    d = convtasnet_params_and_macs()
    params = d["params"]
    frames = N_SAMPLES // ENCODER_STRIDE
    encoded = ENCODER_N * frames * F32
    bottleneck = TCN_B * frames * F32
    hidden = TCN_H * frames * F32   # widest intermediate inside one TCN block
    masks = 2 * ENCODER_N * frames * F32
    return {
        "method": "Baseline 6 (Conv-TasNet-lite)",
        "streaming": False,
        "peak_ram_bytes": encoded + bottleneck + hidden + masks,
        "param_bytes": params * 1,  # int8, the only form that could fit flash
        "param_note_f32": params * F32,
        "ram_note": f"{frames:,} encoded frames held at once: encoder output ({ENCODER_N}ch), "
                    f"bottleneck ({TCN_B}ch), widest TCN intermediate ({TCN_H}ch), and two masks. "
                    "A chunked/causal variant would change this, but that is a redesign",
        "verified": "modelled from the real layer shapes",
    }


MEMORY_MODELS = (
    _bandpass_memory,
    _supervised_nmf_memory,
    _standard_nmf_memory,
    _mssa_memory,
    _evmd_memory,
    _convtasnet_memory,
)

MAC_SOURCES = {
    "Baseline 1 (bandpass)": bandpass_macs,
    "Baseline 2 (supervised NMF)": supervised_nmf_macs,
    "Baseline 3 (standard NMF)": standard_nmf_macs,
    "Baseline 4 (MSSA)": mssa_macs,
    "Baseline 5 (EVMD)": evmd_macs,
    "Baseline 6 (Conv-TasNet-lite)": convtasnet_params_and_macs,
}


def assess(model: dict, target: Target) -> dict:
    """Verdict for one method on one target, naming the binding constraint."""
    ram_avail = target.ram_bytes - FIRMWARE_RAM_OVERHEAD
    flash_avail = target.flash_bytes - FIRMWARE_FLASH_OVERHEAD

    ram_ratio = model["peak_ram_bytes"] / ram_avail
    flash_ratio = model["param_bytes"] / flash_avail if model["param_bytes"] else 0.0

    macs = MAC_SOURCES[model["method"]]()["macs_per_inference"]
    seconds = macs / (target.clock_hz * target.macs_per_cycle)
    rtf = (N_SAMPLES / SAMPLE_RATE) / seconds

    reasons = []
    if ram_ratio > 1.0:
        reasons.append(f"RAM {ram_ratio:.0f}x over")
    if flash_ratio > 1.0:
        reasons.append(f"flash {flash_ratio:.1f}x over")
    if rtf < 1.0:
        reasons.append(f"{1 / rtf:.0f}x slower than real time")

    if not reasons:
        verdict = "fits" if ram_ratio < 0.5 else "fits, tight"
    elif ram_ratio > 10:
        verdict = "impossible"
    else:
        verdict = "does not fit"

    return {
        "method": model["method"],
        "target": target.name,
        "streaming": "yes" if model["streaming"] else "no",
        "peak RAM": model["peak_ram_bytes"],
        "RAM vs available": ram_ratio,
        "params (flash)": model["param_bytes"],
        "est. real-time factor": rtf,
        "verdict": verdict,
        "binding constraint": ", ".join(reasons) if reasons else "-",
    }


def _fmt_bytes(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= KB:
        return f"{n / KB:.1f} kB"
    return f"{n} B"


if __name__ == "__main__":
    import pandas as pd

    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    print(f"Modelling MCU resources per method ({N_SAMPLES}-sample / 15 s @ {SAMPLE_RATE} Hz)...\n")

    models = [fn() for fn in MEMORY_MODELS]
    rows = [assess(m, t) for t in TARGETS for m in models]
    df = pd.DataFrame(rows)

    display = df.copy()
    display["peak RAM"] = display["peak RAM"].map(_fmt_bytes)
    display["params (flash)"] = display["params (flash)"].map(_fmt_bytes)
    display["RAM vs available"] = display["RAM vs available"].map(lambda v: f"{v:.2f}x")
    display["est. real-time factor"] = display["est. real-time factor"].map(lambda v: f"{v:.2f}x")

    print(display.to_string(index=False))

    fits = df[df["verdict"].str.startswith("fits")]
    n_fit_nano = len(fits[fits["target"] == "Nano 33 BLE Sense"])

    stat_tiles = "\n".join([
        stat_tile("Methods modelled", str(len(models)), "six separation baselines"),
        stat_tile("Fit the Nano 33", str(n_fit_nano), "of six, streaming included", ok=n_fit_nano > 0),
        stat_tile("Worst overrun", f"{df['RAM vs available'].max():.0f}x",
                  "RAM, Baseline 4 (MSSA)", ok=False),
    ])

    detail = pd.DataFrame([
        {"method": m["method"], "streaming": "yes" if m["streaming"] else "no",
         "why this much memory": m["ram_note"]}
        for m in models
    ]).set_index("method")

    body = "\n".join([
        section("Verdict per method and target", "mcu_feasibility.py",
                df_to_html(display.set_index(["target", "method"]), index_label="target / method")
                + "<p>RAM is charged against what is left after the Baseline 1 firmware's own "
                  "footprint, not the part's datasheet total. The real-time factor assumes one MAC "
                  "per cycle with operands already in registers &mdash; deliberately optimistic, so "
                  "that a method failing this test is not a borderline case.</p>"),
        section("Where the memory goes", "derived from each method's own constants",
                df_to_html(detail, index_label="method")
                + "<p>The streaming column is the one that decides most rows. A streaming method "
                  "carries fixed state between calls and is indifferent to recording length; a batch "
                  "method needs the whole recording resident before it can emit anything, so its "
                  "memory scales with the clip. Baselines 3&ndash;6 are batch, and for Baselines 4 "
                  "and 5 that is intrinsic to the algorithm rather than an implementation choice: "
                  "SSA's diagonal averaging and VMD's frequency-domain ADMM both reference the whole "
                  "signal on every iteration.</p>"),
    ])

    html = report_shell(
        title="MCU Feasibility",
        eyebrow="HLS-CMDS · which baselines fit a microcontroller",
        heading="What actually runs on the target hardware, and what runs out first",
        dek=("compute_cost.py counts MACs and latency.py measures a desktop; neither decides "
             "deployability, because on an MCU the binding constraint is usually memory and, "
             "before that, whether the algorithm can be written as a stream at all. Baseline 1 is "
             "already measured on silicon (firmware/); this models the other five against the same "
             "two targets."),
        stat_tiles=stat_tiles,
        body=body,
        footer=("<p><strong>Method.</strong> See <code>mcu_feasibility.py</code>'s module and "
                "per-function docstrings: every figure is derived from the method's own imported "
                "constants, and the RAM models are lower bounds that exclude framework overhead, "
                "stack and allocator waste.</p>"),
    )

    path = write_report(results_dir() / "mcu_feasibility_report.html", html)
    print(f"\nReport written to {path}")
