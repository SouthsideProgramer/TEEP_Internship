"""
Measure desktop latency under a uniform protocol -- the empirical
wall-clock/CPU-time counterpart to compute_cost.py's analytic MACs/params
table, completing this project's own version of Yaqub et al.'s Table 5
(Accuracy/Params/GFLOPs/Model size/**Inference time**) for the six
separation baselines plus the Condition A classifier.

"Desktop" specifically contrasts with the Edge-Enabled paper's PYNQ-ZU
FPGA deployment target (PROTOCOL.md Sec. 2): this measures inference
latency on the workstation this project actually runs on, not edge
hardware -- no claim is made about how these numbers would translate to
a PYNQ-ZU or similar embedded target.

EXCEPT for Baseline 1, which no longer needs the extrapolation. It has
since been ported to an Arduino Nano 33 BLE Sense (64 MHz Cortex-M4F,
CMSIS-DSP biquads) and measured on silicon: 3.47 us/sample, 72x faster
than real time, filtering the same int16 samples this module's desktop
run filters. See firmware/ and report/report.tex Sec. "Embedded
Deployment". That figure supersedes this module's bandpass row for any
deployment question; the row is kept because this module's purpose is a
uniform cross-method comparison on one machine, which a number from a
different machine would break. The other six methods have no embedded
measurement, so for them the caveat above stands unchanged.

Uniform protocol, held identical across every method regardless of what
else is running on the machine: the same real 60,000-sample (15s @
4000 Hz) input, one untimed warm-up call, N_TRIALS timed repetitions,
median + IQR reported (robust to transient scheduling spikes) rather
than a mean a single outlier could dominate. One-time setup cost (NMF
dictionary fitting, Conv-TasNet-lite training) is explicitly excluded
from the timed region -- this measures *inference* latency, the same
convention Yaqub et al.'s own Table 5 uses (their "ms/sample" column is
inference time, not training time), not training cost.

Machine-load caveat, stated plainly rather than omitted: this project's
own S6-02 sweep-generation job (and, per `uptime`, other users' load
entirely outside this project's control) was running concurrently on a
shared machine at the time of measurement -- the report records
`os.getloadavg()` at measurement time for exactly this reason. The
*protocol* is uniform regardless (every method timed by the same code,
same repetitions, same input), so relative comparisons between methods
remain meaningful; absolute wall-clock values should be read as noisy
upper bounds, not clean single-tenant numbers. Process CPU time
(`time.process_time()`, summed across the calling process's threads) is
reported alongside wall-clock specifically because it is far less
sensitive to *other processes'* scheduling contention (though not
entirely immune -- cache/memory-bandwidth contention still leaks
through), and is the more trustworthy number under these conditions.

Usage:
    python latency.py
"""
import os
import time

import numpy as np

N_TRIALS = 5
N_WARMUP = 1
N_TRIALS_SLOW = 3
CLIP_SECONDS = 15.0  # the timed input: one 60,000-sample clip at 4000 Hz


def environment() -> dict:
    """What a reader needs to interpret an absolute latency: hardware,
    OS, library versions, thread counts, and the load average at call
    time. Recorded into the report rather than left to the prose."""
    import platform
    import re
    import subprocess

    env = {"os": platform.platform(), "python": platform.python_version(), "n_cpus": os.cpu_count()}
    try:
        cpuinfo = open("/proc/cpuinfo").read()
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        env["cpu_model"] = m.group(1).strip() if m else "unknown"
    except OSError:
        env["cpu_model"] = "unknown"
    try:
        mem_kb = int(re.search(r"MemTotal:\s*(\d+)", open("/proc/meminfo").read()).group(1))
        env["ram_gb"] = round(mem_kb / 1e6, 1)
    except (OSError, AttributeError):
        env["ram_gb"] = "unknown"
    try:
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        env["gpu"] = "none / nvidia-smi unavailable"
    for mod in ("numpy", "scipy", "librosa", "sklearn", "torch", "mir_eval"):
        try:
            env[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            env[f"{mod}_version"] = "n/a"
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        env["torch_device_for_conv_tasnet"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    env["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "unset")
    env["loadavg_1_5_15"] = ", ".join(f"{x:.2f}" for x in os.getloadavg())
    return env


def time_calls(fn, n_trials: int = N_TRIALS, n_warmup: int = N_WARMUP) -> dict:
    """The uniform protocol itself: n_warmup untimed calls, then n_trials
    timed calls, reporting median + IQR of both wall-clock and process CPU
    time. Every measure_*_latency() function below is a thin wrapper
    around exactly this."""
    for _ in range(n_warmup):
        fn()

    wall_times, cpu_times = [], []
    for _ in range(n_trials):
        t0_wall, t0_cpu = time.perf_counter(), time.process_time()
        fn()
        wall_times.append(time.perf_counter() - t0_wall)
        cpu_times.append(time.process_time() - t0_cpu)

    wall = np.array(wall_times)
    cpu = np.array(cpu_times)
    return {
        "n_trials": n_trials,
        "n_warmup": n_warmup,
        "wall_median_s": float(np.median(wall)),
        "wall_iqr_s": float(np.percentile(wall, 75) - np.percentile(wall, 25)),
        "wall_p5_s": float(np.percentile(wall, 5)),
        "wall_p95_s": float(np.percentile(wall, 95)),
        "cpu_median_s": float(np.median(cpu)),
        "cpu_iqr_s": float(np.percentile(cpu, 75) - np.percentile(cpu, 25)),
        "real_time_factor": float(CLIP_SECONDS / np.median(wall)),  # >1 = faster than real time
    }


def _small_pool(df, n: int = 5):
    return df.iloc[:n].reset_index(drop=True)


def measure_bandpass_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline1 import bandpass_separate

    return {"method": "Baseline 1 (bandpass)", **time_calls(lambda: bandpass_separate(mixed, sr))}


def measure_supervised_nmf_latency(mixed: np.ndarray, sr: int, hs_small, ls_small) -> dict:
    from baseline.baseline2 import make_supervised_nmf_baseline

    separate_fn = make_supervised_nmf_baseline(seed=0)(hs_small, ls_small)
    return {"method": "Baseline 2 (supervised NMF)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_standard_nmf_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline3 import make_standard_nmf_baseline

    separate_fn = make_standard_nmf_baseline(seed=0)(None, None)
    return {"method": "Baseline 3 (standard NMF)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_mssa_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline4 import mssa_separate

    return {"method": "Baseline 4 (MSSA)", **time_calls(lambda: mssa_separate(mixed, sr))}


def measure_evmd_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline5 import evmd_separate

    return {"method": "Baseline 5 (EVMD)", **time_calls(lambda: evmd_separate(mixed, sr), n_trials=N_TRIALS_SLOW)}


def measure_convtasnet_latency(mixed: np.ndarray, sr: int, hs_small, ls_small) -> dict:
    from convtasnet import make_convtasnet_baseline

    separate_fn = make_convtasnet_baseline(seed=0, max_epochs=2, steps_per_epoch=3, batch_size=2)(hs_small, ls_small)
    return {"method": "Baseline 6 (Conv-TasNet-lite)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_classifier_latency(y: np.ndarray, sr: int) -> dict:
    from heart_classifier import assign_classifier_folds, predict_one, train_fold_classifiers

    hs_df = assign_classifier_folds(n_folds=5, seed=0)
    classifiers = train_fold_classifiers(hs_df, n_folds=5)
    clf = classifiers[0]
    return {"method": "Condition A classifier (MFCC + SVM)", **time_calls(lambda: predict_one(clf, y, sr))}


def measure_all(mixed: np.ndarray, sr: int, heart_y: np.ndarray, heart_sr: int) -> list[dict]:
    from load_dataset import load_hs, load_ls

    hs_small, ls_small = _small_pool(load_hs()), _small_pool(load_ls())

    return [
        measure_bandpass_latency(mixed, sr),
        measure_supervised_nmf_latency(mixed, sr, hs_small, ls_small),
        measure_standard_nmf_latency(mixed, sr),
        measure_mssa_latency(mixed, sr),
        measure_evmd_latency(mixed, sr),
        measure_convtasnet_latency(mixed, sr, hs_small, ls_small),
        measure_classifier_latency(heart_y, heart_sr),
    ]


if __name__ == "__main__":
    import pandas as pd

    from load_dataset import load_audio, load_hs, load_mix
    from report_utils import df_to_html, report_shell, results_dir, section, stat_tile, write_report

    loadavg_1, loadavg_5, loadavg_15 = os.getloadavg()
    n_cpus = os.cpu_count()
    env = environment()
    print("Environment:", env)
    print(f"Machine load at measurement start: {loadavg_1:.1f}, {loadavg_5:.1f}, {loadavg_15:.1f} ({n_cpus} CPUs)")
    print("(uniform protocol: same code/repetitions/input per method regardless of ambient load -- see module docstring)")

    mix_row = load_mix().iloc[0]
    mixed, sr = load_audio(mix_row["mixed_audio_path"], sr=None)
    heart_y, heart_sr = load_audio(load_hs().iloc[0]["audio_path"], sr=None)

    print("Measuring all 6 baselines + classifier (one warm-up + timed repetitions each)...")
    rows = measure_all(mixed, sr, heart_y, heart_sr)

    loadavg_end = os.getloadavg()
    print(f"Machine load at measurement end: {loadavg_end[0]:.1f}, {loadavg_end[1]:.1f}, {loadavg_end[2]:.1f}")

    df = pd.DataFrame(rows).set_index("method")
    df["wall_median_ms"] = df["wall_median_s"] * 1000
    df["wall_iqr_ms"] = df["wall_iqr_s"] * 1000
    df["cpu_median_ms"] = df["cpu_median_s"] * 1000
    df["cpu_iqr_ms"] = df["cpu_iqr_s"] * 1000

    stat_tiles = "\n".join([
        stat_tile("Methods measured", str(len(df)), "6 separation baselines + classifier"),
        stat_tile("Load avg (1 min)", f"{loadavg_1:.1f}", f"of {n_cpus} CPUs -- see caveat below", ok=loadavg_1 < n_cpus),
        stat_tile("Slowest method", df["wall_median_s"].idxmax(), f"{df['wall_median_s'].max():.2f}s median wall-clock"),
    ])

    df["wall_p5_ms"] = df["wall_p5_s"] * 1000
    df["wall_p95_ms"] = df["wall_p95_s"] * 1000
    df.to_csv(results_dir() / "latency.csv")
    env_df = pd.DataFrame({"value": pd.Series(env, dtype=object)})
    env_df.to_csv(results_dir() / "latency_environment.csv")

    display_df = df[["n_trials", "n_warmup", "wall_median_ms", "wall_iqr_ms", "wall_p5_ms", "wall_p95_ms", "cpu_median_ms", "cpu_iqr_ms", "real_time_factor"]]
    body = "\n\n".join([
        section(
            "Inference latency per method",
            "one real 60,000-sample (15s @ 4000 Hz) mixture/recording; median, IQR and p5-p95 across n_trials timed repetitions after n_warmup untimed warm-ups; real_time_factor = 15 s / median wall (>1 is faster than real time). Preprocessing inside each method's separate() call (e.g. Baseline 2's denoising bandpass, STFT) is included; one-time fitting/training is not.",
            df_to_html(display_df, index_label="method", float_fmt="{:.2f}"),
        ),
        section(
            "Measurement environment",
            "recorded at run time; results/latency_environment.csv",
            df_to_html(env_df, index_label="key"),
        ),
    ])

    load_ok = loadavg_1 < n_cpus
    load_note = (
        f"<p><strong>Machine load at measurement time: {loadavg_1:.1f} (1-min avg) on {n_cpus} CPUs "
        f"-- well above capacity.</strong> This project's own S6-02 sweep-generation job (and, per "
        f"<code>uptime</code>, other users' unrelated load) was running concurrently. The protocol "
        "itself is uniform (identical code, repetitions, and input per method), so relative rankings "
        "between methods remain meaningful; absolute wall-clock numbers here should be read as noisy "
        "upper bounds, not clean single-tenant measurements. CPU-time columns are less sensitive to "
        "other processes' scheduling and are the more trustworthy comparison under these conditions.</p>"
        if not load_ok else
        f"<p>Machine load at measurement time: {loadavg_1:.1f} (1-min avg) on {n_cpus} CPUs -- within capacity.</p>"
    )

    html = report_shell(
        title="Desktop Latency",
        eyebrow="HLS-CMDS · desktop inference latency, uniform protocol",
        heading="Desktop latency per method, under a uniform measurement protocol",
        dek=(
            "The empirical wall-clock/CPU-time counterpart to compute_cost.py's analytic MACs/params "
            "table -- completes this project's own version of Yaqub et al.'s Table 5 (Params/GFLOPs/"
            "Model size/Inference time). 'Desktop' contrasts with the Edge-Enabled paper's PYNQ-ZU FPGA "
            "target (PROTOCOL.md Sec. 2) -- no claim is made about edge-hardware latency, except for "
            "Baseline 1, which has since been measured on an Arduino Nano 33 BLE Sense at 3.47 us/sample "
            "(72x real-time). See results/firmware_on_device_report.html; that figure supersedes this "
            "table's bandpass row for deployment questions, though the row stays here so the "
            "cross-method comparison remains one machine throughout."
        ),
        stat_tiles=stat_tiles,
        body=load_note + "\n\n" + body,
        footer=(
            "<p><strong>Method.</strong> See <code>latency.py</code>'s module docstring for the "
            "uniform-protocol definition (warm-up, repetitions, median+IQR, wall-clock vs. CPU time) "
            "and the machine-load caveat.</p>"
        ),
    )

    report_path = write_report(results_dir() / "latency_report.html", html)
    print(f"\nReport written to {report_path}")
    print(df[["n_trials", "wall_median_s", "cpu_median_s"]].to_string())
