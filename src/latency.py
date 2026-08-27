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
N_TRIALS_SLOW = 3  # EVMD's own K-sweep is expensive per call -- fewer reps, same protocol shape


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
        "wall_median_s": float(np.median(wall)),
        "wall_iqr_s": float(np.percentile(wall, 75) - np.percentile(wall, 25)),
        "cpu_median_s": float(np.median(cpu)),
        "cpu_iqr_s": float(np.percentile(cpu, 75) - np.percentile(cpu, 25)),
    }


def _small_pool(df, n: int = 5):
    return df.iloc[:n].reset_index(drop=True)


def measure_bandpass_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline1 import bandpass_separate

    return {"method": "Baseline 1 (bandpass)", **time_calls(lambda: bandpass_separate(mixed, sr))}


def measure_supervised_nmf_latency(mixed: np.ndarray, sr: int, hs_small, ls_small) -> dict:
    from baseline.baseline2 import make_supervised_nmf_baseline

    separate_fn = make_supervised_nmf_baseline(seed=0)(hs_small, ls_small)  # dictionary fit once, untimed
    return {"method": "Baseline 2 (supervised NMF)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_standard_nmf_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline3 import make_standard_nmf_baseline

    separate_fn = make_standard_nmf_baseline(seed=0)(None, None)  # hs/ls unused, per this baseline's own contract
    return {"method": "Baseline 3 (standard NMF)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_mssa_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline4 import mssa_separate

    return {"method": "Baseline 4 (MSSA)", **time_calls(lambda: mssa_separate(mixed, sr))}


def measure_evmd_latency(mixed: np.ndarray, sr: int) -> dict:
    from baseline.baseline5 import evmd_separate

    # Fewer reps (N_TRIALS_SLOW): EVMD's own K=2..10 sweep already costs
    # ~15s/call by itself (BACKLOG.md's own measured figure) -- same
    # protocol shape (warm-up + timed repetitions, median+IQR), just fewer
    # of them so this doesn't add disproportionate load to an already
    # contended machine.
    return {"method": "Baseline 5 (EVMD)", **time_calls(lambda: evmd_separate(mixed, sr), n_trials=N_TRIALS_SLOW)}


def measure_convtasnet_latency(mixed: np.ndarray, sr: int, hs_small, ls_small) -> dict:
    from convtasnet import make_convtasnet_baseline

    # Reduced training config (matches test_convtasnet.py's own testability
    # pattern): training quality doesn't affect *inference* speed, only
    # setup cost, which is untimed here anyway.
    separate_fn = make_convtasnet_baseline(seed=0, max_epochs=2, steps_per_epoch=3, batch_size=2)(hs_small, ls_small)
    return {"method": "Baseline 6 (Conv-TasNet-lite)", **time_calls(lambda: separate_fn(mixed, sr))}


def measure_classifier_latency(y: np.ndarray, sr: int) -> dict:
    from heart_classifier import assign_classifier_folds, predict_one, train_fold_classifiers

    hs_df = assign_classifier_folds(n_folds=5, seed=0)
    classifiers = train_fold_classifiers(hs_df, n_folds=5)  # fit once, untimed
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

    display_df = df[["n_trials", "wall_median_ms", "wall_iqr_ms", "cpu_median_ms", "cpu_iqr_ms"]]
    body = "\n\n".join([
        section(
            "Inference latency per method",
            "one real 60,000-sample (15s @ 4000 Hz) mixture/recording; median + IQR across n_trials timed repetitions after 1 untimed warm-up",
            df_to_html(display_df, index_label="method", float_fmt="{:.2f}"),
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
            "target (PROTOCOL.md Sec. 2) -- no claim is made about edge-hardware latency."
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
