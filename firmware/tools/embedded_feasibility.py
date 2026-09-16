"""
Embedded feasibility figures for the paper: the board's captured output
(capture_fpu.bin, Nano 33 BLE Sense with the FPU enabled) against the
desktop CAUSAL reference and, for context, the desktop zero-phase
baseline, on the same three embedded clips.

For each clip and source:
    max |board - causal|, RMS error, relative RMS error (vs the causal
    reference's RMS), SDR of board / causal / zero-phase output, and the
    Architecture 1 (SVM) and Architecture 2 (CNN) predictions on the heart
    output of each -- so "prediction agreement" is board-vs-causal and
    causal-vs-zero-phase, on n = 3 clips (reported as counts, not rates).
Plus the timing the board reported per clip, converted to per-256-sample
block, and the static RAM / Flash from the PlatformIO build.

Usage:
    cd firmware && ../.venv/bin/python tools/embedded_feasibility.py [--capture ../capture_fpu.bin]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from run_on_device import FileSource, receive  # noqa: E402
from gen_filter_coeffs import HEART_BAND, LUNG_BAND, design_bandpass  # noqa: E402

BLOCK = 256          # BENCH_BLOCK in src/main_bench.cpp
SR = 4000
RAM_STATIC_BYTES = 45992     # PlatformIO "RAM: used 45992 bytes from 262144" for env nano33ble-bench (2026-09-13 build)
FLASH_BYTES = 437612         # PlatformIO "Flash: used 437612 bytes from 983040"; 360,000 B of it is the three embedded clips
FLASH_CLIPS_BYTES = 3 * 60000 * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=str(HERE.parent.parent / "capture_fpu.bin"))
    args = ap.parse_args()

    import heart_classifier
    import heart_classifier_cnn
    from baseline.baseline1 import bandpass_separate
    from condition_b import build_condition_b_fold_basis
    from load_dataset import load_mix
    from metrics import evaluate_heart_lung
    from report_utils import results_dir

    clips = receive(FileSource(args.capture))
    mix_df = load_mix().set_index("Mixed Sound ID")
    basis, hs_df = build_condition_b_fold_basis(n_folds=5, seed=0)
    fold_of = basis.set_index("Mixed Sound ID")["fold"]
    clf_svm = heart_classifier.train_fold_classifiers(hs_df, n_folds=5)
    clf_cnn = heart_classifier_cnn.train_fold_classifiers(hs_df, n_folds=5)

    heart_sos, lung_sos = design_bandpass(*HEART_BAND, SR), design_bandpass(*LUNG_BAND, SR)
    rows = []
    for clip in clips:
        row = mix_df.loc[clip["clip_id"]]
        n = clip["n_samples"]
        mixed = sf.read(row["mixed_audio_path"], dtype="int16")[0][:n] / 32768.0
        heart_ref = sf.read(row["heart_audio_path"], dtype="int16")[0][:n] / 32768.0
        lung_ref = sf.read(row["lung_audio_path"], dtype="int16")[0][:n] / 32768.0
        causal = {"heart": sosfilt(heart_sos, mixed), "lung": sosfilt(lung_sos, mixed)}
        zp_h, zp_l = bandpass_separate(mixed, SR)
        zero_phase = {"heart": zp_h, "lung": zp_l}
        board = {"heart": clip["heart"], "lung": clip["lung"]}
        fold = int(fold_of[clip["clip_id"]]) if clip["clip_id"] in fold_of.index else None

        sdr = {name: evaluate_heart_lung(heart_ref, lung_ref, est["heart"], est["lung"])
               for name, est in (("board", board), ("causal", causal), ("zero_phase", zero_phase))}
        preds = {}
        for name, est in (("board", board), ("causal", causal), ("zero_phase", zero_phase)):
            preds[name] = {
                "svm": heart_classifier.predict_one(clf_svm[fold], est["heart"], SR) if fold is not None else None,
                "cnn": heart_classifier_cnn.predict_one(clf_cnn[fold], est["heart"], SR) if fold is not None else None,
            }
        rec = {"clip": clip["clip_id"], "fold": fold, "true_class": heart_classifier.HEART_TYPE_TO_GROUP[row["Heart Sound Type"]],
               "compute_ms": clip["compute_micros"] / 1000.0, "us_per_sample": clip["compute_micros"] / n,
               "ms_per_block_mean": clip["compute_micros"] / 1000.0 / np.ceil(n / BLOCK),
               "block_duty_pct": 100 * (clip["compute_micros"] / 1e6 / np.ceil(n / BLOCK)) / (BLOCK / SR),
               "rtf": (n / SR) / (clip["compute_micros"] / 1e6)}
        for src in ("heart", "lung"):
            d = board[src] - causal[src]
            rec[f"{src}_max_abs_err"] = float(np.max(np.abs(d)))
            rec[f"{src}_rms_err"] = float(np.sqrt(np.mean(d**2)))
            rec[f"{src}_rel_rms_err"] = float(np.sqrt(np.mean(d**2)) / np.sqrt(np.mean(causal[src]**2)))
            for name in ("board", "causal", "zero_phase"):
                rec[f"{src}_sdr_{name}"] = sdr[name][src]["sdr"]
        for arch in ("svm", "cnn"):
            for name in ("board", "causal", "zero_phase"):
                rec[f"pred_{arch}_{name}"] = preds[name][arch]
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("clip")
    df.to_csv(results_dir() / "embedded_feasibility.csv")

    pd.set_option("display.width", 250)
    print(df[["fold", "compute_ms", "us_per_sample", "ms_per_block_mean", "block_duty_pct", "rtf"]].round(3).to_string())
    print(df[[c for c in df.columns if "err" in c]].to_string(float_format=lambda v: f"{v:.3g}"))
    print(df[[c for c in df.columns if "_sdr_" in c]].round(3).to_string())
    print(df[["true_class"] + [c for c in df.columns if c.startswith("pred_")]].to_string())
    for arch in ("svm", "cnn"):
        print(f"{arch}: board==causal {int((df[f'pred_{arch}_board']==df[f'pred_{arch}_causal']).sum())}/{len(df)}; "
              f"causal==zero-phase {int((df[f'pred_{arch}_causal']==df[f'pred_{arch}_zero_phase']).sum())}/{len(df)}")
    print(f"\nstatic RAM {RAM_STATIC_BYTES} B ({RAM_STATIC_BYTES/1024:.1f} kB of 256 kB); flash {FLASH_BYTES} B "
          f"({FLASH_BYTES/1024:.1f} kB), of which {FLASH_CLIPS_BYTES/1024:.1f} kB are the embedded clips -> "
          f"{(FLASH_BYTES-FLASH_CLIPS_BYTES)/1024:.1f} kB of code, mbed-OS and CMSIS-DSP")
    print(f"block: {BLOCK} samples = {1000*BLOCK/SR:.0f} ms of audio")


if __name__ == "__main__":
    main()
