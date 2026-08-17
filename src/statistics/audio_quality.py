"""
Audio quality + per-class/per-location statistics for the HLS-CMDS dataset.

Scans every .wav file referenced by HS.csv, LS.csv, and Mix.csv (via
load_dataset.py) and reports, per file:
    - duration (s)
    - sample rate (Hz)
    - channel count
    - clipping (fraction of samples pinned at full-scale +/-32767)

Then aggregates those per-file properties by class (Heart/Lung Sound Type)
and by recording Location.

Usage:
    python src/statistics/audio_quality.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for load_dataset
from load_dataset import load_hs, load_ls, load_mix

OUTPUT_DIR = Path(__file__).resolve().parent

# A 16-bit PCM sample is clipped if it sits at the full-scale rail (+/-32767/32768).
CLIP_THRESHOLD_INT16 = 32767


def audio_properties(path: str) -> dict:
    """Read one .wav file's format + clipping stats without decoding to float."""
    info = sf.info(path)
    data, _ = sf.read(path, dtype="int16", always_2d=True)
    clipped_mask = np.abs(data) >= CLIP_THRESHOLD_INT16
    n_clipped = int(clipped_mask.any(axis=1).sum())
    return {
        "duration_s": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "n_frames": info.frames,
        "n_clipped_samples": n_clipped,
        "clipping_ratio": n_clipped / info.frames if info.frames else 0.0,
        "is_clipped": n_clipped > 0,
    }


def _with_audio_properties(df: pd.DataFrame, path_col: str) -> pd.DataFrame:
    props = df[path_col].apply(audio_properties).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), props.reset_index(drop=True)], axis=1)


def build_hs_stats() -> pd.DataFrame:
    return _with_audio_properties(load_hs(), "audio_path")


def build_ls_stats() -> pd.DataFrame:
    return _with_audio_properties(load_ls(), "audio_path")


def build_mix_stats() -> pd.DataFrame:
    """One row per triplet member (heart/lung/mixed) so audio checks cover all 435 mix files."""
    mix_df = load_mix()
    rows = []
    for _, row in mix_df.iterrows():
        for role, path_col, id_col in [
            ("heart", "heart_audio_path", "Heart Sound ID"),
            ("lung", "lung_audio_path", "Lung Sound ID"),
            ("mixed", "mixed_audio_path", "Mixed Sound ID"),
        ]:
            rows.append({
                "Mixed Sound ID": row["Mixed Sound ID"],
                "role": role,
                "id": row[id_col],
                "Gender": row["Gender"],
                "Heart Sound Type": row["Heart Sound Type"],
                "Lung Sound Type": row["Lung Sound Type"],
                "Location": row["Location"],
                "audio_path": row[path_col],
            })
    return _with_audio_properties(pd.DataFrame(rows), "audio_path")


def per_class_stats(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    grouped = df.groupby(class_col).agg(
        n=("audio_path", "count"),
        duration_mean=("duration_s", "mean"),
        duration_std=("duration_s", "std"),
        duration_min=("duration_s", "min"),
        duration_max=("duration_s", "max"),
        sample_rates=("sample_rate", lambda s: sorted(s.unique().tolist())),
        n_clipped=("is_clipped", "sum"),
    )
    grouped["clipped_pct"] = 100 * grouped["n_clipped"] / grouped["n"]
    return grouped.sort_index()


def per_location_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("Location").agg(
        n=("audio_path", "count"),
        duration_mean=("duration_s", "mean"),
        duration_std=("duration_s", "std"),
        sample_rates=("sample_rate", lambda s: sorted(s.unique().tolist())),
        n_clipped=("is_clipped", "sum"),
    )
    grouped["clipped_pct"] = 100 * grouped["n_clipped"] / grouped["n"]
    return grouped.sort_index()


def summarize_all() -> dict:
    hs, ls, mix = build_hs_stats(), build_ls_stats(), build_mix_stats()
    all_files = pd.concat([
        hs[["duration_s", "sample_rate", "channels", "is_clipped"]],
        ls[["duration_s", "sample_rate", "channels", "is_clipped"]],
        mix[["duration_s", "sample_rate", "channels", "is_clipped"]],
    ], ignore_index=True)

    return {
        "hs": hs,
        "ls": ls,
        "mix": mix,
        "overall": {
            "n_files": len(all_files),
            "sample_rates": sorted(all_files["sample_rate"].unique().tolist()),
            "channels": sorted(all_files["channels"].unique().tolist()),
            "duration_mean": all_files["duration_s"].mean(),
            "duration_min": all_files["duration_s"].min(),
            "duration_max": all_files["duration_s"].max(),
            "n_clipped": int(all_files["is_clipped"].sum()),
            "clipped_pct": 100 * all_files["is_clipped"].mean(),
        },
        "hs_by_type": per_class_stats(hs, "Heart Sound Type"),
        "ls_by_type": per_class_stats(ls, "Lung Sound Type"),
        "hs_by_location": per_location_stats(hs),
        "ls_by_location": per_location_stats(ls),
    }


if __name__ == "__main__":
    result = summarize_all()
    overall = result["overall"]

    print(f"Scanned {overall['n_files']} audio files")
    print(f"  Sample rate(s): {overall['sample_rates']} Hz")
    print(f"  Channel count(s): {overall['channels']}")
    print(f"  Duration: mean={overall['duration_mean']:.2f}s "
          f"min={overall['duration_min']:.2f}s max={overall['duration_max']:.2f}s")
    print(f"  Clipped files: {overall['n_clipped']} ({overall['clipped_pct']:.1f}%)")

    print("\n=== Heart Sound Type: per-class stats ===")
    print(result["hs_by_type"].to_string())

    print("\n=== Lung Sound Type: per-class stats ===")
    print(result["ls_by_type"].to_string())

    print("\n=== Heart recordings: per-location stats ===")
    print(result["hs_by_location"].to_string())

    print("\n=== Lung recordings: per-location stats ===")
    print(result["ls_by_location"].to_string())

    csv_dir = OUTPUT_DIR / "audio_quality_reports"
    csv_dir.mkdir(exist_ok=True)
    result["hs_by_type"].to_csv(csv_dir / "hs_by_type.csv")
    result["ls_by_type"].to_csv(csv_dir / "ls_by_type.csv")
    result["hs_by_location"].to_csv(csv_dir / "hs_by_location.csv")
    result["ls_by_location"].to_csv(csv_dir / "ls_by_location.csv")
    print(f"\nPer-class / per-location CSVs written to {csv_dir}/")
