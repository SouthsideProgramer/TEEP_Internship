"""Load the HS, LS, and Mix CSVs into pandas DataFrames and resolve each
row to its audio file inside the extracted HS/, LS/, and mix/ folders.

Usage:
    from load_dataset import load_hs, load_ls, load_mix, load_audio

    hs_df = load_hs()
    y, sr = load_audio(hs_df.loc[0, "audio_path"])
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "HLS-CMDS" / "Dataset.v2"

# LS.csv's "Lung Sound ID" column uses different abbreviations for these two
# types than the filenames actually stored under LS/, e.g. the CSV has
# "F_C_LUA" but the file on disk is "LS/F_FC_LUA.wav". Correct for it when
# resolving audio paths.
LS_TYPE_TO_FILE_ABBREV = {
    "Fine Crackles": "FC",
    "Coarse Crackles": "CC",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def load_hs() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "HS.csv"))
    df["audio_path"] = str(DATA_DIR / "HS") + "/" + df["Heart Sound ID"] + ".wav"
    return df


def load_ls() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "LS.csv"))
    id_abbrev = df["Lung Sound ID"].str.split("_").str[1]
    abbrev = df["Lung Sound Type"].map(LS_TYPE_TO_FILE_ABBREV).fillna(id_abbrev)
    file_id = df["Gender"] + "_" + abbrev + "_" + df["Location"]
    df["audio_path"] = str(DATA_DIR / "LS") + "/" + file_id + ".wav"
    return df


def load_mix() -> pd.DataFrame:
    df = _clean(pd.read_csv(DATA_DIR / "Mix.csv"))
    mix_dir = str(DATA_DIR / "mix")
    df["heart_audio_path"] = mix_dir + "/" + df["Heart Sound ID"] + ".wav"
    df["lung_audio_path"] = mix_dir + "/" + df["Lung Sound ID"] + ".wav"
    df["mixed_audio_path"] = mix_dir + "/" + df["Mixed Sound ID"] + ".wav"
    return df


def load_audio(path: str, sr=None):
    """Decode a .wav file with librosa."""
    import librosa

    return librosa.load(path, sr=sr)


def _validate(df: pd.DataFrame, path_col: str, label: str) -> None:
    missing = [p for p in df[path_col] if not Path(p).is_file()]
    if missing:
        raise ValueError(f"{label}: {len(missing)} rows reference missing audio files, e.g. {missing[:5]}")


if __name__ == "__main__":
    hs_df = load_hs()
    ls_df = load_ls()
    mix_df = load_mix()

    _validate(hs_df, "audio_path", "HS")
    _validate(ls_df, "audio_path", "LS")
    for path_col, label in [
        ("heart_audio_path", "Mix (heart)"),
        ("lung_audio_path", "Mix (lung)"),
        ("mixed_audio_path", "Mix (mixed)"),
    ]:
        _validate(mix_df, path_col, label)

    print(f"HS:  {len(hs_df)} rows, all audio files found")
    print(f"LS:  {len(ls_df)} rows, all audio files found")
    print(f"Mix: {len(mix_df)} rows, all audio files found")
