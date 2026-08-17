# TEEP_Internship

## Layout

```
thang/
├── HLS-CMDS/            # dataset repo (Heart and Lung Sounds Dataset), incl. Dataset.v2/{HS,LS,mix}/ audio + CSVs
├── src/                 # analysis scripts, kept outside HLS-CMDS so they aren't tied to that repo
│   ├── load_dataset.py      # loads HS/LS/Mix CSVs, resolves + validates audio paths
│   ├── metrics.py            # BSS Eval (SDR/SIR/SAR) for heart/lung separation
│   ├── split.py               # leakage-safe, triplet-level fold assignment
│   ├── eval_harness.py        # k-fold cross-validation wiring split.py + metrics.py
│   ├── test_metrics.py, test_split.py
│   ├── visualization/        # plotting scripts
│   │   ├── audio_plotter.py       # waveforms for the 3 example files in HLS-CMDS/Examples
│   │   ├── audio_spectrogram.py   # mel-spectrograms for the same 3 example files
│   │   ├── donut_chart.py         # sound-type donut chart, no dataset files needed
│   │   ├── plot_per_class.py      # waveform/spectrogram grid, one file per class
│   │   └── plots/                 # generated PNGs (gitignored)
│   └── statistics/           # dataset/audio statistics
│       ├── audio_quality.py       # duration/sample-rate/clipping, per class + location
│       └── audio_quality_reports/ # generated CSVs (gitignored)
└── requirements.txt     # pinned deps for the audio_env conda environment
```

`src/visualization/*.py` and `src/statistics/audio_quality.py` resolve dataset/example paths relative
to their own location, so they only work with this exact layout — `src/`, `src/visualization/`,
`src/statistics/`, and `HLS-CMDS/` must keep their relative positions under `thang/`.

## Environment

Scripts require librosa, matplotlib, numpy, pandas, and (for `audio_plotter.py`) ipython. These are
installed in the conda environment `audio_env` (`/home/internship/miniconda3/envs/audio_env`), not the
`base` Anaconda environment.

Activate it before running anything:

```
source /home/internship/miniconda3/etc/profile.d/conda.sh
conda activate audio_env
```

(`conda activate audio_env` by name may fail if your default `conda` doesn't have
`/home/internship/miniconda3/envs` in its `envs_dirs` — activate by full path instead:
`conda activate /home/internship/miniconda3/envs/audio_env`.)

## Running the scripts

From `src/`, with `audio_env` active:

```
python3 load_dataset.py                    # loads HS/LS/Mix CSVs into pandas DataFrames, validates all audio paths
python3 metrics.py                         # BSS Eval sanity baselines on one real mix row
python3 split.py                           # prints the leakage-safe fold assignment + dictionary pool sizes
python3 eval_harness.py                    # runs a no-op baseline through the full k-fold harness
python3 -m pytest test_metrics.py test_split.py -v

python3 visualization/audio_plotter.py     # plots + saves waveforms for the 3 example files in HLS-CMDS/Examples
python3 visualization/audio_spectrogram.py # plots mel-spectrograms for the same 3 example files
python3 visualization/donut_chart.py       # standalone chart, no dataset files needed
python3 visualization/plot_per_class.py    # waveform + spectrogram grid, one representative file per class

python3 statistics/audio_quality.py        # per-class/per-location duration, sample rate, clipping stats
```
