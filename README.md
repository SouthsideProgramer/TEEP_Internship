# TEEP_Internship

## Layout

```
thang/
├── HLS_CMDS/            # dataset (Heart and Lung Sounds Dataset), {HS,LS,Mix}/ audio + CSVs
├── results/             # generated HTML reports + plots (gitignored) -- see below
├── src/                 # analysis scripts, kept outside HLS_CMDS so they aren't tied to that repo
│   ├── load_dataset.py      # loads HS/LS/Mix CSVs, resolves + validates audio paths
│   ├── report_utils.py       # shared HTML report building blocks for results/
│   ├── metrics.py            # BSS Eval (SDR/SIR/SAR) for heart/lung separation
│   ├── split.py               # leakage-safe, triplet-level fold assignment
│   ├── eval_harness.py        # k-fold cross-validation wiring split.py + metrics.py
│   ├── baselines.py           # Baseline 0 (raw mixture) + report-generation glue
│   ├── baseline/               # Baselines 1-5, one module each (bandpass, NMF x2, SSA, EVMD)
│   ├── convtasnet.py           # Baseline 6 (Conv-TasNet-lite, first neural model)
│   ├── test/                   # test_metrics.py, test_split.py, test_load_dataset.py, etc.
│   ├── visualization/        # plotting scripts
│   │   ├── audio_plotter.py       # waveforms for 3 illustrative recordings pulled from HS.csv/LS.csv
│   │   ├── audio_spectrogram.py   # mel-spectrograms for the same 3 example files
│   │   ├── donut_chart.py         # sound-type donut chart, no dataset files needed
│   │   └── plot_per_class.py      # waveform/spectrogram grid, one file per class
│   └── statistics/           # dataset/audio statistics
│       └── audio_quality.py       # duration/sample-rate/clipping, per class + location
└── requirements.txt     # pinned deps for the audio_env conda environment
```

`src/visualization/*.py` and `src/statistics/audio_quality.py` resolve dataset/example paths relative
to their own location, so they only work with this exact layout — `src/`, `src/visualization/`,
`src/statistics/`, and `HLS_CMDS/` must keep their relative positions under `thang/`.

Every script above prints only short progress lines to the terminal; its actual results (tables,
plots) are rendered as a self-contained HTML file under `results/` (e.g. `results/baselines_report.html`,
`results/split_report.html`), with generated PNGs under `results/plots/` and `audio_quality.py`'s CSVs
under `results/audio_quality_reports/`. `results/` is gitignored — run `make <target>` and open the
HTML file in a browser. `load_dataset.py`'s own validation reports (`dataset_validation.html`,
`mix_pairing_validation.html`) still write to `src/` directly (unchanged, pre-existing behavior).

## Dataset

**Status: provisional, pending re-download.** Per TEEP2026_Sprint0_Review
(2026-08-18), `HLS_CMDS/` currently in this working tree was sourced from
[github.com/Torabiy/HLS-CMDS](https://github.com/Torabiy/HLS-CMDS), not the
Mendeley release the project charter (task S0-06) specifies. Two problems
with the GitHub copy, found via this project's own audit tooling:

- **Sample rate is 4000 Hz**, not the 22,050 Hz the HLS-CMDS descriptor
  paper and project charter both state — a 5.5× mismatch, likely indicating
  a downsampled convenience distribution rather than the released dataset.
- **109 of 145 `Mix.csv` rows are acoustically unrelated to their named
  heart/lung sources** — `mixed ≈ a·(heart + lung)` holds (to 16-bit
  quantization) for only 36 rows; see `src/load_dataset.py`'s
  `verify_additive_triplets()` (run `python3 load_dataset.py` from `src/`
  for a live count) and `TEEP2026_Sprint0_Review` for the full analysis.

**Action item (blocking):** re-download from Mendeley
(https://data.mendeley.com/datasets/8972jxbpmp/3) into a directory outside
this git repo, then fill in below:

- Version:
- Download date:
- Source URL: https://data.mendeley.com/datasets/8972jxbpmp/3
- Per-file SHA-256: (see `checksums.txt` once generated, or embed here)

Until this is done, treat every separation-quality number in `PROTOCOL.md`
and `report/report.pdf` as provisional.

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

python3 visualization/audio_plotter.py     # plots + saves waveforms for 3 illustrative recordings pulled from HS.csv/LS.csv
python3 visualization/audio_spectrogram.py # plots mel-spectrograms for the same 3 example files
python3 visualization/donut_chart.py       # standalone chart, no dataset files needed
python3 visualization/plot_per_class.py    # waveform + spectrogram grid, one representative file per class

python3 statistics/audio_quality.py        # per-class/per-location duration, sample rate, clipping stats
```
# Knowledge Base — Heart & Lung Sound Analysis

Reference papers for the heart/lung sound classification project.
Synced manually from Claude Project knowledge (no automatic sync — see Updating).

## Contents

| File | Topic | Used for |
|---|---|---|
| `papers/HLSCMDS_Dataset_Descriptor.pdf` | Heart and lung sounds dataset recorded from a clinical manikin with a digital stethoscope | Dataset structure, sampling rate, label schema |
| `papers/Cardiorespiratory_Separation_SSA.pdf` | Separating heart sounds from lung sounds via Singular Spectrum Analysis | Preprocessing / source separation stage |
| `papers/BSS_Eval_Performance_Metrics.pdf` | Definitions of SDR, SIR, SAR for blind source separation | Evaluation metrics for the separation stage |
| `papers/Respiratory_Disease_Classification_NMF_LogMel_CRNN.pdf` | NMF-enhanced log-Mel spectrograms feeding a CRNN | Model architecture & feature pipeline |
| `papers/Spectrotemporal_Heart_Sound_Clinical_Noise.pdf` | Spectro-temporal deep learning for heart sounds under clinical noise | Noise robustness |
| `papers/Edge_Portable_Lung_Sound_CNN.pdf` | CNN lung sound classifier running on edge hardware | Deployment, model size constraints |

## Where this goes in the repo

```
your-project/
│   ├── README.md
│   └── papers/*.pdf
├── src/
└── CLAUDE.md
```
```markdown
## Reference material
Foundational papers live in `knowledge-base/papers/`.
See `knowledge-base/README.md` for which paper maps to which part of the pipeline.
When you need specifics on feature extraction, evaluation metrics, or model
architecture, read the relevant file instead of inferring.
```

## A note on Git

These are binary PDFs totalling ~18MB. To keep the repo light:

- Use Git LFS: `git lfs track "knowledge-base/papers/*.pdf"`
- Or add `knowledge-base/papers/` to `.gitignore` and commit only this README
  alongside links to each paper's original source.

## Updating

Claude Project knowledge does not sync back and forth with your repo. When you add
new documents to the Project, download them here manually and update the table above.