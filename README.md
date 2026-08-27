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

**HLS-CMDS**, Mendeley v3 —
[data.mendeley.com/datasets/8972jxbpmp/3](https://data.mendeley.com/datasets/8972jxbpmp/3).
Descriptor: Torabi, Shirani & Reilly, *IEEE Data Descriptions*, 2025,
[doi:10.1109/IEEEDATA.2025.3566012](https://doi.org/10.1109/IEEEDATA.2025.3566012).
Licensed CC BY 4.0 — citing the descriptor paper is mandatory.

- **Version:** v3 — 535 recordings (50 isolated heart, 50 isolated lung, 145 mixed)
- **Retrieved:** 2026-08-19
- **Source URL:** https://data.mendeley.com/datasets/8972jxbpmp/3
- **Working copy:** `HLS_CMDS/`

**Provenance is verified.** This copy was originally obtained from
[github.com/Torabiy/HLS-CMDS](https://github.com/Torabiy/HLS-CMDS), which raised
the possibility that it was a downsampled convenience mirror rather than the
release the charter specifies. That possibility was tested on 2026-08-19 by
re-downloading from Mendeley and checksumming independently of this project's
own tooling: all six top-level artifacts are byte-identical. The mirror is
faithful, and every number produced here is a measurement on HLS-CMDS as
published.

```
$ md5sum HS.csv HS.zip LS.csv LS.zip Mix.csv Mix.zip
b61a866d2e8d70ae2910e4c049327d1e  HS.csv
7d678aff168184c13d8893b3bef4f4ba  HS.zip
fa40494d1dcbcf173ee5e9583b3faf82  LS.csv
d56a0d8ce5941342f33bfa9b04ef2bd9  LS.zip
377b55ea6361ff7c48994cddaf5c6e6f  Mix.csv
f28243aa5394324c58f90a466d60c2f4  Mix.zip

$ sha256sum HS.csv HS.zip LS.csv LS.zip Mix.csv Mix.zip
46d5dc3fc3d96c122620bafc68c34652876842b5abbde4aacd5a778d621a5a15  HS.csv
aa2c80a1430b2d105b49071e8cc72b9da3014df85e8c60be75683155c6f431fe  HS.zip
2f5ba55a7d0d3ded2edaae003eccc3164e3d860437836cdc998c9a4b928d8bd0  LS.csv
ac3c9df63518a5aa0431ec7bb50531f63f930e403a3d858445545592ef309811  LS.zip
c021907ac4a8775900e4a60cd47e8caac4d6ec383466d018c98bb3d4322433a1  Mix.csv
c0c3eb1a36ed20c1d323bac8390dcc4ddb6e46052ce9e70242bb70da57f7fb17  Mix.zip
```

Two properties of the release follow, both of which were initially mistaken
for symptoms of a bad mirror:

- **Sample rate is 4000 Hz** — mono, 16-bit, 15.0 s (60,000 frames), all 535
  files without exception. The descriptor paper states 22,050 Hz. Since this
  copy *is* the release, the discrepancy belongs to the descriptor or to a
  transcription downstream of it, not to the distribution. Confirming which,
  against the descriptor's own text and with a page reference, is open (task
  S1-14). It is not a problem for the science — heart energy sits at
  20–200 Hz and lung at 100–1000 Hz, comfortably under a 2000 Hz Nyquist. It
  *is* a problem for tooling: pretrained Conv-TasNet/Sepformer checkpoints
  expect 8–16 kHz, which is why Baseline 6 trains from scratch at the native
  rate.
- **Only 36 of 145 `Mix.csv` rows are genuinely paired** — `mixed ≈ a·(heart +
  lung)` holds, to 16-bit quantization, for M0087 and M0111–M0145 and for
  nothing else. Re-downloading cannot rescue the other 109, because the
  re-download is the same bytes. See `verify_additive_triplets()` in
  `src/load_dataset.py` (run `python3 load_dataset.py` from `src/` for a live
  count), and `report/dataset_audit_comment.tex` for the write-up.

Consequence: the synthetic mixing protocol (`src/synthetic_mix.py`) is the
primary evaluation substrate, and the 36 native additive rows are used to
validate that protocol rather than as the main evaluation set.

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