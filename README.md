# TEEP_Internship — heart/lung sound separation, and what it does to a classifier

Code, data-handling and reports for the TEEP 2026 internship on the HLS-CMDS
heart-and-lung-sound dataset: six separation baselines under a leakage-safe
evaluation harness, a controlled SDR sweep, two heart-sound classifiers run on
the separated audio, a microcontroller port of Baseline 1, and the dataset audit
that found only 36 of the 145 published mixtures to be additive.

The work is released as **two repositories** with one shared history:

| Repository | Holds | Produces |
|---|---|---|
| `TEEP_Internship` (this one) | `src/`, `firmware/`, `report/`, this README | the internship report (`report/report.pdf`), the dataset-audit comment, and every `results/` table |
| `teep-spike` (Satya Adhiyaksa; URL to be added at publication) | everything above at commit `f8e11a3`, plus `spike/` and `paper/` | the SPL letter (`paper/paper.pdf`): its Tables I–II and Figure 1 |

`teep-spike` was forked from this repository's working tree on 14 Sep 2026
(its commit `f8e11a3` is byte-identical to this repository's `c5e7d97`) and has
carried the band-limiting study since; the letter's numbers come from
`spike/`, which imports this repository's `src/` unchanged. Nothing in
`spike/` is duplicated here. Section "Reproducing the SPL letter" below says
which script produces which cell; it applies to a checkout of `teep-spike`.

## Setup from a clean checkout

Tested on Linux 6.8 / Python 3.12.4 with an NVIDIA GPU (driver ≥ 530); the
classifiers train on CUDA if present and fall back to CPU. Five steps:

```sh
git clone https://github.com/SouthsideProgramer/TEEP_Internship.git teep && cd teep   # or the teep-spike URL
make install                     # python3 -m venv .venv && pip install -r requirements.txt (CUDA wheels, ~3 GB)
# download HLS-CMDS v3 from https://data.mendeley.com/datasets/8972jxbpmp/3 (login required)
# and save the archive as ./HLS-CMDS.zip -- it is CC BY 4.0 but not redistributed here
make dataset                     # unpack into HLS_CMDS/{HS,LS,Mix}/ and verify the six release checksums
make test                        # ~4 min: 191 tests -- metrics, split, loader, synthetic-mix, EVMD, Conv-TasNet, the review scripts
```

`make dataset` refuses to run without the archive, and fails loudly if any of
the six top-level artifacts differs from the checksums in "Dataset" below —
the numbers in every report were measured on exactly those bytes. Every other
target assumes these five steps have been run. `make help` lists them all.

### The clean-checkout proof

The SPL letter ends with "Code, commit history and every tested configuration
are released." The test of that sentence, run before publication and recorded
in `report/DAILY_LOG.md`, is: clone `teep-spike` into an empty directory, do
the five setup steps, and run the four read-only review checks that
re-derive the letter's data guarantees from the checkout alone:

```sh
make spike-checks
# == spike/check_heldout_distinct.py   -> PASS: no test recording is a copy or scaled copy of a training recording
# == spike/check_lung_overlap.py       -> PASS (in the sense that the finding reproduces): train/test lung pools overlap
# == spike/check_labels_folds.py       -> PASS: IDs unique, labels as pre-registered, folds 0-4 x 10
# == spike/review_checks.py            -> per-condition median SDR table and the recording/type/Kish bootstrap table
# spike-checks: all four passed
```

Each script asserts against the values observed when the letter was written
(`OBSERVED_MAX = 0.374`, `OBSERVED = {...}`, the class counts), so a pass means
the checkout carries the same data, split, seeds and cached scores that the
letter was computed from. `review_checks.py` reads the released per-recording
scores (`spike/out/*_scores.csv`) and refits nothing, so all four together take about a
minute on a CPU. The first three write nothing. The fourth rewrites
`spike/out/review_bootstrap.csv` byte-identically and re-serialises
`spike/out/confirm_levels.csv`, so `git status` shows the latter modified:
the diff is float formatting in the last digit on rows the script does not
recompute (the parsed values are identical, max abs difference 0.0).

## Reproducing the SPL letter (`teep-spike`: `paper/paper.tex`)

All scripts run from the repository root as
`PYTHONPATH=src:spike .venv/bin/python spike/<script>.py`, and every one is
preceded in the commit history by the `.md` file that pre-registers it
(`HYPOTHESIS.md` → `CONFIRM.md` → `EXTEND.md` → `CIRCOR.md`). Seeds are fixed
in the scripts; the CNN and ResNet-18 numbers are five-seed means, and GPU
training is not bit-reproducible, so expect those cells to move by a few tenths
of a point on re-training. The SVM cells are deterministic.

| Letter | Cells | Script | Reads | Writes (released) |
|---|---|---|---|---|
| Table I, columns CNN and SVM; bracketed INT / SEP SDR and SI-SDR | all 8 rows | `spike/confirm.py` | HLS_CMDS, `src/` separators | `spike/out/confirm_summary.txt`, `confirm_scores.csv`, `confirm_levels.csv`; cache `spike/out/confirm_cache/` (1,625 waveforms, **not released**, rebuilt by this script on 12 worker processes) |
| Table I, bracketed BAND SDR / SI-SDR | row BAND | `spike/review_checks.py` | `confirm_levels.csv` | `confirm_levels.csv` (BAND rows merged) |
| Table I, column ResNet-18 | all 8 rows | `spike/resnet_arch3.py` | `confirm_cache/` | `resnet_summary.txt`, `resnet_scores.csv` |
| Table I, column SVM tuned; the "40 configurations … 25 / 13" sentence | all 8 rows | `spike/svm_tuned.py` | `confirm_cache/` | `svm_tuned_summary.txt`, `svm_tuned_scores.csv`, `svm_tuned_cv.csv`, `svm_tuned_spec_curve.csv` |
| Table II, HLS-CMDS rows R1 / R2 | CNN, SVM | `spike/confirm.py` | as above | `confirm_summary.txt` ("tests:" block) |
| Table II, HLS-CMDS rows R1 / R2 | ResNet-18, SVM tuned | `resnet_arch3.py`, `svm_tuned.py` | as above | `resnet_summary.txt`, `svm_tuned_summary.txt` |
| Table II, CirCor row; the CirCor paragraph of §Results | ResNet-18 | `spike/circor.py` | CirCor DigiScope 1.0.3 (see below), HLS_CMDS lungs | `circor_summary.txt`, `circor_counts.csv`; cache `circor_cache/` (not released) |
| §Results, the ten-source-type cluster intervals | CNN, ResNet-18, SVM tuned | `spike/review_checks.py` | `*_scores.csv` | `review_bootstrap.csv` |
| Figure 1 | — | `spike/fig_inversion.py` | `*_summary.txt`, `*_scores.csv` | `spike/out/fig_inversion.pdf` → `paper/fig_inversion.pdf` |
| The exploratory run disclosed in §Pre-registration (not in any table) | — | `spike/bandlimit.py` | HLS_CMDS | `spike/out/summary.txt`, `levels.csv`, `predictions.csv` |

Order from a clean checkout: `confirm.py` first (it builds the cache the two
extension scripts consume and refuse to run without — `assert_cache_params`
checks the stamp `spike/out/confirm_cache_params.json`), then
`resnet_arch3.py` and `svm_tuned.py` in either order, then `review_checks.py`
and `fig_inversion.py`. `circor.py` is independent of the others.

**CirCor.** `spike/circor.py` reads the CirCor DigiScope training set
(PhysioNet, ODC-By 1.0; `s3://physionet-open/circor-heart-sound/1.0.3/`) from
the path in its `CIRCOR` constant, which is a machine-specific absolute path in
the released script. To reproduce that row, download the training set, verify
it against PhysioNet's `SHA256SUMS.txt`, and point `CIRCOR` at it. The
recording labels, patient grouping and split are re-derived by the script
from `training_data.csv` (rule in `spike/CIRCOR.md`).

## Reproducing the internship report (`report/report.tex`)

Every table is regenerated by one `make` target that writes a self-contained
HTML report (and, where stated, a CSV) under `results/`; the LaTeX tables were
transcribed from those files, and `results/canonical_rows.csv` is the one
row-level file the classification tables agree with. `results/` is gitignored on purpose: it is
output, and every file in it is reachable from the targets below. Targets that
read another target's output are listed after it; run them in this order on a
clean checkout.

| `\label{tab:…}` | Content | Target | Output |
|---|---|---|---|
| `checksums` | provenance check | (none — recorded in "Dataset" below and verified by `make dataset`) | — |
| `perclass`, `perlocation` | audio-quality audit | `make stats` | `results/audio_quality_reports/*.csv` |
| `leakgroups`, `foldsizes` | content-hash leak groups, 5-fold split | `make split` | `results/split_report.html` |
| `additivesubset` | the 36-of-145 additive audit | `make validate` | `src/mix_pairing_validation.html`; relaxations: `make additive-audit-sensitivity` → `results/additive_audit_sensitivity_report.html` |
| `synthtests` | BSS Eval synthetic ground-truth checks | `make test` (`src/test/test_metrics.py`) | pytest output |
| `psd` | PSD audit behind the band choice | `make baseline1` | `results/baselines_report.html` |
| `baselinefull` | Baselines 1–4, full 145 rows | `make baselines` (~20 min) | `results/baselines_report.html` |
| `baselinevalid`, `pooledvalid`, `sixmethodnative` | 36 additive rows, six methods | `make baselines`, `make baseline5`, `make baseline6`; consolidated by `make canonical-rows` | `results/canonical_rows.csv`, `canonical_summary.csv` |
| `sixmethodsynth` | synthetic substrate, six methods | `make synthetic-set`, then `make first-sdr-table` (1–2 h) | `results/first_sdr_sir_sar_table.html` |
| `b6diag` | Conv-TasNet-lite training diagnostics | `make baseline6` | `results/baseline6_report.html` |
| `conditionAfold` | Condition A, MFCC+SVM per fold | `make heart-classifier` | `results/heart_classifier_report.html` |
| `conditionb` | Condition B, both architectures, 36 rows | `make sdr-sweep` (slow, cached), then `make condition-b`, `make condition-b-stats` | `results/condition_b_report.html`, `condition_b_stats_report.html` |
| `knee`, `kneecompare` | accuracy-vs-SDR crossings | `make sdr-accuracy-curve`, then `make sdr-knee-point` | `results/sdr_knee_point_report.html`, `sdr_accuracy_curve*.csv` |
| `arch2` | Architecture 2 vs 1, Condition A | `make heart-classifier-cnn` | `results/heart_classifier_cnn_report.html` |
| `kneeboot` | knee-point bootstrap | `make sdr-knee-bootstrap` | `results/sdr_knee_bootstrap_report.html` |
| `macs` | MACs and parameters | `make compute-cost` | `results/compute_cost_report.html` |
| `latency`, `latencyenv` | desktop latency, environment | `make latency` | `results/latency_report.html`, `latency.csv`, `latency_environment.csv` |
| `pareto` | SDR-vs-compute plane | `make sdr-compute-plane` (needs `canonical-rows` and `latency`) | `results/sdr_compute_plane_report.html` |
| `causal` | causal vs zero-phase filtering | `make firmware-causal-check` | `results/firmware_causal_vs_zerophase.html` |
| `ondevice` | Nano 33 BLE Sense, three clips | `make firmware-on-device PORT=…` — **needs the board** | `results/firmware_on_device_report.html` |
| `servervsboard` | server vs board on identical work | `make server-vs-board`, `make plot-server-vs-board` (read the on-device report) | `results/plots/server_vs_board.pdf` |
| `mcufeas` | RAM feasibility of Baselines 2–6 on the MCU | `make mcu-feasibility` | `results/mcu_feasibility_report.html` |
| `related` | related-work table | (prose) | — |
| Figures: trajectory bands, SDR-α audit, compute plane | | `make trajectory-points`, `make sdr-alpha-audit`, `make sdr-compute-plane` | `results/plots/trajectories_bands.{png,pdf}`, `sdr_alpha_curves.png`, `sdr_compute_plane.png` |

The dataset-audit comment (`report/dataset_audit_comment.tex`) rests on the
`additivesubset` table: `make validate` prints the live 36/145 count, and
`make additive-audit-sensitivity` the relaxations it discusses.

### What a clean checkout cannot reproduce, and why

- **`ondevice`, and `servervsboard` downstream of it,** need an Arduino Nano 33
  BLE Sense on a serial port. The released `results/firmware_on_device_report.html`
  numbers are the ones in the report; `make firmware` (no board) still runs the
  whole host-side chain and diffs the C filter against scipy on golden vectors.
- **The CirCor row of the letter** needs the PhysioNet download described above.
- **GPU-trained cells** (Conv-TasNet-lite, the log-Mel CNN, ResNet-18) are
  seed-fixed but not bit-reproducible across GPUs or CUDA builds.
- **`papers/`** (the reference PDFs) is not redistributed; the table at the end
  of this file gives each paper's DOI.
- **Runtimes** quoted above are for a Core i9-9900K / 65.7 GB / RTX 3060; the
  `make help` text gives per-target estimates.

## Layout

```
├── HLS_CMDS/            # dataset, laid out by `make dataset` (gitignored; CC BY 4.0, from Mendeley)
├── results/             # generated HTML reports, CSVs and plots (gitignored; regenerated by make targets)
├── src/                 # analysis code; every script is a `make` target
│   ├── load_dataset.py      # loads HS/LS/Mix CSVs, resolves + validates audio paths, additive-triplet audit
│   ├── metrics.py           # BSS Eval (SDR/SIR/SAR) for heart/lung separation
│   ├── split.py             # leakage-safe, content-hash leak groups, triplet-level fold assignment
│   ├── synthetic_mix.py     # S1-09/S1-10/S1-13 synthetic mixing set and its source-file-level split
│   ├── eval_harness.py      # k-fold cross-validation wiring split.py + metrics.py
│   ├── baselines.py         # Baseline 0 (raw mixture) + report-generation glue
│   ├── baseline/            # Baselines 1-5, one module each (bandpass, NMF x2, SSA, EVMD); common.py holds the bands
│   ├── convtasnet.py        # Baseline 6 (Conv-TasNet-lite)
│   ├── heart_classifier.py, heart_classifier_cnn.py   # Architecture 1 (MFCC+SVM) and 2 (log-Mel CNN)
│   ├── degradation.py, sdr_sweep.py, condition_b.py, sdr_accuracy_curve.py, sdr_knee_*.py   # the SDR sweep
│   ├── canonical_rows.py, condition_b_stats.py, trajectory_points.py, sdr_alpha_audit.py    # audit artifacts
│   ├── compute_cost.py, latency.py, sdr_compute_plane.py, energy_model.py, mcu_feasibility.py, server_vs_board.py
│   ├── test/                # pytest suite (`make test`)
│   ├── visualization/, statistics/   # figures and the audio-quality audit
│   └── code_description.md  # per-baseline notes: what is literal paper text vs. interpretation
├── firmware/            # PlatformIO project: Baseline 1 ported to MCU (Nano 33 BLE Sense, ESP32-S3); see firmware/README.md
├── report/              # report.tex/.pdf (EN), report_vi (VI), paper.tex (13 Sep draft), dataset_audit_comment.tex, DAILY_LOG.md
├── PROTOCOL.md          # the evaluation protocol and the reading of each reference paper against it
├── BACKLOG.md           # what was built when, and what is open
├── Makefile             # every target; `make help`
└── requirements.txt     # pinned to the environment the results were produced in
```

`src/visualization/*.py` and `src/statistics/audio_quality.py` resolve dataset
paths relative to their own location, so `src/` and `HLS_CMDS/` must keep their
relative positions. Every script prints only progress lines; its results are
the HTML/CSV files under `results/`.

`firmware/` is a self-contained PlatformIO project (`make firmware` runs the
whole server-side chain — design, build, tests — with no board required). It
imports the heart/lung bands from `src/baseline/common.py` so the firmware
cannot drift from Baseline 1, and it filters **causally** rather than with the
baseline's zero-phase `sosfiltfilt` — see `firmware/README.md` for why and what
that costs. The measurement firmware has **no microphone**: it replays real
HLS-CMDS mixtures compiled into flash, so the board filters the same int16
samples the Python baseline filters, and its output is diffable against scipy
sample by sample.

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

## Environment the results were produced in

Python 3.12.4 in `.venv` (`make install`), package versions pinned in
`requirements.txt`; Core i9-9900K (16 threads), 65.7 GB RAM, RTX 3060
(driver 610.57), Linux 6.8. LaTeX for `report/` is not part of the
environment — the PDFs are committed beside their sources.

## Reference papers

`papers/` holds the PDFs the protocol was read against and is not
redistributed. `PROTOCOL.md` records what each was checked for.

| Key | Paper |
|---|---|
| `hlscmds` | Torabi, Shirani & Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," *IEEE Data Descriptions* 2, 2025. doi:10.1109/IEEEDATA.2025.3566012 |
| `aidriven` | Torabi, "AI-Driven Cardiorespiratory Signal Processing: Separation, Clustering, and Anomaly Detection," Ph.D. dissertation, McMaster University, 2025 |
| `ssa` | Han & Quan, "Cardiorespiratory Sound Separation Using Singular Spectrum Analysis," *IEEE ICSPS*, 2025. doi:10.1109/ICSPS66615.2025.11347745 |
| `nmfcrnn` | Han, Quan, Matuszewski & Corbett, "Respiratory Disease Classification Using NMF-Enhanced Log-Mel Spectrograms and Convolutional Recurrent Neural Networks," *Sensors* 26(13), 2026. doi:10.3390/s26134268 |
| `spectrotemporal` | Yaqub et al., *CMES*, 2025. doi:10.32604/cmes.2025.071571 |
| `edgelung` | Puneet, Shankar, Koluguri & Srivastava, "Edge-Enabled Portable Classifier for Lung Sounds Using Convolutional Neural Networks," *IEEE BioCAS*, 2025. doi:10.1109/BioCAS67066.2025.00016 |
| BSS Eval | Vincent, Gribonval & Févotte, "Performance Measurement in Blind Audio Source Separation," *IEEE TASLP* 14(4), 2006 |
