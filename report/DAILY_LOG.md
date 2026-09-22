# Daily Log

Chronological, one entry per work session — what got done, in plain
factual bullets. Distinct from `BACKLOG.md` (current done/open task state,
reorganized as work supersedes it) and `PROTOCOL.md` (the research memo
itself). This file doesn't get rewritten as things change; it's the
append-only record of *when* something happened. Bold a bullet's key
number/finding, backtick file/function names, note clock times where known
(from file mtimes / commit timestamps) since sessions here often span
midnight relative to when work is committed.

## 2026-08-13/14 — Setup

- Initial commit: environment/project structure (`62bb08b`, `195608f`).
- Branch merge (`00915e0`).
- (Reconstructed from commit subjects only — no deeper detail recorded for
  these two.)

## 2026-08-16 — First dataset scripts

- `parse_datagram` (`5fcca6a`, 10:45).
- `validation` (`284676e`, 13:17).
- (Reconstructed from commit subjects only.)

## 2026-08-17 — Metrics, split, first baseline (commit `ba0c453`, 17:53)

- `src/metrics.py` — BSS Eval (SDR/SIR/SAR) wrapper around
  `mir_eval.separation.bss_eval_sources`; `src/test_metrics.py` (10 tests).
- `src/split.py` — leakage-safe, triplet-level fold split (content-hash
  based); `src/test_split.py` (15 tests).
- `src/eval_harness.py` — `cross_validate()` wiring split + metrics.
- Layout refactor: `src/visualization/`, `src/statistics/` split out.
- `PROTOCOL.md` and `BACKLOG.md` started.
- `report/report.pdf` produced 17:48, just before the commit.

## 2026-08-18 — Sprint 0 review, Baseline 1+2, dataset finding (committed next morning as `ef55168`, 2026-08-19 08:07)

- `HLS_CMDS/` created 09:53 — **still the GitHub-sourced copy**
  (github.com/Torabiy/HLS-CMDS), *not* a Mendeley re-download; confirmed
  4000 Hz sample rate (Mendeley release is 22,050 Hz per the descriptor
  paper). The Mendeley re-download remains an **open, blocking item** —
  attempted programmatically, the file-content endpoints sit behind an
  authenticated JS-driven flow plain HTTP tooling here can't complete; see
  `PROTOCOL.md` §8.
- `verify_additive_triplets()` added to `load_dataset.py` (+
  `src/test_load_dataset.py`) — least-squares gain fit per row. **Found
  only 36/145 Mix.csv rows are genuinely additive** (`mixed ≈
  a·(heart+lung)`); the other 109 are acoustically unrelated to their
  named sources.
- `src/baselines.py` started: Baseline 1 (bandpass) and Baseline 2
  (supervised NMF, ref S3-02), both run on the full 145 rows.
- `summarize_pooled` added to `eval_harness.py` — row-level median/mean/std
  printed alongside the fold-level figures, explicitly labeled (pooled std
  is much wider than fold-level std — an order of magnitude, per
  `PROTOCOL.md` §8's statistics note).
- `Makefile` added.
- `README.md` given a real Dataset section documenting the GitHub-vs-Mendeley
  discrepancy; every separation number in `report/` and `PROTOCOL.md`
  marked provisional pending the Mendeley re-run.
- Test suite grew with `test_load_dataset.py`'s additions.
- TEEP2026_Sprint0_Review (Satya Adhiyaksa) received and filed at
  `report/TEEP2026_Sprint0_Review_Thang.pdf`, 14:54 — findings folded into
  `PROTOCOL.md`/`BACKLOG.md` the same session (dataset provenance, dataset
  discrepancy, citations, statistics note).

## 2026-08-19 — Housekeeping (`863231a`, 08:08)

- `.gitignore` tweak only; no other changes this session.

## 2026-08-20 — Baseline 3, papers/ cross-check, Baseline 4 (commit `84a7834`, 17:10)

- **Baseline 3** (`make_standard_nmf_baseline`) — standard NMF, no learned
  dictionary, the ablation against Baseline 2/S3-02. Ref kept as **S3-03**
  though pulled forward from Sprint 3 into Sprint 2 (Baselines 1/2 landed
  before kickoff). 5-fold on the 36-row additive-only subset: heart
  **+3.44±2.02 dB**, lung **+2.22±3.91 dB** — matches or beats Baseline 2
  there.
- `papers/` populated with 6 primary sources. Cross-check against
  Baseline 2's claimed source (`[nmfcrnn]`, Han et al., *Sensors* 2026)
  found and fixed two real gaps: a missing pre-NMF denoise bandpass
  (50–1800 Hz, paper §3.2.1) and an unverified STFT hop (128 → corrected to
  256, §3.2.3). Resolved the `[nmfcrnn]` dataset-discrepancy open item:
  HLS-CMDS is the auxiliary dictionary corpus, ICBHI+Fraiwan is the
  paper's own classification dataset — not a conflict. `[ssa]` paper's
  Table I confirmed directly (Butterworth baseline 5.7 dB cardiac /
  **-5.7 dB respiratory**). One BSS-Eval-paper misattribution in
  `PROTOCOL.md` corrected. Post-fix, Baseline 2 re-run on the additive-only
  subset for the first time: heart +2.69±2.96 dB, lung +0.62±3.14 dB —
  Baseline 3 (no dictionary) now edges it out on both sources.
- **Baseline 4** (`fit_ssa_baseline` / `mssa_separate`) — two-stage SSA
  reproducing Han & Quan, ICSPS 2025 (`papers/Cardiorespiratory_..._SSA.pdf`).
  Hyperparameters straight from the paper: window L=50, 250 Hz
  cardiac/respiratory split, 2% eigenvalue threshold, 50% cross-correlation
  threshold. Confirmed the 250 Hz split is an absolute physiological
  frequency (not sample-rate-normalized), so it's unchanged at this
  project's 4 kHz/2 kHz-Nyquist data. Built a synthetic 10×5-pair test set
  matching the paper's own evaluation recipe (+2% RMS Gaussian noise) for
  a like-for-like comparison against their Table I: respiratory SDR
  **6.81 dB** landed close to their 5.3 dB, but cardiac SDR (**1.24 vs.
  26.4 dB**) and both correlations came in far short — flagged as an open,
  unresolved gap in `PROTOCOL.md` §8, not papered over. On this project's
  own real Mix.csv (additive-only 36 rows), Baseline 4 came out **best of
  all four baselines** (heart +5.17±4.05 dB, lung +5.32±3.20 dB) — opposite
  direction from the synthetic-set gap, suggesting the gap is specific to
  the synthetic-set reproduction rather than a general implementation
  break.

## 2026-08-21/23 — No activity recorded

- No commits, no file changes on disk in this range. (The 2026-08-22
  session in `BACKLOG.md` — `report_utils.py`, `results/` HTML
  consolidation, Baseline 0, the desync test on the 109 rows — is dated
  there but its files first appear in the 2026-08-24 commit below.)

## 2026-08-24 — Synthetic set, Baseline 5 (EVMD), first SDR table, citation audit (commit `7b93d41`, 19:23)

- `src/synthetic_mix.py` (S1-09/S1-10/S1-13) — synthetic mixing substrate,
  `mixed = a*(heart+lung) + noise`, gain calibrated from the 36 native
  additive rows, `SNR_SWEEP_DB = (-5, 15, 35)` (cut from 5 levels to 3 after
  measuring per-row cost), source-file-level split (`assign_source_folds`).
  **1500 rows** (500 leakage-safe pairs × 3 SNRs). S1-10 validation: the
  construction reproduces **36/36** native additive rows via
  `verify_additive_triplets()` itself. `test_synthetic_mix.py`, 17 tests.
- **Baseline 5: EVMD** (S4-01, pulled forward from Sprint 4) — VMD written
  directly (frequency-domain ADMM), K=2..10 sweep per the BioCAS 2025 paper.
  **~15 s/mixture**, so its synthetic column runs on a disclosed subsample.
  K-selection checked on 9 mixtures: only **2/9 converged**, the rest fell
  to the K=10 ceiling — stated plainly rather than implied clean.
  `test_baselines.py`, 6 tests. Also corrected PROTOCOL.md: `[edgelung]`'s
  hardware is a PYNQ-ZU FPGA (~15 W), not an MCU.
- `src/first_sdr_table.py` → `results/first_sdr_sir_sar_table.html`
  (16:45): all 5 baselines × {synthetic, native-additive, Han & Quan Table I},
  every cell with mean / 95% CI / n. `src/baseline12_synthetic_report.py`
  (17:16): on the synthetic set Baseline 1 heart 2.10±0.43 / lung
  −1.24±0.44 dB, Baseline 2 −0.70±0.40 / −3.73±0.40 dB.
- Additive-mixture forensics on raw int16: residual **mean 0.48 LSB**
  (~1.65× the single-rounding floor → ≥2 rounding steps), per-segment gain
  CV 0.0074% — the 36 rows are computed sums, not recordings.
- Report/citation audit: 4 of `report.tex`'s "[Author(s) needed]"
  placeholders filled from the Reading List; Yaqub et al.
  (`[spectrotemporal]`) read in full for the first time — it validates
  **on HLS-CMDS itself** and its 89.0%→41.0% collapse (Table 9, n=5365)
  comes from bandpass-separated heart sounds, i.e. this project's Baseline
  1 method class. `report.tex` related-work row, Introduction, and
  PROTOCOL.md §2/§3/§8 rewritten. No LaTeX toolchain here — PDF not rebuilt.
- `Makefile`: `synthetic-set`, `baseline5`, `first-sdr-table`,
  `baseline12-synthetic`. `src/code_description.md` started (412 lines).
- Session ran past midnight: Conv-TasNet and NeoSSNet PDFs landed in
  `papers/` at 00:31 on the 25th, the start of the Baseline 6 work.

## 2026-08-25 — Baseline 6 (Conv-TasNet-lite), `baselines.py` split, tests → `src/test/` (commit `9750efc`, 12:55)

- All standing reports regenerated 10:15–10:27 (`split_report`,
  `eval_harness_smoke_test`, `audio_quality_report`, the four plot reports,
  `baselines_report`).
- **Baseline 6** (`src/convtasnet.py`, `baseline6_report.py`) — first neural
  baseline. Sample-rate decision made first: train from scratch at native
  4000 Hz, no resampling to public 8–16 kHz checkpoints. Conv-TasNet
  encoder/TCN/decoder sized down to **~325K params**, fixed source order
  (no PIT), per-fold training on the leakage-safe pool with on-the-fly
  synthetic mixing, AdamW + early stopping (`MAX_EPOCHS=25`). Full 5-fold
  on both substrates: **568.2 s** on CUDA. Result (10:42): synthetic heart
  **3.16±0.40** / lung **0.37±0.35** dB (first positive lung SDR on the
  synthetic set); native heart 5.15±2.74 / lung 2.14±2.14 dB. Disclosed as
  a first working config, no sweep. `test_convtasnet.py`, 12 tests.
- Refactor: `src/baselines.py` (~790 lines) split into `src/baseline/`
  (`common.py`, `baseline1.py`..`baseline5.py`; Baseline 3 imports its NMF
  machinery from Baseline 2); `baselines.py` keeps only Baseline 0 + report
  glue. All tests moved to `src/test/` with a `conftest.py`;
  `test_baselines.py` → `test_baseline5.py`. `make test` now discovers
  `test/` (which quietly fixes two files the old hand-list had skipped).
  **66 tests pass**; every top-level script import-checked and smoke-run.

## 2026-08-26 — Dataset audit comment drafted

- `report/dataset_audit_comment.tex` carries "Draft — 2026-08-26" as its own
  date line (the file's mtime is 09-09, when S1-14 closed it). No commit,
  nothing else on disk for this day. Reconstructed from that date line only.

## 2026-08-27 — The whole C2 pipeline in one sitting (committed as `5ed5f2c`, 2026-08-28 00:26)

Long session; report mtimes run 21:02 (degradation) → 22:59 (compute cost)
→ 23:21 / 23:46 (CNN and latency tests), `PROTOCOL.md` at 23:32, commit
just after midnight. Machine load climbed ~20 → 57 → 88 on 16 cores over
the session, mostly other users.

- `src/heart_classifier.py` — **Condition A** (S2-03, pulled forward from
  S5). Decided grouped classes, exactly Yaqub et al.'s four (Normal 9 /
  Murmur 24 / Extra Sound 7 / Rhythm Disorder 10), confirmed from the
  primary source. `split.assign_hs_folds()` extends the leak-group split to
  HS.csv. One architecture: 13 MFCC mean+std → RBF SVM, sklearn defaults.
  **58.0% ± 7.3%**, macro-F1 0.43; recall Normal 44 / Murmur 88 / Extra 29
  / Rhythm 20%. Confusion matrices added the same session: both minority
  classes' errors pull toward **Murmur**. 18 tests.
- `src/degradation.py` (S6-01, pulled forward twice: S6→S5→S2) —
  `degraded = g + alpha*(s − g)`. The x-axis decision: **SDR, never SNR**
  (SNR is undefined for Yaqub's point). Validated on real audio: SDR(α)
  monotone, α=1 reproduces the measured SDR. Uniform α grid is badly uneven
  (~262 dB at α=0 → 10.6 dB at α=0.1) → `find_alpha_for_target_sdr()`
  (bisection). 11 tests.
- `src/sdr_sweep.py` (S6-02) — 6 baselines × 36 rows × 2 sources × 7
  targets (25…−5 dB), separated audio cached under `results/sdr_sweep_cache/`.
  Batched root-finder ~1.8× faster. Generation launched; Baseline 1 done in
  ~26 min (cache 22:07), rest running overnight. Launching it alongside a
  full `pytest` starved both — killed the test run.
- `src/condition_b.py` — **weight-sharing decision: same trained weights**
  per fold (Yaqub's Exp. 3 model *is* their Exp. 4 model). Found and fixed a
  fold-basis mismatch between the 145-row and 36-row substrates. First
  Baseline-1-only result: isolated **50.0% → separated 41.7%**, p=0.57.
- `src/sdr_accuracy_curve.py` (S6-03) — checkpoint/resume every 25 rows,
  fold-consistency assertion; 10 tests. `src/sdr_knee_point.py` (S6-04) —
  knee = where a baseline's curve crosses no-separation accuracy
  (`crossed` / `always_above` / `always_below` / `noisy_crossing`);
  `condition_b.evaluate_no_separation()` and
  `sdr_sweep.build_provenance_from_cache()` added so partial caches can be
  plotted. 7 tests.
- `src/compute_cost.py` — MACs/params per method: bandpass 2.4M <
  classifier 8.1M < supervised NMF 219M < standard NMF 1.16G < MSSA 1.20G <
  Conv-TasNet-lite 1.96G (325,465 params, cross-checks the 08-25 figure) <
  EVMD 5.22G — **~2,175× spread**. 14 tests.
- `src/heart_classifier_cnn.py` (S7-06) — Architecture 2, log-mel + shallow
  CNN; `backend` parameter threaded through Condition B / curve / knee.
  First config: **30.0% ± 8.8%**, collapses to Normal (a *different*
  attractor than the SVM's Murmur). Knee report auto-caveats when Arch 2
  isolated accuracy < 40%.
- `src/latency.py` — wall-clock/CPU per method under **load 74.6**:
  Conv-TasNet 23.9 ms < bandpass 36.0 < classifier 63.0 < sup. NMF 5,200 <
  MSSA 12,910 < std. NMF 21,036 < EVMD 90,075 ms. Finding at the time: MACs
  don't predict latency. (Retracted 2026-09-13 — the ordering of the two
  fastest was a contention artifact.)

## 2026-08-28 — Sweep finishes, `report.tex` expanded, SDR-vs-compute plane (commit `5ed5f2c` 00:26; the day's own work committed 08-31 in `70bc80a`)

- 00:26 `5ed5f2c` "updated": every 08-27 module plus 12 test files, 645
  BACKLOG lines, `code_description.md` (584 lines); also **removed
  `HLS_CMDS/` from git tracking** (1117 files changed — the dataset is
  untracked from here on).
- **S6-02 generation completed for all six baselines**: cache dirs finish
  B2 07:52, B3 12:26, B4 12:33, B5 12:48, B6 12:58 →
  `results/sdr_sweep_report.html` 12:58, 3024 provenance rows. (This was
  not noticed as finished until 2026-09-13 — the report's "still pending"
  bullet stayed stale for two weeks.)
- `report/report.tex` **1433 → 2431 lines**: new §2 "Background and
  Preliminaries" (student-level primer on BSS, BSS Eval, NMF, SSA, VMD,
  Conv-TasNet, CV/leakage, classification metrics, MACs/Pareto) and new §9
  "Downstream Classification" + §10 "Computational Cost". Brace /
  `\begin`-`\end` / label-ref balance checked programmatically; no toolchain
  to build the PDF.
- `src/sdr_compute_plane.py` — Baseline 6 native SDR re-measured and
  **reproduces 5.15±2.74 dB exactly**; hardcoded into `SDR_HEART_DB`. MACs
  plane: bandpass Pareto-dominates everything. Latency plane: bandpass and
  Conv-TasNet-lite both non-dominated (retracted 09-13). `code_description.md`
  section written 17:19.

## 2026-08-29 — Vietnamese report started

- `report/report_vi.tex` created (BACKLOG's 09-13 entry: the embedded
  section "had been missing since 2026-08-29"). `report/` is untracked, so
  no commit; the file's current mtime is 09-13. Reconstructed from that
  note only.

## 2026-08-31 — Refactor & clean (commit `70bc80a`, 16:29)

- 45 files, +323/−388: inline explanatory comments and section banners
  stripped out of every `src/` module and test file; the explanations now
  live only in `src/code_description.md` (+74 lines, incl. a new
  `sdr_compute_plane.py` section). No behaviour change. The 08-28 BACKLOG
  entry landed in this commit.

## 2026-09-01/06 — No project activity

- Only `report/TEEP2026_Sprint0_Review_Thang.pdf` re-saved (09-02, 13:00).
  Nothing else on disk, no commits.

## 2026-09-07 — Baseline 1 ported to microcontrollers (commits `d10997f` 17:50, `7067b37` 17:53, `7c722ec` 17:56)


## 2026-09-09 — On silicon, the FPU bug, server-vs-board, energy (10 commits, 13:00–17:24)

- First hardware runs on the Nano 33 (`capture.bin` 13:04): self-test
  **5.3e-8 heart / 1.5e-8 lung**; bench SDR matches scipy to the last
  printed digit; **62.4 µs/sample, 4.0× real time**.
- 13:00 `53cd797` — `embed_clips.py` had embedded M0001–M0003 (non-additive),
  so the bench's SDR column measured the dataset defect; now draws from the
  additive subset via `verify_additive_triplets()`, refuses non-additive
  `--ids` without `--any`.
- 13:08 `95d588d` — the accuracy tolerance was calibrated on the wrong
  amplitude regime (additive rows are peak-normalised, ~20× louder): board
  1.045e-5 vs a fixed 1e-5 threshold, while scipy's own float32 deviates by
  1.114e-5. Now scored against the measured per-clip float32 floor, 4×;
  real ratios 0.51–1.57, injected faults 14.3 / 15,691 / 296,503.
- 13:36 `report/dataset_audit_comment.tex` finalised — **S1-14 closed**: the
  descriptor states 22,050 Hz in Table I (p. 6) as an instrument spec and
  never states the files' rate. 13:37 `1ce3380` — `report.tex` §11
  "Embedded Deployment"; `latency.py`'s "no edge claim" line corrected.
- 13:53 `bfed61f` — ESP32-S3 two-port trap documented (board still not run).
- 14:17–14:57 — `src/mcu_feasibility.py` predicted ~400× real time against
  the measured 4.0×; the 100× gap led to the finding that **the
  Cortex-M4F's FPU was never enabled** (`87c1c35`): arduino-mbed appends
  `-mfloat-abi=soft` after the framework's `softfp`, zero VFP instructions
  in the disassembly; `build_flags` can't override it, `build_unflags`
  does. Re-measured (`capture_fpu.bin` 14:53, `42d3b9b`): **3.47 µs/sample,
  72× real time — 18× faster from a build flag**, numerics unchanged
  (6.5e-8 / 1.7e-8), flash −1,248 B. Feasibility answer: the discriminator
  is streamability, not MACs — only Baseline 2 streams (9.1 kB); B4/B5 are
  batch by construction (167× / 71× over RAM); B6 batch as written (9.2 MB
  activations); int8 rescues nothing.
- 15:04–16:29 — `src/server_vs_board.py` (`24b57b5`), corrected at 15:26
  (`c0acb82`: the server side had been timing filter *design*, not
  filtering), `plot_server_vs_board.py` (`31d418f`), pre-FPU figures swept
  from docs (`1526f59`). Result: i9-9900K 129 cycles/sample vs M4F 222 →
  **1.72× per-clock penalty**; almost all of the 97× gap is clock rate. The
  pre-fix row (3,997 cycles/sample) kept deliberately as the wrong
  conclusion it would have supported.
- 17:24 `6274357` — `src/energy_model.py`: server energy **measured** via
  RAPL — **14.3 mJ/clip marginal, 35.8 mJ full package**; board *not*
  measured (no current sense), so reported as a break-even: **69 mW /
  172 mW** — inside the range a datasheet guess would cover, hence not
  substituted.
- `report.tex` §10 gains "Why the other five baselines do not run here" and
  "A plausible measurement that was wrong by 18×"; 43 pp., compiles.

## 2026-09-10/12 — No activity recorded

- No commits or file changes. (The next session begins 00:34 on the 13th.)

## 2026-09-13 — Sweep finally consumed, level confound, review pass, paper revision (00:34 → 22:51; commit `c7ea625` 16:02; everything after uncommitted)

- 00:34–01:01 — `src/oracle_mask_ceiling.py` (+5 tests): IRM / Wiener /
  binary masks over all 36 rows. **IRM 11.5±1.9 heart / 9.2±1.7 lung dB,
  Wiener 12.4 / 10.6** vs raw 2.5 / −2.0 and bandpass 5.5 / 4.1 — refutes
  the old one-row `self_learning.md` note; Baseline 2 sits 7.8 / 9.7 dB
  under its own family's ceiling.
- Two "still open" report bullets found stale (the sweep had finished
  08-28; Baseline 2's hyperparameters were cross-checked 08-17). **First
  run of `condition-b` → `sdr-accuracy-curve` → `sdr-knee-point` on the
  full sweep**: isolated 50.0%, separated 33.3–41.7%, no delta significant;
  knees B1 17.0 / B2 7.4 / B3 15.0 / B4 14.2 / B5 18.1 / B6 21.7 dB, every
  real output 5–17 dB below its knee. B1/B4/B6 gave *identical*
  predictions.
- Architecture 2 was **underfitting** (train acc 23–40%): per-bin
  standardisation, mini-batch 8, 300 epochs, wd 1e-3 → 62.0% ± 15.7%.
  Cross-architecture verdict: knee values disagree (4.4–9.3 dB), shape
  agrees. `report.tex` / `report_vi.tex` updated (EN 49 pp., VI 56 pp.; VI
  gains the embedded Section 10).
- 16:02 `c7ea625` "plot" — oracle ceiling, CNN changes, and 20 PNGs under
  `src/plots/` committed.
- 16:23–16:57 — **the identical-predictions result was a defect**: HS.csv
  training audio is 0.0005–0.013 RMS, separated estimates 0.03–0.09 (20–25×
  louder); MFCC[0] put them 11–15 standardised units outside the training
  cloud, RBF ≈ 0, prediction a per-fold constant. Fix:
  `heart_classifier.level_normalize()` (unit RMS, both backends, train and
  inference). Re-run: Arch 1 **60.0±10.7%**, Arch 2 **62.0±3.9%**; Condition
  B isolated 52.8%, B1 30.6 / B2 41.7 / B3 27.8 / B4 36.1 / B5 27.8 / B6
  36.1; only B3 significant (p=0.010). Knees Arch 1 B1 17.8 / B3 15.0 / B4
  15.6 / B5 18.2 / B6 6.7, B2 always above. What survives both classifiers:
  the grouping (mask methods never fall below no-separation; strippers
  cross at 15–18 / 8–9 dB). `results/plots/` had been emptied — server-vs-
  board and compute-plane figures regenerated (16:33).
- Two review-flagged table discrepancies fixed: 13 → **14** singleton leak
  groups (08-17 transcription error); Condition B delta column was
  mean-of-fold-deltas next to pooled accuracies — both conventions now
  emitted and labelled.
- 18:42 `src/sdr_knee_bootstrap.py` (+2 tests): 2000 replicates, row and
  leak-group cluster; P(crossed) 0.15–0.98, tightest interval 12 dB; Holm =
  BH q = 0.061 for B3, nothing significant after correction; Kish N 17.5–36.
- 19:48–20:38 — `src/sdr_alpha_audit.py`: SDR(α) is a closed form; **7 of
  432 curves are non-monotone** (all Conv-TasNet heart rows, polarity dips
  to −6..−20 dB), so the bisection assumption was wrong there.
  `degradation.find_alphas_for_target_sdrs_grid` solves the roots exactly;
  sweep points scored with fixed heart/lung assignment (73 permuted points
  flagged). Provenance regenerated 21:11 (99 alphas moved > 0.01).
- 20:05 `src/additive_audit_sensitivity.py`: 36/145 under all seven
  alternative models (DC, polarity, lag, two gains, …); gap > 2000×.
- 20:06 `src/condition_b_stats.py`: leak-group cluster bootstrap, sign-flip
  permutation, McNemar; **delta-vs-raw never favours separation**; CNN on
  B1/B4/B5 identical (16.7%) now described as spectral OOD collapse, not a
  bug. 20:54 `src/canonical_rows.py` → one row-level file explaining every
  table mismatch the review found (fold-mean vs pooled, pre-hop-fix runs,
  separate GPU run).
- 21:11–21:17 — curve / knee / bootstrap re-run: Arch 1 B1 20.1 / B3 15.0 /
  B4 15.6 / B5 18.2 / B6 4.5, B2 always above; Arch 2 B1 9.3 / B4 9.4 / B5
  7.9. Wording: "knee" → "raw-mixture crossing", region **15–20 dB (SVM) /
  8–9 dB (CNN)**.
- 21:22 — `latency.py` re-measured **idle (load 0.6)**: B1 3.6 ms,
  classifier 3.7, B6 16.7 (GPU), B2 17.2, B3 61, B4 1,097, B5 16,812 ms.
  The loaded run had inflated CPU-bound methods 10–340× — **the "bandpass
  and Conv-TasNet jointly non-dominated" finding is retracted**; bandpass
  dominates both planes. Environment now logged to
  `results/latency_environment.csv`.
- 21:26–21:31 — `report.tex` 54 pp., `report_vi.tex` 61 pp.,
  `paper_inserts.tex` 6 pp. rebuilt; every changed number verified to
  appear in all three; `sdr_compute_plane.py` now reads from artifacts.
  `test_review_scripts.py`; **179 tests pass**.
- 22:21 `firmware/tools/embedded_feasibility.py` → `paper.tex` IV-F: board
  == causal reference to 3 decimals of SDR, SVM and CNN predictions agree
  3/3, SRAM 45,992 B (17.5%), 0.89 ms per 256-sample block (1.4% duty).
- 22:42 `src/trajectory_points.py` + `results/plots/trajectories_bands.{png,pdf}`:
  pointwise leak-group bootstrap bands; at −5/0 dB only 6–16 rows are
  attainable per baseline, bands ≥ 25 pts wide. 22:51 `report/paper.tex`
  major revision (12 pp.): bands figure, idle-machine plane, operational
  crossing definition, per-point table, inference paragraph, configuration
  and reproducibility tables. `BACKLOG.md` updated 22:44.

## 2026-09-14 — Test run only

- `pytest` executed 11:41 (`.pytest_cache` touched); no source or result
  file changed.

## 2026-09-15/16 — No activity recorded

- Working tree still carries the 09-13 review pass uncommitted: modified
  `BACKLOG.md`, `Makefile`, `condition_b.py`, `degradation.py`,
  `heart_classifier.py`, `heart_classifier_cnn.py`, `latency.py`,
  `sdr_compute_plane.py`, `sdr_sweep.py`, `test_degradation.py`; untracked
  `firmware/tools/embedded_feasibility.py`, `additive_audit_sensitivity.py`,
  `canonical_rows.py`, `condition_b_stats.py`, `sdr_alpha_audit.py`,
  `sdr_knee_bootstrap.py`, `trajectory_points.py`, `test_review_scripts.py`,
  `test_sdr_knee_bootstrap.py`. Open items as of 09-13: synthetic-substrate
  sweep + classifiers, Arch 2 second seed / nested selection, lung-side
  Condition A/B, per-condition standardised-distance report, Baseline 2
  port, ESP32-S3 hardware run.

## 2026-09-16 — Review pass committed (`c5e7d97` "C3", 15:22)

- Everything listed under 09-15/16 as uncommitted is in this commit. It is
  byte-identical to `f8e11a3` in Satya's `teep-spike`, which was rsynced from
  this tree on 09-14 12:09 and has carried the band-limiting spike since.

## 2026-09-17 — Audit comment: citation fix

- `report/dataset_audit_comment.tex` `\bibitem{nmfcrnn}`: author initials
  corrected to B. Han, W. Quan, B. Matuszewski, D. Corbett, checked against
  page 1 of the paper PDF (the last unapplied item of the 09-14 review).
  `report/dataset_audit_comment_17sep.pdf` (off-server build, 10:21) predates
  this fix and needs a rebuild; the server's texenv cannot build (no
  `share/tlpkg`, so no `pdflatex.fmt`).

## 2026-09-18 — Release readiness: repo cleanup, reproducibility README, clean-clone proof (Sprint 5 row, due 09-28)

Why now: the SPL letter's last sentence, "Code, commit history and every
tested configuration are released", is false until a public repository
exists, and the letter goes out ~10-05.

- `.gitignore`: `report/` was ignored wholesale, so the audit comment, both
  reports and this log were unreachable from any checkout (the S1-12
  leftover). Now only LaTeX intermediates are ignored; sources, Markdown and
  built PDFs are tracked. The stale 09-09 `dataset_audit_comment.pdf` is left
  untracked; `_17sep.pdf` is tracked.
- `requirements.txt`: three dependencies the results were produced with were
  undeclared — `scikit-learn==1.9.0` and `threadpoolctl==3.6.0`
  (`src/heart_classifier.py`; they arrive transitively via librosa, but
  unpinned: a fresh install today gets 1.9.1 / 3.7.0) and
  `torchvision==0.18.1+cu121` (`spike/resnet_arch3.py`, the ResNet-18 column
  of the letter's Table I; installed in Satya's env on 09-14 14:17, after
  both `.freeze-*.txt` snapshots, so no freeze recorded it). Python version
  in the header corrected 3.11 → 3.12.
- `Makefile`: `make dataset` (Mendeley archive → `HLS_CMDS/`, the six release
  checksums verified, inner zips unpacked; there was no route from the
  download to the layout `load_dataset.py` expects) and `make spike-checks`
  (the four review checks, for a `teep-spike` checkout).
- `README.md` rewritten: two-repository structure, five-step setup, the
  clean-checkout proof, table-by-table reproduction map for the letter
  (Tables I–II, Figure 1, the cluster-bootstrap paragraph) and for
  `report/report.tex` (every `\label{tab:…}` → `make` target → `results/`
  file), and what a checkout cannot reproduce (board, CirCor path, GPU
  nondeterminism, `papers/`). The conda `audio_env` instructions it replaced
  described an environment that no longer exists.
- **Clean-clone proof, run 1 (Satya's HEAD `ef9a09d` as-is).** `git clone
  /home/satya/teep-spike` into an empty scratch directory; `python3 -m venv`;
  `pip install -r requirements.txt` with the file as committed; the
  Mendeley archive laid out with the new `dataset` target (all six checksums
  OK; 50 HS, 50 LS, 435 Mix files). Then the four scripts, `PYTHONPATH=src:spike`:
  `check_heldout_distinct` PASS (max |corr| 0.374, median 0.181);
  `check_lung_overlap` PASS (shared 34 / 34 / 50, mean 3.26);
  `check_labels_folds` PASS (57 test, 50 train, folds 0–4 × 10);
  `review_checks` ran to completion, `review_bootstrap.csv` byte-identical to
  the released file, `confirm_levels.csv` re-serialised with identical
  parsed values (max abs diff 0.0; float-repr differences on the untouched
  INT rows only). Total under a minute on CPU. So the claim holds for the data
  guarantees today, with one caveat: sklearn/threadpoolctl came in at
  1.9.1/3.7.0, not the pinned versions.
- Not reproducible from a checkout, recorded in the README: the CirCor row
  (`spike/circor.py:21` hardcodes `/home/satya/data/circor`); `ondevice` and
  `servervsboard` (need the Nano 33 BLE Sense); GPU-trained cells bit-exact.
- Still open for the 09-28 row, not doable from this account: `teep-spike`
  has no push remote (`DISABLED-do-not-push`) and needs its own public GitHub
  repository under Satya's account, then a merge of this repository's `main`
  (`requirements.txt`, `Makefile`, `README.md`) so its checkout carries the
  pins and the targets; the `teep-spike` URL in the README is a placeholder
  until then. `TEEP_Internship` is still `PRIVATE` on GitHub — flipping it
  is a one-line `gh repo edit --visibility public`, held for explicit go-ahead.
- **Clean-clone proof, run 2 (the merge Satya's checkout will carry).** In the
  same scratch clone, `git pull` of this repository's `faec69c` merged clean;
  `make install` brought scikit-learn 1.9.0, threadpoolctl 3.6.0,
  torchvision 0.18.1+cu121 exactly; `make spike-checks`: all four passed,
  `git status` afterwards shows only `spike/out/confirm_levels.csv`
  re-serialised; `make test`: 191 passed in 3:57. The claim in the letter's
  last sentence is true of that checkout, on this machine, today.

## 2026-09-22 — Audit comment §5 quantified; CirCor robustness spike, and GPU nondeterminism

Two threads. The second was not planned at the start of the day.

### Audit comment: §5 turned from a conditional into a table

- `report/dataset_audit_comment.tex` §5 rewritten. The old text said "**if**
  either study's evaluation set includes rows outside the 36 …", which an
  editor can wave away. It now carries `tab:downstream`: per paper, which
  `Mix.csv` rows enter its evaluation, with a page or table number on every
  figure, and "not stated" recorded as the finding where a paper is silent.
- `[spectrotemporal]` (Yaqub): **all 145 rows, not stated in the text but
  reconstructed exactly**. Table 9 (p. 2528) scores 5,365 windows; 400 ms
  windows (p. 2514) give 37 whole windows per 15 s recording; 145 × 37 = 5365.
  The check that settles it is per-class — Table 9's supports
  (481 / 2109 / 1147 / 1628) equal 37 × `Mix.csv`'s own per-class row counts
  (13 / 57 / 31 / 44) for **every one of the four classes**. So
  **4,033 of the 5,365 windows (75.2%)** behind its 41.0% headline sit on
  rows the audit finds non-additive.
- `[edgelung]` (Puneet): **all 145, stated** (p. 23) — the six lung-class
  counts printed there (28/28/25/23/22/19) match `Mix.csv`'s lung-type
  distribution on every class. But **no separation metric is reported on
  them**; the 80.55% (p. 24) is HF_Lung V1 (p. 23). The old §5 wording ("use
  the listed heart/lung IDs as ground truth") was wrong about this.
- `[nmfcrnn]` (Han): **none scored** — "used exclusively for dictionary
  learning … not included in the … classification experiments" (§3.2.3, p. 7).
  Two "not stated"s: which isolated recordings the dictionaries were fit on,
  and the provenance of Figure 3's (p. 8) mixture + ground-truth pair. The old
  §5 claim that Han never touches `Mix.csv` overstated it by one figure.
- `[aidriven]` (Torabi): **none** — §4.2.3 (p. 52) re-pairs HLS-CMDS segments
  into 25,000 mixtures of its own; §4.1.3 (p. 41) uses "210 clinical manikin
  recordings" (not 145, HLS-CMDS unnamed — not stated); §5.5 (p. 65) and
  §6.2.2 (p. 72) use the 50+50 isolated recordings.
- Abstract corrected: it claimed the findings bear on ground truth used by
  "at least four published studies", which now contradicts the paper's own
  table. **Two** of the four evaluate on `Mix.csv`. PDF rebuilt 15:09, 5 pp.,
  clean. `pdflatex` is at `~/texlive/2026/bin/x86_64-linux`, not on `PATH` —
  the 09-17 note that the server cannot build is wrong; tectonic cannot (it is
  XeTeX-only and the T5 author line needs pdfTeX's `t5-lmr`), pdflatex can.
- **`[aidriven]` cross-checked at last.** The thesis is on arXiv as
  `2602.09210v1` (xxi+123 pp.); added to `papers/` 15:05. PROTOCOL.md's "the
  one exception" note and its §2 row rewritten from the full text. Two
  corrections it forces: the row said separation metrics were "not reported
  (paired t-test on an unspecified quality proxy)" — wrong, SDR/SIR/SAR are at
  Eqs. 4.9–4.11 (p. 42) with 95% CIs in Fig. 4.4 and Table A.1 (p. 82); and
  **`report/paper.tex` §1's "26.8 dB on this dataset" is wrong** — 26.8 dB is
  VAE-WMT on "Dataset One" (Kaggle + CirCor + Chest Wall), Table A.2 (p. 83);
  on HLS-CMDS ("Dataset Two") the same model scores **15.1 dB**. Still open.

### CirCor robustness spike (`~/spike-review`, branch `circor-robust`, not pushed)

Brief: after the 17 Sep resampling-unit sensitivity, ResNet-18's HLS-CMDS
R1/R2 intervals contain 0 under source-type clustering and only the small CNN
survives, so the letter's only real-patient branch was also the only place
ResNet-18 still held — on one architecture, one lung draw. Scope: pre-register
first, add the small CNN, 3 lung draws, everything else fixed.

- `spike/CIRCOR_ROBUST.md` written and committed (`d541754`, 15:24) **before
  `spike/circor_robust.py` existed**, per the `HYPOTHESIS.md`/`CIRCOR.md`
  convention. `spike/circor_robust.py` committed before being run
  (`93394cc`, 15:28); it imports `circor.py`'s label rule, groups, split, crop,
  conditions, windows, optimiser and stopping rule rather than restating them.
- **Run 1 died: `CUDA OOM`, 500 MiB**, in the validation pass of
  `resnet18/unmatched/seed0` — `circor.py` splits validation and prediction
  into 4096-window batches and another job held 9.65 of the 12 GB card.
  Fixed to 512 (`db345e7`, 15:47) after checking it changes nothing: BatchNorm
  is in eval mode, so 4096 vs 512 agrees to **2.7e-7** with identical argmax on
  both architectures. Training batch 32 untouched.
- **Run 2 completed and failed G3.** SDRs reproduced *exactly*
  (BAND +6.2 / INT +1.3 / SEP_B1 −2.6) and the split reproduced at
  486 / 86 / 244 groups, so the data path is bit-faithful. But R1 came back
  **+14.8 vs the 14 Sep +11.1**, R2 **+20.6 vs +17.6**, with different stopping
  epochs.
- **Cause: `spike/circor.py` is not reproducible.** `cudnn.deterministic` is
  `False`, so ResNet-18's conv backward accumulates nondeterministically.
  Measured: the *same* `fit_es`, same seed, same data, run twice → different
  weights (max abs diff **9.5e-2**) and epochs **8 vs 6**; `fit_es` vs
  `circor.fit_es` differ by the same order (1.7e-1, 8 vs 11). One function run
  twice diverges as much as two different functions. **The letter's published
  R1 +11.1 / R2 +17.6 are one draw from a distribution.** The three seeds
  average over initialisation, not over cuDNN.
  - The 09-18 README line — GPU cells "not bit-reproducible across GPUs or
    CUDA builds" — understates this: they are not bit-reproducible on the
    *same* GPU and CUDA build. That line needs correcting. It also lists the
    log-Mel CNN as GPU-trained; `src/heart_classifier_cnn.py` never touches
    CUDA and runs on CPU.
- **G3 amended, tolerance not widened** (`9ddc3a0`, 17:34), written before R1
  or R2 were read for either architecture, disclosing what the failure had
  already printed. G3's ±1.0 BA half compared against a quantity that is not
  reproducible, so it was *replaced*, not loosened: G3a keeps the SDR
  comparison (data path is CPU/numpy and does reproduce); **G3b** requires a
  refit of `resnet18/unmatched/seed0` to be **bit-identical** under
  deterministic mode.
- `spike/determinism.py` (`2491607`, 17:37) — `use_deterministic_algorithms` +
  `cudnn.deterministic` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, called from both
  scripts' `main()`. Also `.gitignore` for `spike/out/*_cache/`: a stray
  `git add -A` had committed the 2.8 GB condition cache and pushed `.git` to
  2.0 GB; reset, unstaged, gc'd back to 60 MB. Neither `circor_cache` nor
  `confirm_cache` is tracked upstream.
- **Run 3 died too** — `circor.py` had been given the determinism call but not
  the batch fix, so the baseline rerun OOM'd at the identical line. Fixed
  (`130b909`, 17:46); `INFER_BATCH` now has one definition.
- **Result (`d489242`, 19:10).** Deterministic, 2 architectures, 3 draws,
  244 test groups. G3a passed exactly; **G3b passed — parameters bit-identical,
  epoch 6 reproduced**, so these numbers rerun.

  | arch | G2 | R1 | R2 |
  |---|---|---|---|
  | `resnet18` | 0.737 | **+15.2** [+12.1, +18.1] | **+21.1** [+17.8, +23.9] |
  | `small_cnn` | 0.644 | **+6.8** [+3.6, +10.0] | **+6.6** [+4.4, +8.8] |

  All four intervals exclude 0. Verdict per the pre-registered rule: **the
  effect is a property of band-limiting, not of one architecture** — though
  the magnitude is 2–3× smaller on the CNN, so the letter's claim is "not an
  artifact of ResNet-18", not "the architectures agree".
- **Three lung draws turned out not to be needed.** Per-draw R1: resnet
  15.3 / 15.1 / 15.0, cnn 7.5 / 6.9 / 6.1 — spread **≤0.3 points** on resnet.
  One draw was always enough; what moved the old numbers ~3 points was GPU
  nondeterminism, not draw luck. This retroactively vindicates `CIRCOR.md`'s
  single draw.
- Still running at 19:28: `spike/circor.py` deterministic, to give the letter a
  baseline that reproduces. Queued after it: `spike/resnet_arch3.py` under
  determinism — of the four rows in `review_bootstrap.csv`, `cnn`, `svm` and
  `svm_tuned` are CPU/deterministic and only `resnet` is GPU-trained
  (`resnet_arch3.py:22`), so the 17 Sep "only the CNN survives type-clustering"
  rests, on its ResNet half, on one nondeterministic draw. To flip it the point
  estimate would have to move ~8–10 points against the ~3 seen on CirCor, but
  HLS-CMDS is 50 recordings not 1,695, so that is arithmetic, not a rerun.
- Handoff still blocked: `SouthsideProgramer/teep-spike` does not exist on
  GitHub (`gh repo view` → "Could not resolve"), so the seven commits sit on a
  local branch whose origin is Satya's working copy. Nothing pushed; Satya's
  tree untouched (read-only to this account anyway).
