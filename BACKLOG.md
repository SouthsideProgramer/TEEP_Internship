# Backlog

Tracks what we've built together in `src/` on top of the existing HLS-CMDS
loading/plotting scripts, and what's still open.

## Done (2026-08-17 session)

- **`src/statistics/audio_quality.py`** — scans every `.wav` referenced by `HS.csv`,
  `LS.csv`, and `Mix.csv` (535 files total) for duration, sample rate,
  channel count, and clipping (samples pinned at the int16 full-scale rail).
  Aggregates per Heart/Lung Sound Type and per Location. Writes CSVs to
  `src/statistics/audio_quality_reports/`.
  - Finding: format is perfectly uniform (4000 Hz, mono, 15.0s) everywhere.
  - Finding: 36/535 files (6.7%) clip, but only 1 sample each, and only in
    `mix/`'s *mixed* recordings (M0087, M0111–M0145) — looks like a
    single-sample summation overflow from combining heart+lung during
    mixing, not a real recording defect. Not yet reported upstream.

- **`src/visualization/plot_per_class.py`** — one representative waveform + mel-spectrogram
  per Heart Sound Type (10) and Lung Sound Type (6), saved as PNG grids to
  `src/visualization/plots/`.

- **`src/metrics.py`** — BSS Eval (SDR/SIR/SAR) wrapper around
  `mir_eval.separation.bss_eval_sources`, specialized for the heart/lung
  2-source case:
  - `evaluate_heart_lung()` — core 2-source metric.
  - `evaluate_mix_row()` / `evaluate_dataset()` — run a caller-supplied
    `separate_fn(mixed, sr) -> (heart_est, lung_est)` over one or all 145
    `Mix.csv` triplets, loading ground truth from `HLS-CMDS/Dataset.v2/mix/`.
  - `summarize_by_class()` — mean/std/count SDR/SIR/SAR grouped by class or
    location.

- **`src/test_metrics.py`** — pytest suite, two kinds of checks:
  - Cross-checks against a direct `mir_eval` call (synthetic + real audio),
    to catch wrapper bugs independent of metric correctness.
  - Synthetic mixtures with closed-form ground truth (additive noise at a
    known SNR, additive cross-source leakage at a known SIR) — metrics must
    land within ~1 dB of the theoretical value. 10/10 passing.

- **Environment**: installed `matplotlib`, `pandas`, `soundfile`, `mir_eval`,
  `pytest` into `.venv` (present in `requirements.txt` intent but missing
  from the env, or missing from `requirements.txt` entirely) and pinned them
  in `requirements.txt`. Added `.pytest_cache/` to `.gitignore`.

- **`src/split.py`** — leakage-safe, triplet-level train/eval split. Content-hashes
  every HS/LS/mix audio file (ignoring WAV headers) to find that Mix.csv's 145 rows
  only draw from 89 distinct heart recordings and 74 distinct lung recordings (reused
  up to 6x), 32/89 and 30/74 of which are byte-identical to a file also listed
  standalone in HS.csv/LS.csv. Mix rows chain into 29 connected components (by shared
  heart or lung recording) ranging from 1 to 32 rows. `assign_folds()` splits at the
  connected-component level (never splitting a leak group across folds), and
  `dictionary_pool()` returns, per fold, the HS.csv/LS.csv rows safe to fit a
  dictionary on for evaluating that fold (excludes anything content-matching that
  fold's held-out mixtures).

- **`src/eval_harness.py`** — `cross_validate(fit_and_separate_fn, n_folds=5)` runs
  the fold loop: for each fold, calls the caller's `fit_and_separate_fn` with that
  fold's restricted dictionary pool, evaluates the returned separator on the held-out
  mix rows via `metrics.evaluate_dataset`, and aggregates two ways —
  `aggregate_by_fold` (mean per fold x source) and `aggregate_across_folds` (mean +/-
  std across folds per source, the headline CV number).

- **`src/test_split.py`** — 15 tests, the load-bearing one being
  `test_dictionary_pool_excludes_all_held_out_recordings`: for every fold, hashes
  every recording in that fold's dictionary pool and asserts none of them match a
  recording used in that fold's held-out mixtures. Also checks leak groups never
  split across folds, fold coverage/balance, reproducibility, and that
  `cross_validate` actually hands each fold its restricted pool (not the full
  HS.csv/LS.csv every time). All passing; runs in ~2 min since it evaluates real BSS
  Eval over real 15s recordings, several folds, several times.

- **Layout refactor**: split `src/`'s flat file list into `src/visualization/`
  (`audio_plotter.py`, `audio_spectrogram.py`, `donut_chart.py`,
  `plot_per_class.py`, plus generated `plots/`) and `src/statistics/`
  (`audio_quality.py`, plus generated `audio_quality_reports/`).
  `load_dataset.py`, `metrics.py`, `split.py`, `eval_harness.py`, and the
  tests stay at `src/` top level (core/eval infra, not plotting or
  descriptive stats). Fixed the two moved files that import `load_dataset`
  to add `src/` back onto `sys.path`, and fixed `audio_plotter.py`/
  `audio_spectrogram.py`'s `EXAMPLES_DIR` for the extra directory depth.
  Re-verified every moved script runs from its new location. Updated
  `README.md`'s layout tree and run commands, and `.gitignore` for the new
  generated-output paths.

- **`PROTOCOL.md`** — research protocol memo: the gap (nobody has compared
  classification accuracy on separated-from-mixture H/L sounds against
  accuracy on the isolated ground truth, on HLS-CMDS, under a leakage-safe
  split), a related-work table of six papers reviewed, and the method that
  maps onto what's already built (`split.py` folds, `metrics.py`/
  `eval_harness.py` scoring) plus what's still open (classifier choice,
  separation method, weight-sharing decision).

- **`src/baselines.py`** — Baseline 1 (simple bandpass filtering), the
  zero-training sanity floor referenced in `PROTOCOL.md` §5.2. Cutoffs
  (heart 20-200 Hz, lung 150-1000 Hz) came from averaged Welch PSD over
  HS.csv/LS.csv's own isolated recordings, not literature defaults.
  Zero-phase Butterworth via `scipy.signal.sosfiltfilt`. Plugs directly into
  `eval_harness.cross_validate` via `fit_bandpass_baseline`. 5-fold result:
  heart SDR -12.65±1.40 dB, lung SDR -13.83±1.12 dB — improves SIR over a
  no-separation reference but collapses SAR (filter artifacts), the
  textbook interference-vs-artifacts tradeoff, demonstrated on real data.

  Also Baseline 2, a leakage-safe adaptation of the Sensors 2026
  supervised-NMF method (`[nmfcrnn]`, Han et al.): two fixed dictionaries
  (Ki=10 heart/interference, Kr=20 lung/respiratory, the paper's own
  notation) learned from isolated H/L recordings via KL-divergence MU-NMF
  (100 iterations), then frozen while 60 MU iterations solve per-mixture
  activations; separation via a soft mask on the mixture's complex STFT.
  Dictionaries are fit only on `eval_harness`'s fold-safe
  `hs_allowed`/`ls_allowed` pool (`make_supervised_nmf_baseline`), never on
  recordings that leak into the held-out fold's mixtures — the leakage trap
  PROTOCOL.md §5.2 flags by name.

  **Cross-checked against the primary source** (papers added to `papers/`
  this session) — hyperparameters, STFT window, and dataset roles all
  confirmed exact; one real gap found and fixed: the paper denoises every
  snippet with a 4th-order Butterworth bandpass (50–1800 Hz) *before*
  STFT/NMF, which this project's initial reproduction skipped entirely. Now
  added (`DENOISE_BAND`), applied identically to Baseline 3 below. Also
  corrected: STFT hop was an unverified 128, paper's actual value is 256.
  Post-fix 5-fold result: full 145 rows heart SDR -12.95±1.34 dB, lung SDR
  -15.09±1.10 dB (essentially unchanged from pre-fix — the missing
  denoising/hop weren't why this underperforms); additive-only 36 rows heart
  SDR +2.69±2.96 dB, lung SDR +0.62±3.14 dB (first time Baseline 2 has been
  measured on the valid subset). Full-set numbers are still essentially tied
  with Baseline 1 on heart and ~1.3 dB worse on lung — see PROTOCOL.md §5.2
  for the full table and Baseline 3 comparison below.

  Also Baseline 3, `make_standard_nmf_baseline` — standard NMF with no
  learned dictionary, the ablation against Baseline 2 (Sprint ref S3-02;
  Baseline 3 itself is ref S3-03, pulled forward from Sprint 3 into Sprint
  2 since baselines 1 and 2 landed before kickoff). Same total rank
  (Ki+Kr=30), STFT params, and denoising pre-filter as Baseline 2, but W and
  H are both factorized directly out of each held-out mixture's own
  spectrogram — `hs_allowed`/`ls_allowed` go unused, same as Baseline 1 —
  isolating how much of Baseline 2's score comes from the learned/frozen
  dictionary itself. Components are unlabeled by construction; assigned to
  heart/lung post-hoc by spectral centroid (heart energy below ~200 Hz, per
  Baseline 1's own PSD survey). Post-fix 5-fold result: full 145 rows heart
  SDR -13.07±1.38 dB, lung SDR -14.41±1.23 dB; additive-only 36 rows heart
  SDR +3.44±2.02 dB, lung SDR +2.22±3.91 dB.

  **Now a genuine apples-to-apples ablation** (both baselines share
  denoising/STFT/mask code, differing only in whether the dictionary is
  pretrained): on the additive-only subset, Baseline 3 (no learned
  dictionary) matches or slightly beats Baseline 2 (learned dictionary) on
  both sources — `[nmfcrnn]`'s pretrained-dictionary strategy isn't earning
  its keep over blind per-mixture NMF on this dataset, once leakage and the
  non-additive rows are both controlled for. A real finding, provisional
  given n=36 additive rows and wide per-row spread (see PROTOCOL.md §8's
  pooled-std note).

  Also Baseline 4, `fit_ssa_baseline`/`mssa_separate` — multi-stage SSA
  (MSSA), reproducing Han & Quan, "Cardiorespiratory Sound Separation Using
  Singular Spectrum Analysis," 2025 ICSPS, doi:10.1109/ICSPS66615.2025.11347745
  (`[ssa]`; `papers/Cardiorespiratory_Sound_Separation_Using_Singular_
  Spectrum_Analysis.pdf`). Zero-training, no dictionary/fold-fitting step at
  all (`hs_allowed`/`ls_allowed` unused, no leakage trap applies). All four
  hyperparameters (window length L=50, 250 Hz cardiac/respiratory split, 2%
  eigenvalue threshold, 50% cross-correlation threshold) are stated
  explicitly in the paper and used as-is — see `src/baselines.py`'s Baseline
  4 docstring section for the per-hyperparameter paper citations. Confirmed:
  the 250 Hz split is a physiological frequency in absolute Hz (S1/S2 heart
  sound range), not normalized to the paper's own sample rate, so it sits
  correctly under this project's 4000 Hz / 2000 Hz-Nyquist HLS_CMDS copy
  with no rescaling needed; the paper's own dataset is the same Torabi et
  al. HLS-CMDS descriptor source this project cites, so there's no
  cross-dataset sample-rate mismatch either.

  **Synthetic-set comparison against the paper's own Table I** (the paper
  evaluates on a self-built synthetic set, not HLS_CMDS's Mix.csv — see
  below — so this is the only like-for-like comparison available;
  `build_synthetic_mixes()` reproduces the recipe: 10 heart x 5 lung
  recordings from this project's own HS.csv/LS.csv, all 50 combinatorial
  pairs, +2% RMS Gaussian noise). Paper's MSSA: cardiac SDR 26.4 dB / corr
  99.2%, respiratory SDR 5.3 dB / corr 80.5%. This reproduction: cardiac SDR
  1.24 dB / corr 64.8%, respiratory SDR 6.81 dB / corr 44.4% — **respiratory
  SDR is actually in the paper's ballpark (slightly better), but cardiac SDR
  and both correlations are far below the paper's reported numbers.** This
  gap is larger than Baseline 2's earlier gap-that-wasn't (which turned out
  to be a genuinely harder real-dataset setup, not a bug) — worth treating
  as an open question rather than assuming either "bug" or "harder setup"
  without further digging (see Open items below).

  Also re-run on this project's own real mixtures via `eval_harness`, full
  145 rows and the additive-only 36-row subset, for consistency with
  Baselines 1–3's reporting (not paper-comparable, per the note above, but
  keeps the four-baseline suite internally consistent). Here the picture
  flips: on the additive-only subset, Baseline 4 is the **best of all four
  baselines so far** on both sources (heart SDR +5.17±4.05 dB, lung
  +5.32±3.20 dB — vs. Baseline 3's +3.44/+2.22 dB), with SAR roughly 2x
  higher than Baselines 2/3 (~19/12 dB vs. single digits), consistent with
  SSA's claimed advantage of preserving signal integrity over
  filtering/NMF-style masking. See PROTOCOL.md §5.2 for the full table.

- **`Makefile`** — setup (`install`), pipeline sanity checks (`validate`,
  `split`, `eval-harness`), the baselines (`baselines`/`baseline1`/`baseline2`/`baseline3`/`baseline4`),
  reports (`stats`, `plots`, headless via `MPLBACKEND=Agg`), `test`, and
  cleanup (`clean`/`clean-pyc`/`clean-all`) targets. `PYTHON` defaults to
  `.venv/bin/python`, overridable.

- **`self_learning.md`** — informal working notes (why each step exists),
  separate from `PROTOCOL.md` (protocol) and this file (task log).

- **Dataset relocation + `verify_additive_triplets()`** — the dataset moved
  from a git submodule at `HLS-CMDS/Dataset.v2/` to a flat, untracked
  `HLS_CMDS/` directory (user-driven; `load_dataset.py`'s `DATA_DIR` and
  `mix_dir` casing updated to match). Then **TEEP2026_Sprint0_Review**
  (Satya Adhiyaksa, 2026-08-18) found this copy — sourced from
  github.com/Torabiy/HLS-CMDS, not the Mendeley release the charter
  specifies — has 109/145 `Mix.csv` rows whose mixed recording is
  acoustically unrelated to its named heart/lung sources (`mixed = a·(heart+lung)`
  holds, to 16-bit quantization, for only 36 rows: `M0087` + `M0111`–`M0145`,
  which independently matches this project's own clipping-based audio-quality
  audit exactly). Added `verify_additive_triplets()` to `load_dataset.py`
  (+ `test_load_dataset.py`, 6 tests: synthetic closed-form cases with a
  known-true gain/residual, plus dataset-agnostic structural checks against
  real data) to make this a first-class, re-runnable check rather than a
  one-off finding — confirms the review's 36/145 count and the 4000 Hz
  (not 22,050 Hz) sample-rate discrepancy on the current copy. `PROTOCOL.md`
  and `README.md` updated with the review's findings, citations, and a
  provisional "Dataset" section; **all separation-quality numbers reported
  so far are provisional pending the Mendeley re-download** (see Open items).

## Not yet committed

Everything through the layout refactor + `PROTOCOL.md` landed in commit
`ba0c453` ("metric + visualization"). Since then:
```
 M BACKLOG.md
 M PROTOCOL.md
 M README.md
 M requirements.txt
 M src/load_dataset.py
 M src/visualization/audio_plotter.py
 M src/visualization/audio_spectrogram.py
 D HLS-CMDS              # submodule removed; dataset now lives in untracked HLS_CMDS/, pending Mendeley move
?? HLS-CMDS.zip           # user-created backup of the old submodule copy, not this project's
?? HLS_CMDS/              # current (provisional, GitHub-sourced) dataset copy
?? Makefile
?? report/                # report.pdf + TEEP2026_Sprint0_Review, pre-existing, not generated by this session
?? self_learning.md
?? src/baselines.py
?? src/test_load_dataset.py
```
Generated output dirs (`src/visualization/plots/`,
`src/statistics/audio_quality_reports/`) are gitignored, not tracked.

## Open / next up

**Blocking (TEEP2026_Sprint0_Review, do first):**
- Re-download HLS-CMDS from Mendeley into a directory outside this repo;
  document version/date/URL/SHA-256 in `README.md`'s new Dataset section.
  Attempted via the Mendeley public API this session — metadata/linkset
  endpoints work, but actual file bytes sit behind an authenticated,
  JS-driven download flow plain HTTP tooling here can't complete. Needs a
  manual download.
- Re-run `verify_additive_triplets()` (now in `load_dataset.py`) on the
  Mendeley copy, report the valid-row count + sample rate — this is what
  the review needs before deciding whether the project's C1 contribution
  needs rescoping (36 native pairs is a much weaker claim than 145).
- Re-issue Baseline 1/2's results table on the confirmed-valid subset, with
  medians alongside means (current across-fold std understates pooled-row
  spread by ~10×, per the review's statistics note).

**Other, from the same review:**
- The single-sample clipping artifact in `M0111`–`M0145` (+ `M0087`) —
  previously logged as unexplained — is now explained: it's peak
  normalization after summing heart+lung (gain 3.5–135×, not integer
  overflow as originally guessed), and it's the fingerprint of exactly the
  36 rows `verify_additive_triplets()` finds valid. Worth folding this
  mechanism explanation into `report/report.pdf` §2.1 directly.
- ~~`[nmfcrnn]`'s dataset (PROTOCOL.md §2 table) is unresolved between two
  conflicting sources~~ **Resolved** (papers/ cross-check, 2026-08-20): not a
  conflict — HLS-CMDS is the auxiliary NMF dictionary corpus, ICBHI+Fraiwan
  is the paper's separate classification dataset. See PROTOCOL.md §2/§5.2.
- Independently re-read Yaqub et al. (`[spectrotemporal]`) from the primary
  source — `[nmfcrnn]` is now done (see above and PROTOCOL.md §8), Yaqub is
  not; currently still sourced only from the review's summary.

**Older, still open:**
- Four `fit_and_separate_fn(hs_allowed, ls_allowed) -> separate_fn(mixed, sr)`
  implementations now exist in `src/baselines.py` (bandpass, supervised NMF,
  standard/no-dictionary NMF, multi-stage SSA) and all plug into
  `cross_validate()`; Baselines 1–4 have now all been re-run on the
  valid-only 36-row subset (see above), but per the review that subset is
  still small enough (n=36) that these comparisons should be treated as
  provisional, not settled. PL-NMF/LingoNMF (`[aidriven]`) remains the one
  method from PROTOCOL.md §2's related-work table not yet reproduced.
- **New, from the SSA reproduction**: Baseline 4's cardiac SDR/correlation
  on the synthetic set are far below the `[ssa]` paper's own reported
  numbers (1.24 dB/64.8% vs. 26.4 dB/99.2%), while respiratory SDR is
  actually competitive (6.81 vs. 5.3 dB) — an asymmetric gap that doesn't
  fit a simple "harder setup" story the way Baseline 2's gap-that-wasn't
  did. Worth digging into whether Stage 1's raw per-RC peak-frequency
  classification (this project's literal reading of "components with
  dominant frequencies below or equal to 250 Hz") is too permissive for
  low-energy/noise-like RCs relative to whatever more selective criterion
  the paper's "periodic structure analysis" phrase (Sec. II, abstract)
  actually implies but doesn't fully spell out.
- The task that asked for the eval harness referenced "the same
  triplet-level split as S2-03" — this is a real row in the Notion Sprint
  Backlog (per the review), not a missing ticket as previously logged here;
  worth checking its neighbors S3-02/S3-04 (which carry the Baseline 2
  hyperparameters and leakage trap) against what's actually implemented.
- `src/visualization/donut_chart.py` still uses hand-copied counts instead
  of reading `HS.csv`/`LS.csv`/`Mix.csv` — could be swapped to compute from
  `load_dataset.py` like `audio_quality.py` does, so it can't drift from
  the actual data.
- No CI wiring for the test suite — it only runs when invoked manually.
