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

  Also Baseline 2, a leakage-safe reproduction of the Sensors 2026
  supervised-NMF method: two fixed dictionaries (Ki=10 heart, Kr=20 lung,
  the paper's own notation) learned from isolated H/L recordings via
  KL-divergence MU-NMF (100 iterations), then frozen while 60 MU
  iterations solve per-mixture activations; separation via a soft mask on
  the mixture's complex STFT. Dictionaries are fit only on
  `eval_harness`'s fold-safe `hs_allowed`/`ls_allowed` pool (`make_supervised_nmf_baseline`),
  never on recordings that leak into the held-out fold's mixtures — the
  leakage trap PROTOCOL.md §5.2 flags by name. 5-fold result: heart SDR
  -12.82±1.31 dB, lung SDR -15.09±0.98 dB — essentially tied with Baseline
  1 on heart and ~1.3 dB worse on lung, i.e. this reproduction does not
  clearly beat the zero-training bandpass floor once leakage is removed.

- **`Makefile`** — setup (`install`), pipeline sanity checks (`validate`,
  `split`, `eval-harness`), the baselines (`baselines`/`baseline1`/`baseline2`),
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
- `[nmfcrnn]`'s dataset (PROTOCOL.md §2 table) is unresolved between two
  conflicting sources (ICBHI+Fraiwan vs. HLS-CMDS isolated recordings) —
  matters because it's also Baseline 2's own reproduction target.
- Independently re-read Yaqub et al. (`[spectrotemporal]`) and `[nmfcrnn]`
  from the primary sources — currently sourced only from the review's
  summary of the Notion Reading List.

**Older, still open:**
- Two `fit_and_separate_fn(hs_allowed, ls_allowed) -> separate_fn(mixed, sr)`
  implementations exist in `src/baselines.py` (bandpass, supervised NMF)
  and both plug into `cross_validate()` — but per the review, both need
  re-running on the valid-only subset before their comparison means
  anything; a third method (SSA / PL-NMF / LingoNMF) is still open too.
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
