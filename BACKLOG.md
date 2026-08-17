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

## Not yet committed

Everything above is on disk but untracked/unstaged — nothing has been
`git add`/`git commit`ed this session. `audio_plotter.py`, `audio_spectrogram.py`,
and `donut_chart.py` were already tracked, so their move shows up as a rename
once staged:
```
?? src/eval_harness.py
?? src/metrics.py
?? src/split.py
?? src/test_metrics.py
?? src/test_split.py
?? src/statistics/audio_quality.py
?? src/visualization/plot_per_class.py
R  src/audio_plotter.py -> src/visualization/audio_plotter.py
R  src/audio_spectrogram.py -> src/visualization/audio_spectrogram.py
R  src/donut_chart.py -> src/visualization/donut_chart.py
 M .gitignore
 M README.md
 M requirements.txt
```
(`README.md`'s and the `HLS-CMDS` submodule pointer's *other* changes were
already present before this session — this session added the layout update
on top.) Generated output dirs (`src/visualization/plots/`,
`src/statistics/audio_quality_reports/`) are gitignored, not tracked.

## Open / next up

- No actual separation model or dictionary-learning method exists in the
  repo yet — `eval_harness.py`/`metrics.py` are ready but have nothing to
  evaluate besides no-op baselines. Next real step is a
  `fit_and_separate_fn(hs_allowed, ls_allowed) -> separate_fn(mixed, sr)`
  implementation (e.g. NMF dictionary learning) to plug into
  `cross_validate()`.
- The task that asked for the eval harness referenced "the same
  triplet-level split as S2-03" — no ticket/spec with that ID exists
  anywhere in this repo, so `split.py` was written as the split itself
  rather than reusing a prior implementation. If S2-03 specifies a
  different fold count, seed, or grouping rule elsewhere, `split.py` needs
  to be reconciled with it.
- `src/visualization/donut_chart.py` still uses hand-copied counts instead of reading
  `HS.csv`/`LS.csv`/`Mix.csv` — could be swapped to compute from
  `load_dataset.py` like `audio_quality.py` does, so it can't drift from the
  actual data.
- The single-sample clipping artifact in `mix/M0111`–`M0145` hasn't been
  reported anywhere or explained — worth a closer look or a note to whoever
  maintains the HLS-CMDS source repo.
- No CI wiring for `test_metrics.py`/`test_split.py` — they only run when
  invoked manually.
