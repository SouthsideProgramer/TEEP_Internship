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
  **Partial data point (2026-08-22):** input SDR (raw mixture vs. each
  reference, no separation at all) on this project's own 50-pair synthetic
  set is cardiac -1.12 dB / respiratory +1.46 dB — i.e. this project's
  synthetic mixes are close to balanced, not heart-dominant. The paper
  doesn't report a literal unprocessed-input SDR in Table I, so this can't
  fully confirm "their mixing ratio is heart-dominant enough that their own
  5.7 dB Butterworth baseline was already most of the way to 26.4 dB" — but
  it does newly surface that on this project's own data, Baseline 4 (MSSA)
  barely beats Baseline 1 (bandpass) for cardiac (+2.36 dB vs. +3.32 dB gain
  over input) despite MSSA being cardiac's specialized stage, which is a
  more concrete, checkable lead into the Stage-1-too-permissive suspicion
  above than the asymmetric-gap observation alone.
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

## Done (2026-08-22 session)

- **`src/report_utils.py`** (new) — shared HTML report building blocks
  (`report_shell`, `section`, `stat_tile`, `df_to_html`, `image_figure`,
  `results_dir`, `write_report`), factored out in the same visual style as
  `load_dataset.py`'s existing `dataset_validation.html`/
  `mix_pairing_validation.html`.
- **Consolidated results into a top-level `results/` folder, HTML instead of
  terminal tables**: `split.py`, `metrics.py`, `eval_harness.py`,
  `statistics/audio_quality.py`, and `baselines.py`'s `__main__` blocks used
  to dump long `DataFrame.to_string()` tables straight to the terminal —
  unreadable once baselines.py alone was printing 4 baselines x 2 subsets x
  3 tables. Each now prints only short progress lines and writes a
  self-contained HTML report to `results/` instead (`split_report.html`,
  `metrics_smoke_test.html`, `eval_harness_smoke_test.html`,
  `audio_quality_report.html` + its CSVs under
  `results/audio_quality_reports/`, `baselines_report.html`). The
  `make baseline1`–`4` one-off Makefile targets got the same treatment via
  `report_utils.write_cv_report()`.
- **Visualization scripts wrapped in HTML too**: `audio_plotter.py`,
  `audio_spectrogram.py` (previously didn't even save its figure — added
  `fig.savefig()`), `donut_chart.py` (same gap), and `plot_per_class.py` now
  save their PNGs under `results/plots/` and each writes its own
  `results/<script>_report.html` embedding them, instead of a bare
  `plt.show()` (a no-op under the `MPLBACKEND=Agg` headless backend `make
  plots` already sets).
- Old scattered output locations (`src/visualization/plots/`,
  `src/visualization/combined_plots.png`,
  `src/statistics/audio_quality_reports/`) are gone — removed the stale
  generated files and updated `.gitignore`/`Makefile`'s `clean` target/
  `README.md`'s layout tree accordingly. `load_dataset.py`'s own
  `dataset_validation.html`/`mix_pairing_validation.html` are unchanged
  (still write to `src/` directly) — out of scope for this pass.

- **Audit of the first real `baselines.py` run surfaced a methodology trap,
  now fixed** (see the review that prompted this): the full-145-row SDR/SAR
  numbers aren't measuring separation quality at all, they're dominated by
  the 109 non-additive rows' residual (every method, including doing
  nothing, inherits deeply negative SDR/SAR there because a large share of
  `mixed`'s energy sits outside both references' span). Added
  **`fit_raw_mixture_baseline`/`raw_mixture_separate`** ("Baseline 0": no
  separation at all, `est = mixed` for both sources) to `baselines.py` as
  the reference point every other baseline's SDR now gets compared against,
  plus a **ΔSDR-vs-Baseline-0 table** in `results/baselines_report.html`
  (additive-only subset only — the full-145-row axis is flagged
  non-interpretable in the report's own dek). Confirmed with a real run:
  on the additive-only 36 rows, Baseline 0 alone already scores heart SDR
  2.65 dB / lung SDR -2.14 dB (not 0, since even a perfectly-summed mixture
  isn't literally either reference) — Baseline 1 (bandpass) improves on
  that by **+2.95 dB heart / +6.08 dB lung**, so the bandpass baseline's
  effect is real, not just what passthrough already gets. On the full 145
  rows, Baseline 0 already sits at -13.7/-15.9 dB SDR before any
  "separation" happens, confirming the full-set numbers reflect the
  additivity trap, not method quality.
- **Checked whether `make_standard_nmf_baseline` (Baseline 3)'s
  heart/lung-component assignment peeks at ground truth** (the suspected
  explanation for B3 > B2, an ablation-direction surprise) — it doesn't:
  `baselines.py`'s `is_heart` is computed purely from each component's
  spectral centroid (`W`, `freqs`), no reference to `heart_ref`/`lung_ref`
  anywhere in that closure. `metrics.bss_eval`'s `compute_permutation=True`
  (the other place a 2-source oracle could sneak in) is applied identically
  to every baseline's evaluation, so it can't explain an asymmetry between
  B2 and B3 either. The real, code-confirmed asymmetry is what the
  docstring already said: B3 is **transductive** (fits `W`/`H` fresh on
  each test mixture's own spectrogram, no generalization needed) while B2
  is inductive (frozen dictionary fit once per fold on a separate pool) —
  not ground-truth leakage, but still a real fairness caveat worth keeping
  next to every B2-vs-B3 comparison.
- **Tested whether the 109 non-additive rows are actually just
  desynchronized** (different per-source gain + a small time offset) rather
  than genuinely unrelated audio: fit two independent gains
  (`m ≈ a·h + b·l`, least-squares) and, separately, per-source delay
  alignment via FFT cross-correlation (±200 samples / ±50 ms search
  window, verified correct — recovers lag 0 on known-additive rows) before
  refitting. Neither rescues a single one of the 109 rows: relative
  residual stays at ~0.975–1.0 (99.75–100% of `mixed`'s energy unexplained
  by any linear combination of heart+lung, gain- and delay-corrected) vs.
  ~1e-8–1e-4 on the 36 genuinely additive rows. This is a clean negative
  result — it rules out "just needs realignment" and reinforces that the
  109 rows are acoustically unrelated files, not a synchronization bug.

## Done (2026-08-24 session)

- **`src/synthetic_mix.py`** (new) — synthetic mixing set (S1-09/S1-10/S1-13),
  now the primary evaluation substrate for the headline separation-quality
  table (only 36/145 native Mix.csv rows are additive — too few to source a
  real SDR-vs-difficulty sweep for the charter's C2 knee-point analysis).
  Construction matches the native dataset's own model
  (`mixed = a*(heart+lung) + noise`, gain calibrated from the 36 native
  rows' own fitted gains via `verify_additive_triplets`), sweeps additive
  Gaussian noise across `SNR_SWEEP_DB = (-5, 15, 35)` dB for a controllable
  difficulty axis (diverging from Han & Quan's fixed 2% RMS level, matching
  their noise type — trimmed from an originally-planned 5 levels to 3 after
  measured per-row cost across all 5 baselines showed a 5-level sweep would
  make the full run multi-hour). Split is source-file-level (S1-13,
  `assign_source_folds`), not triplet-level (S2-03) — full combinatorial
  pairing over HS.csv x LS.csv collapses S2-03's leak-group logic into one
  giant component, so it moves up to assigning each of the 50 heart/50 lung
  files independently to a fold, with a synthetic pair only used for
  held-out evaluation when both sources share the held-out fold. 1500 total
  rows (500 leakage-safe same-fold pairs x 3 SNR levels).

  **S1-10 validation**: `validate_against_native()` applies this exact
  construction (each native row's own real audio + its own fitted gain,
  zero noise) to the 36 native additive rows and re-runs
  `load_dataset.verify_additive_triplets()` itself (reused directly, not
  reimplemented) against the result. All 36/36 reproduce, with residuals
  landing in the same ~1e-4 cluster the genuine native rows occupy (not
  just barely under the pass/fail threshold) — real evidence the synthetic
  substrate is faithful to the dataset's own additive mixtures, not an
  arbitrary process.

  `src/test_synthetic_mix.py` (new, 17 tests): source-file fold-safety
  (mirrors `test_split.py`'s load-bearing leakage test — no heart/lung file
  shared between a fold's dictionary pool and that fold's held-out pairs),
  provenance completeness, reproducibility, and the S1-10 validation
  assertion. All passing.

- **Baseline 5: EVMD** (S4-01, pulled forward from Sprint 4 into Sprint 3,
  same pull-forward pattern as Baseline 3's S3-03), reproducing the
  separation stage of Puneet, Shankar, Koluguri & Srivastava, "Edge-Enabled
  Portable Classifier for Lung Sounds Using Convolutional Neural Networks,"
  IEEE BioCAS 2025 (`papers/Edge-Enabled_Portable_Classifier_for_Lung_
  Sounds_Using_Convolutional_Neural_Networks.pdf`). That paper runs EVMD
  directly on HLS-CMDS mixtures but reports no separation metric at all —
  Baseline 5 is the first SDR/SIR/SAR measurement for it on this dataset.
  Zero-training (no dictionary/leakage trap, like Baselines 1/4). VMD
  itself implemented directly (no package available/installed) via
  frequency-domain ADMM; the K=2..10 selection sweep (`alpha=2000`,
  `mu1..mu4` thresholds) follows the paper's own stated values, with the
  criteria-combination logic flagged throughout `code_description.md` as
  this project's best-faith interpretation — the paper's own description is
  admittedly thin here, matching the project lead's own framing when this
  baseline was scoped ("an exact reproduction may not be possible... write
  up what you tried and where it broke"). Also corrects a related-work
  mischaracterization in PROTOCOL.md: `[edgelung]`'s edge hardware is a
  PYNQ-ZU **FPGA board** (~15 W), not an MCU — the ESP32 only handles
  acquisition.

  **Measured, not assumed**: the full K-sweep costs ~15s/mixture at this
  dataset's 15s/4000Hz recordings, dominated by VMD's ADMM iterations —
  infeasible at the full synthetic-set n within this session's timebox
  (~6 hours extrapolated). Baseline 5's synthetic-set column in the table
  below runs on a stratified subsample (n stated explicitly in the report,
  smaller than the other 4 baselines' n on the same column — a disclosed
  compute-driven reduction, not a silent one); its native-additive column
  (n=36) runs in full. `src/test_baselines.py` (new, 6 tests): VMD
  reconstruction-fidelity and permutation-entropy sanity checks (not an
  attempt to re-verify the paper's own ambiguous K-selection semantics,
  already flagged as interpretation). All passing.

  **K-selection convergence, checked directly (not assumed)**: sampled 9
  real mixtures (6 synthetic across the SNR sweep, 3 native additive) and
  logged whether `_evmd_select_k` actually found a `K` satisfying every one
  of the paper's stated criteria. Only **2/9 converged** (both native, both
  at `K=2`); every synthetic-set sample and one native sample fell back to
  the `K=10` ceiling. This is exactly the outcome flagged as possible when
  this baseline was scoped -- the paper's mu1-mu4 cascade, under this
  project's best-faith reading, rarely resolves cleanly. The separation
  itself still runs at the fallback `K` and produces real signal estimates
  (this is what the SDR/SIR/SAR table reports), but "K=10 was used because
  nothing else satisfied the paper's criteria" is a materially different
  claim than "K was found optimal" -- worth stating plainly rather than
  letting the table's numbers imply the reproduction is cleaner than it is.

- **`src/first_sdr_table.py`** (new) — the first SDR/SIR/SAR table across
  all 5 baselines, three columns per (baseline, source): the synthetic set
  (n=1500, or Baseline 5's smaller subsample), the 36 native additive rows,
  and Han & Quan's own Table I (confirmed directly from the primary source
  this session — includes the paper's NMF-baseline row, not just Butterworth/
  MSSA) as a static, non-recomputed column marked as their setup. Every
  computed cell states mean, 95% CI, and n explicitly, so the synthetic
  column's tight CI (large n) is never confused with the native column's
  wider CI (n=36) — per the brief that a tight interval on synthetic data
  and a tight interval on native pairs are not the same claim. Output:
  `results/first_sdr_sir_sar_table.html`. Full run launched in the
  background this session (~1-2 hours, dominated by Baselines 4/5's
  per-row cost at synthetic scale) — see the report itself for the actual
  numbers and interpretation once it completes.

- **`Makefile`**: new `synthetic-set`, `baseline5`, `first-sdr-table`
  targets, matching the existing `baseline1`–`4`/`baselines` pattern.

- **`src/baseline12_synthetic_report.py`** (new) — a focused re-run of
  Baselines 1 (bandpass, S3-01) and 2 (supervised NMF, S3-02) on the
  synthetic set specifically, native additive rows (n=36) alongside as a
  secondary column (no Han & Quan column here — that's the full
  `first_sdr_table.py` table's job). Both implementations are unchanged;
  only the evaluation substrate is new, so this reuses
  `synthetic_mix.evaluate_synthetic`/`eval_harness.cross_validate` directly,
  no new baseline code. `make baseline12-synthetic` ->
  `results/baseline1_2_synthetic_report.html`.

- **Additive-mixture forensics**: confirmed the 36 native additive rows are
  computed sums, not acoustic recordings — two checks on raw int16 samples
  (not the normalized floats `verify_additive_triplets` uses). (1) Residual
  (`mixed − a·(heart+lung)`) is bounded to a handful of LSBs in every row
  (mean RMS 0.48 LSB, max 2.1–10.5 LSB) — categorically inconsistent with
  real acoustic capture (which would show residual on the order of the
  signal's own dynamic range), but ~1.65× the theoretical single-rounding
  floor (0.289 LSB), pointing to at least two compounded integer-rounding
  steps in construction, not one. (2) Gain `a` fit per 1.5s sub-segment is
  flat within every row (mean CV 0.0074%, max 0.029%) — no real drift, only
  short-window estimation noise. Together: solid evidence for "no
  acoustically-native mixtures, these are computed sums," with the one
  honest caveat that the exact construction formula involved more than a
  single rounding operation.

- **Report/citation audit** (`report/report.tex`, `PROTOCOL.md`): fixed 4
  of report.tex's "[Author(s) needed]" bibliography placeholders
  (`aidriven`, `spectrotemporal`, `edgelung`, `ssa` — `nmfcrnn` already had
  full details) using the Notion Reading List's citations, cross-checked
  against primary-source PDFs already in `papers/` for 3 of the 4
  (`aidriven`'s Ph.D. dissertation remains uncross-checked). Independently
  read Yaqub et al. (`[spectrotemporal]`) in full — report.tex's related-work
  table had it filed as "PhysioNet 2016, no separation, binary
  normal/abnormal classification," which was wrong on every count: the
  paper externally validates its classifier **on HLS-CMDS itself**, and its
  Experiment 4 stress test — heart sounds computationally separated from
  HLS-CMDS's own mixed recordings via a **standard bandpass filter** (same
  method class as this project's own Baseline 1) — is the source of the
  89.0%→41.0% accuracy collapse (Table 9: n=5365) that is the single result
  this whole project exists to explain. Rewrote report.tex's related-work
  table row and added a dedicated subsection making this explicit, plus
  updated the Introduction to cite Yaqub as the motivating result rather
  than stating the research question as purely rhetorical. Mirrored the
  same corrections into `PROTOCOL.md` §2/§3/§8, which had partially caught
  this in an earlier session (2026-08-18, from the review's summary) but
  explicitly flagged itself as "not yet independently confirmed from the
  primary source" until now. No LaTeX toolchain is available in this
  environment (`pdflatex`/`latexmk` not installed) — `report.tex` is
  updated but `report.pdf` was not regenerated; needs a local/Overleaf
  rebuild.

## Done (2026-08-25 session)

- **Baseline 6: Conv-TasNet-lite** (`src/convtasnet.py`, `src/baseline6_report.py`,
  `src/test_convtasnet.py`, new) — the first neural, end-to-end learned
  separation baseline, unlike Baselines 1-5's zero-training or
  fixed-dictionary fits.

  **Sample-rate decision**, made explicitly before any training code was
  written (the task brief called this out as a blocking decision): this
  dataset is natively 4000 Hz, so public Conv-TasNet/Sepformer checkpoints
  (trained at 8-16 kHz) don't apply without resampling, which would invent
  no real information and need declaring in the paper as an artifact.
  Decision: train from scratch at the native 4000 Hz rather than resample
  up. This costs nothing physiologically — heart energy sits at 20-200 Hz
  and lung at 100-1000 Hz (Baseline 1's own Welch-PSD survey), comfortably
  under a 2000 Hz Nyquist at 4 kHz.

  **Architecture**: the same encoder/TCN-separator/decoder design as
  Conv-TasNet itself (Luo & Mesgarani, IEEE/ACM TASLP 2019,
  `papers/Conv-TasNet_Surpassing_Ideal_Time-Frequency.pdf`) and NeoSSNet
  (Poh et al., IEEE OJEMB 2024,
  `papers/NeoSSNet_Real-Time_Neonatal_Chest_Sound_Separation_Using_Deep_Learning.pdf`
  — the closest prior work, also a masked Conv-TasNet-style model
  separating heart/lung from one chest channel at 4 kHz), sized down
  ("lite") to ~325K parameters for this dataset's much smaller training
  pool (vs. Conv-TasNet's own ~5M-parameter speech config or NeoSSNet's
  8.4M-parameter transformer-augmented model) — no transformer mask
  generator, just the original stacked-dilated-TCN separator (`TCN_B=64,
  TCN_H=128, TCN_SC=64, TCN_P=3, TCN_X=6, TCN_R=2`, receptive field
  ~1.27s). Sigmoid mask activation per source, independent (no unit-sum
  constraint, per Conv-TasNet's own Sec. IV-A ablation). Fixed source
  order (heart = channel 0, lung = channel 1) instead of
  permutation-invariant training — heart and lung are distinguishable
  classes here, not interchangeable speakers, matching NeoSSNet's own
  choice.

  **Training loop**: trains a fresh network per fold from that fold's
  leakage-safe `hs_allowed`/`ls_allowed` pool (same contract every
  baseline's `fit_and_separate_fn` receives). Training mixtures are
  synthesized on the fly, reusing `synthetic_mix.py`'s own mixing recipe
  (`mixed = a*(heart+lung) + noise`, gain calibrated from the native
  additive rows, noise drawn to a uniformly-sampled SNR within
  `SNR_SWEEP_DB`'s range) rather than inventing a separate one. A
  file-level 80/20 split inside the allowed pool (never touching the
  outer CV fold's held-out data) gives an internal validation set for
  early stopping and LR scheduling — AdamW, LR halved after 4 epochs
  without validation-SI-SDR improvement, best-checkpoint restore, matching
  NeoSSNet's own training recipe (`MAX_EPOCHS=25, STEPS_PER_EPOCH=40,
  BATCH_SIZE=8`, ~4-second training crops).

  **Measured, not assumed**: full 5-fold CV on both substrates took
  568.2s total this run (429.8s synthetic set, 138.4s native additive) on
  the available CUDA GPU — every one of the 10 fold-trainings ran the
  full 25 epochs without early-stopping, and validation SI-SDR improved
  monotonically fold-to-fold (0.78-3.35 dB on synthetic, 1.91-3.02 dB on
  native), i.e. the training loop is actually learning something, not
  just running epochs. Results (`results/baseline6_report.html`):

  | source | synthetic SDR (n=1500) | native-additive SDR (n=36) |
  |---|---|---|
  | heart | 3.16 ± 0.40 dB | 5.15 ± 2.74 dB |
  | lung  | 0.37 ± 0.35 dB | 2.14 ± 2.14 dB |

  This **beats Baselines 1 (bandpass) and 2 (supervised NMF) on the
  synthetic set on both sources** (Baseline 1: heart 2.10±0.43 / lung
  -1.24±0.44; Baseline 2: heart -0.70±0.40 / lung -3.73±0.40 — see
  `results/baseline1_2_synthetic_report.html`), the first baseline in this
  project to post a positive lung SDR on the synthetic set at all. On the
  native-additive column (n=36, wide CI) it's roughly tied with Baseline 1
  (heart 5.46±2.57, lung 4.05±2.29) rather than a clear win — read the
  synthetic column (n=1500) as the more reliable comparison, same caveat
  every other baseline's synthetic-vs-native table carries.

  **Honest scope**: this is the first neural baseline, establishing the
  training-loop infrastructure — not a tuned, converged model. No
  hyperparameter sweep was run (unlike NeoSSNet's own Table VII ablation);
  the config above is this project's first working choice, not a search
  result. `test_convtasnet.py` (new, 12 tests, all passing): model
  forward-pass shape/finiteness for arbitrary input lengths, SI-SDR loss
  sanity (scale invariance, identity ceiling, uncorrelated-estimate
  floor), the augmented-batch sampler's output shapes, and an end-to-end
  training-loop smoke test that the `fit_and_separate_fn` contract works
  and a too-small allowed pool raises rather than silently training on
  nothing.

  `Makefile`: new `baseline6` target -> `results/baseline6_report.html`;
  `test` target now also runs `test_convtasnet.py`.

- **Refactor: split `baselines.py`, move tests into `test/`** (requested
  directly, not tied to a Notion ref). `src/baselines.py` had grown to
  ~790 lines holding Baselines 0-5 plus report-generation glue; split for
  readability now that there are 6 baselines (and growing) each with their
  own paper, hyperparameters, and interpretation notes.

  - **`src/baseline/`** (new package, still under `src/` per instruction):
    `common.py` (the few pieces literally shared across baselines —
    `HEART_BAND`/`LUNG_BAND`, `_bandpass`, `_peak_frequency`), and
    `baseline1.py` .. `baseline5.py`, one module per baseline, each keeping
    its own docstring/paper citation/hyperparameters/interpretation notes
    verbatim from the old `baselines.py`. `baseline3.py` imports its NMF
    machinery directly from `baseline2.py` (`DENOISE_BAND`, `N_FFT`,
    `HOP_LENGTH`, `K_HEART`, `K_LUNG`, `_nmf_kl`, `_EPS`) rather than
    duplicating it, since Baseline 3 is explicitly an ablation of Baseline
    2 — the import itself now documents that relationship. No `__init__.py`
    re-exports: every downstream import site (`first_sdr_table.py`,
    `baseline12_synthetic_report.py`, `Makefile`'s `baseline1`-`baseline5`
    targets, `test/test_baseline5.py`) was updated to import directly from
    `baseline.baselineN`, not through a compatibility shim.
  - **`src/baselines.py`** now holds only Baseline 0 (raw mixture, no
    separation) plus the report-generation glue (SSA-paper-style synthetic
    set, `_run_and_report`, `_delta_vs_raw_mixture_section`) and the
    `__main__` block that builds `results/baselines_report.html` —
    unchanged behavior, just re-sourcing Baselines 1-4 from their new
    modules. Baseline 6 (`convtasnet.py`) stays where it is — a
    fundamentally different kind of module (trains a model, not a fixed
    `separate_fn`/`fit_fn` pair), not moved into `baseline/`.
  - **`src/test/`** (new package): all `test_*.py` files moved here
    (`git mv` where already tracked), plus a `conftest.py` that puts `src/`
    on `sys.path` so every test file's existing imports
    (`from load_dataset import ...`, `from baseline.baseline1 import ...`)
    keep working unchanged despite tests now living one directory below
    the modules they import. `test_baselines.py` renamed to
    `test_baseline5.py` (its 6 tests are 100% about EVMD internals) with
    its import updated to `from baseline.baseline5 import ...`.
  - `Makefile`: `test` target now runs `pytest test/` (auto-discovers all
    6 test files, whereas the previous hand-listed target had silently
    excluded `test_synthetic_mix.py`/`test_baselines.py` — folded that gap
    closed as a natural consequence of the reorg, not a separate fix); the
    `baseline1`-`baseline5` targets' inline `-c` snippets updated to import
    from `baseline.baselineN`; help text and the "Separation baselines"
    header updated to name the new file layout.
  - `code_description.md`: split the old single `## baselines.py` section
    into `## baselines.py` / `## baseline/common.py` / `## baseline/
    baseline1.py` .. `## baseline/baseline5.py` / `## convtasnet.py`
    (section headers match file paths, per this doc's own stated
    convention), with in-body file-path mentions and `test_baselines.py`/
    `test_convtasnet.py` references updated to their new locations.
    `PROTOCOL.md` §5.2's per-baseline "(implemented, `src/baselines.py`)"
    pointers updated to the correct new per-baseline file; dated historical
    entries (§8's "Done this session" log, the 2026-08-18 review update)
    left as-is, same as BACKLOG.md's own past entries — those are accurate
    records of what was true at the time, not current-state pointers.

  **Verified, not assumed**: full `make test` (66 tests across all 6 files)
  passes from the new layout; every top-level script
  (`baselines.py`, `baseline12_synthetic_report.py`, `first_sdr_table.py`,
  `baseline6_report.py`, `convtasnet.py`) imports cleanly; a direct
  functional smoke test ran Baselines 0/1/4/5's `separate_fn`s and
  Baselines 2/3's `fit_and_separate_fn`s end-to-end post-split (confirming
  Baseline 3's import-from-Baseline-2 wiring actually works at runtime, not
  just at import time).
