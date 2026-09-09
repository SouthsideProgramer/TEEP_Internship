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

**Closed 2026-08-19 (provenance check):**
- ~~Re-download HLS-CMDS from Mendeley~~ — done, v3, retrieved 2026-08-19.
  Byte-identical to this working copy on all six top-level artifacts
  (`HS`/`LS`/`Mix`, each `.csv` and `.zip`). Checksums live in `README.md`.
  The earlier suspicion that the GitHub copy was a downsampled derivative was
  wrong: the mirror is faithful.
- ~~Re-run `verify_additive_triplets()` on the Mendeley copy~~ — moot, same
  bytes. 36/145 additive and 4000 Hz are properties of the release. The C1
  rescoping decision this gated was taken 2026-08-19.
- ~~Re-issue Baseline 1/2's results on the confirmed-valid subset with
  medians~~ — done; see `results/baselines_report.html`.

**Still blocking nothing, but still open:**
- Verify the descriptor paper's stated 22,050 Hz against its own text, with a
  page reference (S1-14). Decides how the audit comment phrases the
  sample-rate finding.

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

## Done (2026-08-27 session)

- **`src/heart_classifier.py`** (new) — Condition A of PROTOCOL.md §5.3
  (S2-03, pulled forward from S5 into S2, three weeks early — the
  separation track finished roughly three sprints ahead of the Gantt, and
  the charter's own ordering rule spends a lead on C2, not more C1 depth).
  The first accuracy number in the project (everything before it is
  SDR/SIR/SAR in dB) and the Condition A anchor the rest of the charter's
  C2 knee-point sweep gets compared against; reported on the same
  leak-group folds every separation baseline already uses, with a CI.

  **Blocking design decision, made before any classifier code**: binary
  vs. grouped classes. Full 10-class Heart Sound Type classification is out
  per the charter — S4 has only 2 recordings total, so under the
  leak-group 5-fold split it cannot appear in every fold's training set at
  all, and AV Block/Tachycardia (n=3 each) are barely better off. Decided
  on grouped classes, using the *exact* four groups Yaqub et al.
  (`[spectrotemporal]`) use for this same dataset — confirmed directly
  from the primary source (not the review's summary, per this project's
  own reading-primary-sources rule), Secs. 5.1–5.2, Tables 6–9: Normal,
  Murmur, Extra Sound, Rhythm Disorder, the same class list their own
  89%→41% collapse (the motivating result this project exists to explain)
  is measured on. Mapping HLS-CMDS's 10 types onto those four is
  unambiguous from clinical terminology: Normal→Normal (n=9), the 4 murmur
  types→Murmur (n=24), S3/S4→Extra Sound (n=7), AFib/Tachycardia/AV
  Block→Rhythm Disorder (n=10). Chosen over binary Normal/Abnormal
  specifically because it keeps Condition A's number comparable to the
  86–89% figures the project's own motivating result is anchored to, not
  just because it solves the imbalance problem (binary would too). Full
  paragraph justifying this is in `heart_classifier.py`'s module docstring
  and is now also PROTOCOL.md §5.3's Methods text verbatim.

  **`split.py`'s new `assign_hs_folds()`** extends the existing leak-group
  split to HS.csv rows themselves (previously `dictionary_pool()` only
  decided whether a recording was *excluded* from a fold's dictionary
  pool, not what fold the recording itself belongs to): a recording
  byte-identical to a Mix.csv leak group's heart component inherits that
  leak group's fold; recordings never reused in any mixture have no
  leak-group constraint and are assigned by balanced round-robin,
  stratified by `class_group` so no fold is starved of a minority group.
  Verified on the real data: every one of the 5 folds' training sets
  contains all 4 class groups (checked directly, not assumed — see
  `test_heart_classifier.py`).

  **Architecture: one, deliberately**, per the charter's own framing (this
  is a measuring instrument, not a contribution — S7-06 repeats the whole
  curve with a second architecture as a robustness check once the curve
  itself exists). 13 MFCCs pooled to mean+std over time (26-dim feature
  vector) + an RBF-kernel, class-balanced SVM, chosen over a CNN or other
  deep spectrogram model because n=50 recordings (as few as ~35–40/fold in
  training) is far too little data for a deep model without the result
  being dominated by overfitting noise rather than signal — PROTOCOL.md
  §5.3's own stated fallback for exactly this regime. No hyperparameter
  search was run (`C=1.0`, `gamma='scale'` are sklearn's defaults, not
  tuned on this data), same disclosed-first-configuration framing as
  Baseline 6.

  **Measured, not assumed** (`results/heart_classifier_report.html`,
  5-fold CV, `n_folds=5, seed=0`): accuracy 58.0% ± 7.3% (95% CI across
  folds), Macro-F1 0.43 ± 0.07. Per-class-group recall, pooled across all
  folds' held-out predictions: Normal 44%, Murmur 88%, Extra Sound 29%,
  Rhythm Disorder 20% — a majority-class bias toward Murmur (n=24/50, the
  largest group), the expected shape of the class-imbalance problem the
  grouping decision reduced (vs. 10-class's n=2/3 classes) but did not
  eliminate. With only 5 folds this CI is indicative, not decisive, per
  PROTOCOL.md §6's own caveat on small fold counts.

  `src/test/test_heart_classifier.py` (new, 12 tests): class-group mapping
  totals match the documented decision exactly, `assign_hs_folds()`'s
  leak-safety property (a recording reused in Mix.csv gets the same fold
  as that mixture's leak group — mirrors `test_split.py`'s load-bearing
  dictionary-pool test), every fold's training set covers all 4 class
  groups, feature-vector shape, and cross-validation output shape/range.
  All passing; full `make test` (78 tests across all 7 files) passes.

  `Makefile`: new `heart-classifier` target → `results/heart_classifier_
  report.html`.

  **Open, unblocked by this session**: Condition B (classifying the
  separated `heart_est` outputs from §5.2's baselines with this same
  classifier) and the weight-sharing decision (same trained weights vs.
  retrained on separated audio) — PROTOCOL.md §5.3.

- **`src/degradation.py`** (new) — S6-01, controlled separation-degradation
  scheme, designed and validated (PROTOCOL.md §5.3.1). Ticket history is
  two deliberate pull-forwards: originally S6, pulled into S5 because S6
  loses two working days to holidays (25 Sep Fri, 28 Sep Mon), leaving only
  three days to produce the paper's main figure — moving the design work
  earlier is the fix the charter asked for by 17 Sep. Pulled forward again
  in the Sprint 1 review, S5 into S2, alongside Condition A above (same "a
  lead is spent on C2, not more C1 depth" reasoning). Ref code kept as
  S6-01 throughout both moves.

  **The scheme**: `degrade_toward_ground_truth(g, s, alpha)` linearly
  blends a real separation baseline's own estimate `s` toward its ground
  truth `g` — `degraded = g + alpha*(s - g)` — so alpha=0 is clean and
  alpha=1 is that baseline's real output. Same "signal + scaled error
  term" shape as `synthetic_mix.py`'s own `SNR_SWEEP_DB`, except the
  injected term is a real method's own characteristic error (its own
  SIR/SAR balance) rather than i.i.d. Gaussian noise — the two schemes are
  kept as complements, not alternatives: the SNR sweep stresses "how hard
  is the separation problem" (mixture-side difficulty), this scheme
  stresses "how much of this specific method's own artifact to keep"
  (output-side, decoupled from mixture difficulty).

  **The actual point of this ticket — the curve's x-axis decision**: three
  candidates (measured SDR from the six real methods, synthetic SNR, or
  both overlaid). Decided: **SDR, never SNR**, with real-method points and
  this scheme's dense sweep sharing one SDR axis. SNR is a mixture-side
  property, undefined for Yaqub et al.'s own bandpass-separated point
  (`[spectrotemporal]`, their 89%→41% collapse, §3) — they never report an
  input SNR, only that they ran on HLS-CMDS's mixtures as released. SDR is
  the one unit computable for every point the curve needs, including
  theirs (via this project's own Baseline 1 reproduction, confirmed the
  same method class) — the specific choice that lets their result be
  placed on the curve as a point rather than cited as an anecdote.

  **Validated on real data, not assumed** (`results/degradation_scheme_
  report.html`, one native additive row, Baselines 1 + 4 — different
  artifact signatures): SDR(alpha) is monotonically non-increasing for
  both, and alpha=1 exactly reproduces each method's independently
  measured SDR.

  **Real finding from that validation, not glossed over**: a uniform alpha
  grid gives a badly uneven SDR spread — on the row checked, SDR fell from
  ~262 dB at alpha=0 to ~10.6 dB by alpha=0.1 alone, then flattened to
  ~4.8 dB by alpha=1, since dB is most sensitive to small absolute error
  near a perfect match. Added `find_alpha_for_target_sdr()` (bisection over
  alpha, valid because SDR(alpha) was checked monotonic) so a future
  curve-builder can request an evenly-spaced target-SDR grid directly —
  verified to land within ~0.05 dB of every requested target across both
  methods.

  `src/test/test_degradation.py` (new, 11 tests): interpolation
  endpoints/midpoint, length truncation, source isolation (degrading heart
  never disturbs the lung estimate's score), the monotonicity property on
  real dataset audio, and the root-finder's accuracy plus endpoint-clamping
  behavior. All passing.

  `Makefile`: new `degradation-scheme` target → `results/degradation_
  scheme_report.html`.

  **Still open, explicitly not this ticket's job**: building the actual
  accuracy-vs-SDR curve (needs Condition B to exist first), and which
  methods'/rows' sweeps populate the final figure.

- **Failure mode analysis (confusion matrices)** for the Condition A
  classifier (`heart_classifier.py`) — the low per-class-group recalls
  already reported (Normal 44%, Extra Sound 29%, Rhythm Disorder 20%)
  don't say *what* each class gets mistaken for, only that it's often
  wrong. Added `confusion_counts()` (raw, pooled across all folds' held-out
  predictions — every recording predicted exactly once, so pooling never
  double-counts), `confusion_recall_pct()` (row-normalized, matching the
  convention Yaqub et al.'s own HLS-CMDS confusion matrices use, Figs.
  13–16, for direct visual comparability), `top_confusions()` (the single
  most common *wrong* prediction per true class, excluding the diagonal),
  and `plot_confusion_heatmap()` (`results/plots/heart_classifier_
  confusion_matrix.png`).

  **Measured, not assumed**: the failure mode has a clear direction, not a
  symmetric spread. Both minority classes' errors pull toward **Murmur**
  specifically — Extra Sound is predicted Murmur *more* often than it's
  predicted correctly (43% vs. 29% recall), and Rhythm Disorder is
  predicted Murmur about as often as any other single outcome (40% vs. 20%
  recall) — while Murmur itself is almost never mistaken for anything else
  (0% as Normal/Extra Sound, 8% as Rhythm Disorder). Normal is the
  exception: its dominant confusion runs toward Rhythm Disorder (33%), not
  Murmur. Consistent with Murmur's training-set dominance (n=24/50)
  pulling the decision boundary toward it despite `class_weight='balanced'`
  reweighting the SVM's loss — reweighting the loss doesn't guarantee
  balanced *predictions* when the 26-dim MFCC-summary feature space gives
  the four classes limited separability to begin with. Full table in
  PROTOCOL.md §5.3, right after the Condition A result.

  Flagged as a concrete, checkable question for the S7-06
  second-architecture robustness check: does a richer feature
  representation (full log-mel spectrograms, not MFCC summary statistics)
  reduce the Extra-Sound/Rhythm-Disorder → Murmur pull, or is it inherent
  to how acoustically similar these classes are in this dataset regardless
  of featurization?

  `src/test/test_heart_classifier.py`: 6 new tests (18 total, up from 12)
  — confusion counts sum to each class's true size, diagonal matches the
  per-class recall already reported, row-normalized percentages sum to
  100, the dominant-confusion lookup never returns the diagonal itself,
  its recall matches the raw counts independently, and a small
  hand-constructed confusion matrix checked by hand. All passing.

- **`src/sdr_sweep.py`** (new, S6-02) — *generates* the controlled SDR
  sweep dataset S6-01 designed: runs all six separation baselines (S4-03)
  once per fold over the 36 native additive rows, caches each baseline's
  real `heart_est`/`lung_est` to `results/sdr_sweep_cache/`, then applies
  `find_alphas_for_target_sdrs()` to hit a `(25, 20, 15, 10, 5, 0, -5)` dB
  target grid per (baseline, row, source). Mirrors `synthetic_mix.py`'s
  own split between a cheap provenance table and audio reconstructed on
  demand — `synthesize_sweep_row()` rebuilds each degraded waveform from
  the cache plus a stored alpha, so the sweep never materializes gigabytes
  of pre-degraded audio.

  **Performance fix found before the real run**: a naive per-target
  bisection (`degradation.find_alpha_for_target_sdr`, S6-01's own
  function) costs one expensive BSS-Eval call per bisection iteration per
  target — multiplying that cost by 7 targets per (row, source) for no
  reason. Added `degradation.find_alphas_for_target_sdrs()` (batched):
  probes one coarse, log-spaced alpha grid per (row, source), then
  refines each target from an already-narrow bracket instead of a
  from-scratch [0, 1] search. Measured ~1.8x wall-clock speedup on the
  test suite's tiny case; agreement with the unbatched function verified
  directly (`TestFindAlphasForTargetSdrsBatched`, `test_degradation.py`,
  14 tests total now).

  `src/test/test_sdr_sweep.py` (7 tests, deliberately cheap — 4 rows,
  Baseline 1 only): schema/row-count, alpha validity, the
  `clamped_to_baseline_floor` flag agreeing with alpha, cached `.wav`
  files existing where expected, target-vs-alpha monotonicity, and
  `synthesize_sweep_row()`'s reconstruction matching what was recorded.
  All passing.

  **Real generation run, launched this session**: `python sdr_sweep.py`
  (all 6 baselines × 36 rows × 2 sources × 7 targets). First real-world
  lesson: launching it alongside a full `pytest test/` run in parallel
  starved both of CPU almost completely (single-digit rows/hour) — killing
  the redundant test run (tests had already been verified individually)
  restored throughput to a reasonable rate immediately. Baseline 1
  (bandpass) completed in ~26 minutes; the remaining five baselines
  (Baseline 2's NMF dictionary fitting is markedly slower per-row than
  bandpass/MSSA) were still generating as of this entry — this is a
  multi-hour background job, tracked to completion in a later session
  update rather than blocking this one.

  `Makefile`: new `sdr-sweep` target → `results/sdr_sweep_report.html`.

- **`src/condition_b.py`** (new) — Condition B of PROTOCOL.md §5.3:
  evaluates the Condition A classifier on real separated audio (S6-02's
  cache) instead of isolated ground truth — the controlled-conditions
  reproduction of Yaqub et al.'s 89%→41% collapse, run with all six of
  this project's separation baselines instead of their one bandpass
  filter.

  **Weight-sharing decision** (PROTOCOL.md §5.3's last open item,
  resolved): the *same* trained classifier weights per fold, not
  retrained on separated audio — matching Yaqub et al.'s own methodology
  exactly (their Experiment 3 model *is* their Experiment 4 model, no
  retraining between the two).

  **Fold-basis trap found and fixed**: `heart_classifier.
  assign_classifier_folds()` defaulted to folding HS.csv against the
  *full* 145-row Mix.csv, but `sdr_sweep.py` folds its 36-row
  native-additive substrate independently — a different row set produces
  a different leak-group partition even with the same seed. Evaluating
  "the same trained weights" requires both to agree on what fold k means,
  so `assign_classifier_folds()` gained a `mix_df_with_folds` parameter,
  and `condition_b.build_condition_b_fold_basis()` computes the
  native-additive fold assignment once and feeds it to both the
  classifier and (implicitly, by construction) `sdr_sweep.py`'s own cache
  layout. Also refactored `heart_classifier.cross_validate_classifier()`
  to share a new `train_fold_classifiers()` helper with Condition B,
  rather than duplicating the per-fold training loop.

  **First real result** (Baseline 1/bandpass only — the other five
  baselines' separated audio was still generating at the time): isolated
  accuracy 50.0%, bandpass-separated accuracy 41.7% (36 rows) — an
  8.9-point drop, same direction as Yaqub's collapse but far smaller, and
  **not statistically significant** at n=5 folds (paired t-test p=0.57;
  one fold even shows separated beating isolated). A real but inconclusive
  single data point — full table pending S6-02's completion.

  `src/test/test_condition_b.py` (new, 8 tests): fold basis matches an
  independent `assign_folds()` computation, the classifier never trains
  on a recording in its own held-out fold, schema/row-count, isolated
  rows carry no SDR while separated rows do, and — load-bearing — the
  isolated and separated predictions for the same row come from literally
  the same fitted classifier object. All passing.

  `Makefile`: new `condition-b` target → `results/condition_b_report.html`.

- **`src/sdr_accuracy_curve.py`** (new, S6-03) — widened across the
  Sprint 6 holiday gap: Sprint 6 has only three working days (24 Sep Thu,
  29 Sep Tue, 30 Sep Wed — 25/28 Sep are public holidays), and with
  design (S6-01) and sweep generation (S6-02) already done in S5, this
  and S6-04 are all that remain, which fits. Measures downstream
  classification accuracy at *every* point in S6-02's sweep (not just
  each baseline's real output, the way Condition B does) — the actual
  accuracy-vs-SDR curve PROTOCOL.md §5.3.1 designed the x-axis for.

  **Scripted to run unattended**, per the brief: `measure_accuracy_at_
  each_sdr_point()` checkpoints to `results/sdr_accuracy_curve.csv` every
  25 rows, and a re-run loads that file and skips any `(baseline,
  mixed_id, target_sdr)` already measured — safe to interrupt across the
  holiday gap and resume without redoing completed work. Also asserts
  that each measured row's fold (from S6-02's own provenance) agrees with
  an independently recomputed classifier fold basis, catching a possible
  future fold-basis drift loudly instead of silently using the wrong
  classifier.

  Reuses Condition B's weight-sharing setup exactly (same trained
  classifier per fold, never retrained on degraded audio) and only
  measures `source == "heart"` rows — the sweep's lung-source rows exist
  for BSS Eval's own 2-source bookkeeping and have no classification
  counterpart in this project.

  `src/test/test_sdr_accuracy_curve.py` (new, 10 tests, deliberately
  cheap — 4 rows, Baseline 1 only): only heart rows measured, output
  round-trips through disk, a simulated interrupt-and-resume produces
  identical predictions to a fresh full run while only re-computing the
  missing points (checked via a call-counter on `synthesize_sweep_row`,
  not just matching final numbers), the fold-consistency check raises on
  a tampered row, and the summary's per-group accuracy matches direct
  computation.

  `Makefile`: new `sdr-accuracy-curve` target →
  `results/sdr_accuracy_curve_report.html`.

  **Still open**: the actual full-scale run (all six baselines × seven
  target-SDR points) is pending S6-02's generation finishing all six
  baselines — tracked to completion in a later session update. S6-04
  (presumably the knee-point write-up once the curve exists) is not
  scoped here.

- **`src/sdr_knee_point.py`** (new, S6-04) — the project's headline figure
  per the Sprint 0 re-scope: plots the accuracy-vs-SDR curve and
  identifies the knee point, treating every upstream module (`split.py`,
  the six baselines, `heart_classifier.py`, `degradation.py`,
  `sdr_sweep.py`, `condition_b.py`, `sdr_accuracy_curve.py`) as scaffolding
  for this one plot.

  **Knee-point definition**: operationalizes PROTOCOL.md §4's own wording
  ("the SDR below which separation actively hurts classification
  accuracy, relative to not separating at all") directly, not a
  curvature/elbow heuristic — the SDR where a baseline's accuracy curve
  crosses the *no-separation* accuracy, walking high SDR to low.
  `find_knee_point()` returns `crossed` / `always_above` / `always_below`
  / `noisy_crossing` so a real knee is never confused with small-n noise;
  no smoothing is applied, since interpolating through only 7 points per
  baseline (each averaging a handful of rows) would manufacture precision
  the data doesn't support.

  Added the missing third reference point this needed:
  `condition_b.evaluate_no_separation()` — classifies each row's raw,
  unseparated mixture directly with the same fold-appropriate weights
  Condition A/B use. `condition_b.summarize_by_baseline()`'s output
  ordering was extended (`NO_SEPARATION_LABEL`, `ISOLATED_LABEL`, then
  each separation baseline) so all three condition types summarize
  through one function without producing spurious all-NaN rows for
  baselines not yet evaluated.

  **Honesty about partial data, built in from the start**: the `__main__`
  block checks which baselines actually have cached separated audio
  before plotting and falls back to a from-cache provenance rebuild
  (`sdr_sweep.build_provenance_from_cache()`, new — see below) for
  whichever are ready, rather than blocking on the full six-baseline
  generation run. The rendered figure's own title states how many
  baselines are present vs. pending.

  **`sdr_sweep.py` gained `build_provenance_from_cache()`** (refactoring
  the target-grid computation out of `build_sdr_sweep()` into a shared
  `_target_grid_rows()` helper first): reads a baseline's already-cached
  separated `.wav` files and only (re)computes the target-SDR grid,
  skipping separation entirely. Lets a baseline's sweep be measured as
  soon as *its own* cache is ready, instead of waiting for every baseline
  in one `build_sdr_sweep()` call to finish — directly useful this session
  since the full 6-baseline generation job was (and remains) running for
  hours on a heavily contended shared machine while only Baseline 1 had
  finished. Verified to reproduce `build_sdr_sweep()`'s own output
  exactly on the same cache (`TestBuildProvenanceFromCache`,
  `test_sdr_sweep.py`, 9 tests now).

  `src/test/test_sdr_knee_point.py` (new, 7 tests, synthetic curves —
  pure arithmetic, no audio I/O): a clean interpolated crossing, an
  exact-equality edge case, always-above/always-below, the noisy-crossing
  case detected distinctly, row-order independence, and multi-baseline
  dispatch. All passing.

  `Makefile`: new `sdr-knee-point` target →
  `results/sdr_knee_point_report.html`.

  **Infrastructure note, not this project's own bug**: the shared compute
  machine this session ran on became severely contended partway through
  (load average climbed from ~20 to ~57 on 16 cores over the course of the
  session, well beyond what this project's own two background jobs
  account for — other users' load, per `uptime`'s 9 logged-in users). The
  full 6-baseline S6-02 generation job and a Baseline-1-only from-cache
  provenance rebuild were both still running, multiple hours in, as of
  this entry. **Still open**: the actual headline figure with real
  numbers and a real knee point — blocked purely on that generation job
  finishing (or enough of it to be informative), not on anything left to
  build. Will be completed and reported in a later session update.

- **`src/compute_cost.py`** (new) — MACs and parameter counts per method,
  the computational-cost dimension this project hadn't reported yet.
  Matches Yaqub et al.'s own Table 5 (Params/GFLOPs/model size) and
  PROTOCOL.md's Edge-Enabled-paper deployment-hardware discussion
  (PYNQ-ZU FPGA, ~15 W), applied to all six separation baselines plus the
  Condition A classifier, on this dataset's real 60,000-sample (15s @
  4000 Hz) signal length. Independent of the sweep/classifier/knee-point
  work above — ran to completion immediately regardless of the
  contended-machine situation.

  **Two explicit precision tiers**: "exact" (literal matrix-multiply/
  conv-layer shapes from each method's own real code, or measured from a
  real forward pass) for bandpass, both NMF baselines, Conv-TasNet-lite
  (via forward hooks on every real `Conv1d`/`ConvTranspose1d` — no
  third-party FLOP-counting package is installed, so this was written
  directly rather than adding a new dependency), and the classifier (a
  real fitted fold's SVM, support-vector count read directly); "estimate"
  (standard textbook complexity formulas — reduced-SVD FLOPs for MSSA,
  ADMM elementwise-loop counts for EVMD) for the two methods with no
  closed-form matmul shape to count exactly.

  **Measured result** (per-mixture inference cost, ascending): bandpass
  2.4M MACs (0 params) < classifier 8.1M MACs (40 support vectors) <
  supervised NMF 219M MACs (7,710 frozen dictionary entries) < standard
  NMF 1.16G MACs (0 persisted params) < MSSA 1.20G MACs (estimate) <
  Conv-TasNet-lite 1.96G MACs (325,465 params) < EVMD 5.22G MACs
  (estimate) — roughly a 2,175x spread between the cheapest and most
  expensive method.

  Two findings worth keeping: (1) Conv-TasNet-lite's measured param count
  (325,465) independently reproduces the ~325K figure the 2026-08-25
  session already reported — a real cross-check across sessions, not a
  coincidence. (2) Standard NMF (Baseline 3) costs ~5.3x more per mixture
  than supervised NMF (Baseline 2) despite being the "simpler" ablation —
  because it's fully transductive (160 MU iterations updating both W and
  H fresh per mixture) vs. Baseline 2's 60 activation-only iterations
  against an already-frozen dictionary, giving the existing "B2 is
  inductive, B3 is transductive" fairness caveat (2026-08-22 session) a
  concrete cost number. EVMD's already-known wall-clock ranking (slowest
  baseline, ~15s/mixture) is independently confirmed by its MAC count too.

  `src/test/test_compute_cost.py` (new, 14 tests): every exact formula
  checked against a hand-computable case or independent recomputation
  from the same real constants; every estimate formula checked against
  its own stated inputs; Conv-TasNet's params cross-checked against a
  direct `model.parameters()` sum and the previously-reported ballpark;
  the classifier's support-vector count cross-checked against a freshly
  fitted model; and the two real orderings above (B3 costs more than B2,
  EVMD is the most expensive classical baseline) checked directly. All
  passing.

  `Makefile`: new `compute-cost` target → `results/compute_cost_report.html`.

- **`src/heart_classifier_cnn.py`** (new, S7-06) — the second classifier
  architecture for the C2 knee-point robustness check, no longer optional
  now that C2's knee point, not classification accuracy on its own, is
  the paper's headline result. The charter's "one architecture only, it's
  a measuring instrument" rule held when C2 was a secondary result; once
  the knee point IS the paper, a reviewer's first question is whether
  it's a property of separation quality or of the one architecture
  (`heart_classifier.py`'s MFCC+SVM) that measured it.

  **Architecture 2**: log-mel spectrogram (40 mel bins, same STFT window
  as Architecture 1) + a shallow CNN (2 conv blocks, global average
  pooling, one linear head — a few thousand parameters). Deliberately
  chosen for maximal architectural distance from Architecture 1 — a 2D
  time-frequency representation instead of pooled summary statistics, a
  gradient-trained model instead of a kernel method — since two flavors
  of SVM wouldn't isolate whether a disagreement is about separation or
  about one specific decision boundary. This is §5.3's own other named
  small-model option (its docstring names both "log-mel + shallow CNN"
  and "MFCC + SVM").

  **Backend contract**: `condition_b.py`, `sdr_accuracy_curve.py`, and
  `sdr_knee_point.py` all gained a `backend` module parameter (default
  `heart_classifier`) — a backend exposes `train_fold_classifiers(hs_df,
  n_folds)` and `predict_one(clf, y, sr)`, and both architectures satisfy
  this without sharing a class hierarchy, the same "same call contract,
  no shared base class" pattern the six separation baselines already use.
  `heart_classifier.py` gained the same two functions as public exports
  (`predict_one` new; `_aggregate_ci95` renamed to public `aggregate_ci95`)
  so both modules genuinely share one contract, not a contract only one
  of them documents.

  **First real result — a genuine, disclosed limitation, not glossed
  over**: Architecture 2's own Condition A accuracy is **30.0% ± 8.8%**
  (95% CI, 5-fold), far below Architecture 1's 58.0%, in this first
  configuration (no hyperparameter search). Its confusion matrix shows a
  collapse toward predicting **Normal** for 44/50 recordings (100% Normal
  recall, but Murmur/Extra Sound/Rhythm Disorder all routing
  overwhelmingly to Normal instead of their own class) — a *different*
  majority-attractor than Architecture 1's own pull toward Murmur (the
  earlier confusion-matrix session's finding). The two architectures
  don't just perform differently, they fail differently — itself
  informative for a robustness check, since it suggests genuinely
  distinct decision boundaries rather than two models converging on the
  same shortcut. This also directly confirms the overfitting risk
  `heart_classifier.py`'s own docstring predicted when it chose SVM over
  CNN for n=50 in the first place — tested here rather than dismissed,
  and the risk turned out real.

  **Practical consequence, stated plainly**: a knee-point disagreement
  measured against this first CNN configuration cannot yet cleanly
  distinguish "the knee is architecture-dependent" from "this CNN config
  isn't a reliable enough classifier to support the comparison" — both
  remain live possibilities. `sdr_knee_point.py`'s own report states this
  caveat automatically whenever Architecture 2's isolated accuracy falls
  below 40%, so a disagreement (once the full sweep is available) won't
  be over-read while this is still true.

  `src/test/test_heart_classifier_cnn.py` (new, 8 tests, using a
  `max_epochs` override for speed — mirrors `convtasnet.py`'s own
  testability pattern): feature-shape consistency across recordings, fit/
  predict round-trip, predicting before fitting raises, seeded
  reproducibility, the backend contract, and end-to-end cross-validation
  output shape. All passing.

  `test_condition_b.py` (+3 tests) and `test_sdr_accuracy_curve.py` (+2
  tests) each gained a `TestBackendParameter` class using a fake stub
  backend (always predicts a fixed label) to prove the `backend`
  parameter is genuinely threaded through rather than silently ignored,
  without paying for a real CNN training run in those test files.

  **`sdr_knee_point.py` gained the actual robustness-check machinery**:
  `run_pipeline_for_backend()` (the full curve pipeline for one backend),
  `present_baselines_for()` (factored out of `__main__`), and
  `compare_knee_points()` — per baseline, whether every architecture's
  knee_sdr falls within a stated tolerance (default 3 dB): `agrees=True/
  False` only when every architecture found a clean `"crossed"` knee;
  `agrees=None` (inconclusive, not a silent "no") when any side found no
  clean crossing to compare. `__main__` now runs both architectures and
  renders a combined comparison report.
  `test_sdr_knee_point.py` (+5 tests, `TestCompareKneePoints`): agreement
  within tolerance, disagreement beyond it, the inconclusive case,
  baseline-intersection behavior, and the at-least-two-backends guard.
  All passing.

  `Makefile`: new `heart-classifier-cnn` target →
  `results/heart_classifier_cnn_report.html`.

  **Still open**: the actual cross-architecture curve comparison with
  real SDR-sweep data — blocked on the same S6-02 generation as every
  other still-open item this session (system load climbed to ~88 on 16
  cores by the end of this session, almost entirely external — see the
  2026-08-27 sdr_knee_point.py entry above). Separately, whether
  Architecture 2's first configuration needs tuning (regularization, more
  epochs, a different learning rate, mini-batching instead of full-batch
  gradient descent) before its own curve is trustworthy enough to draw
  the robustness conclusion from — not attempted this session, consistent
  with this project's "first working configuration, disclosed, not
  polished" pattern for every other first-of-its-kind model
  (Baseline 6, this one).

- **`src/latency.py`** (new) — the empirical wall-clock/CPU-time
  counterpart to `compute_cost.py`'s analytic MACs/params table,
  completing this project's own version of Yaqub et al.'s Table 5
  (Params/GFLOPs/Model size/**Inference time**). "Desktop" contrasts with
  the Edge-Enabled paper's PYNQ-ZU FPGA deployment target (PROTOCOL.md
  §2) — measures this workstation's inference latency, no claim about
  edge hardware.

  **Uniform protocol**: one untimed warm-up call + `N_TRIALS` (fewer for
  EVMD specifically, given its own ~15s/call baseline cost) timed
  repetitions, median + IQR for both wall-clock and process CPU time, on
  one real 60,000-sample recording, identical code path for every method
  regardless of what else is running on the machine. One-time setup
  (NMF dictionary fitting on a small pool, Conv-TasNet-lite training with
  a reduced epoch count — inference speed doesn't depend on training
  quality) is fit once, untimed, outside the measured region.

  **Measured under severe, disclosed machine contention** — load average
  74.6 on 16 CPUs at measurement start (this project's own S6-02
  generation job plus, per `uptime`'s 9 logged-in users, unrelated load
  from other people on the shared machine). Recorded directly via
  `os.getloadavg()` and stated prominently in the report rather than
  hidden; the protocol stays uniform regardless (same code/reps/input per
  method), so relative rankings are still meaningful even though absolute
  wall-clock numbers are noisy upper bounds, not clean single-tenant
  measurements.

  **Real numbers** (median wall-clock / CPU time, ms, ascending
  wall-clock): Conv-TasNet-lite 23.9/5.9 < bandpass 36.0/5.6 < classifier
  63.0/199.4 < supervised NMF 5,200/6,179 < MSSA 12,910/23,538 < standard
  NMF 21,036/25,767 < EVMD 90,075/24,616 (n=3 reps, not 5, given its own
  cost).

  **The interesting finding this measurement exists to surface**: MACs
  do **not** predict latency here. Conv-TasNet-lite has the *lowest*
  measured wall-clock latency (23.9 ms) despite having the
  *second-highest* MAC count (1.96G, `compute_cost.py`) — ~820x more
  MACs than the bandpass filter, yet faster in wall-clock. This is an
  implementation-efficiency effect, not a computation-cost one: PyTorch's
  conv kernels are executed by heavily optimized, vectorized code, while
  this project's own NMF/MSSA/EVMD baselines are direct-from-the-paper
  Python+NumPy implementations (explicit MU-update loops, per-mode ADMM
  loops) with none of that low-level optimization. Concretely: EVMD has
  ~2.7x Conv-TasNet's MACs (5.22G vs. 1.96G) but ~3,770x its measured
  latency (90s vs. 24ms) — MACs and latency diverge by more than three
  orders of magnitude in relative ranking between these two methods. This
  is exactly why measuring both under `compute_cost.py` and `latency.py`
  separately is worth doing — a MACs table alone would have implied
  Conv-TasNet is the second-most-expensive method to deploy; the real
  answer is closer to the opposite.

  Also visible in the same numbers: for the three classical DSP methods
  that do share an implementation family (Baselines 2/3, both this
  project's own NMF code), the CPU-time ratio (Baseline 3 : Baseline 2 ≈
  25.8s : 6.2s ≈ 4.2x) is in the same ballpark as `compute_cost.py`'s
  independently-derived MACs ratio (~5.3x) — consistent cross-validation
  between the analytic and empirical measurements *within* one
  implementation family, even though it breaks down *across* families
  (classical DSP vs. PyTorch) for the reason above.

  `src/test/test_latency.py` (new, 7 tests): `time_calls()`'s protocol
  logic checked with fast, deterministic synthetic functions (correct
  call count across warm-up + trials, non-negative medians/IQRs, a
  known-duration `time.sleep()` case confirming the wall-clock/CPU-time
  distinction actually holds under real scheduling, tolerant of the
  session's own heavy contention rather than asserting a tight bound);
  two cheap real methods (bandpass, the classifier) exercised end-to-end.
  The four expensive methods share the exact same tested protocol
  function, so weren't re-exercised in the automated suite to avoid
  piling more load onto an already severely contended machine. All
  passing.

  `Makefile`: new `latency` target → `results/latency_report.html`.

## Done (2026-08-28 session)

- **`report/report.tex` substantially expanded** (requested directly: "very
  detail even a student can understand it... report why, what knowledge
  was used," not tied to a Notion ref) — grew from 1433 to 2431 lines.
  Two changes:

  1. **New `Section~2, "Background and Preliminaries"`**, inserted right
     after the Introduction: a from-scratch, student-level primer on
     every technical idea the rest of the report depends on and had
     previously assumed as background — blind source separation and why
     it's underdetermined; BSS Eval's SDR/SIR/SAR decomposition and why
     all three are reported instead of just SDR; NMF (non-negative
     factorization, multiplicative updates, supervised vs. blind); SSA
     (trajectory-matrix embedding, SVD, Hankelization); VMD (ADMM,
     narrow-band mode optimization); Conv-TasNet's convolutions,
     dilation, and encoder/separator/decoder design; $k$-fold
     cross-validation and data leakage (with this project's own leak-group
     split as the concrete example); classification metrics (accuracy's
     blind spot, macro-F1, confusion matrices); and compute-cost
     vocabulary (MACs, the Pareto front). Every later section now cross-
     references this one by name instead of re-deriving a concept inline.
  2. **New `Section~9, "Downstream Classification"` and `Section~10,
     "Computational Cost"`**, inserted before Related Work — the report
     previously stopped at Baseline 6 and the six-baseline results table,
     missing everything from S2-03 (the classifier) onward. These two new
     sections cover, in the same why-and-knowledge style as the rest of
     the (already detailed) report: the one-architecture-first decision
     and the Yaqub-scheme class-grouping decision; Architecture 1
     (MFCC+SVM) and its real Condition A result (58.0% ± 7.3% accuracy)
     with its confusion-matrix failure mode explained; the controlled
     degradation scheme (including the real uniform-alpha-grid problem
     found and the target-SDR root-finder fix); the SDR sweep generator;
     Condition B's weight-sharing decision and its one real result so far
     (Baseline 1: 50.0% isolated → 41.7% separated, not significant);
     the knee-point definition; the second-architecture robustness check
     (Architecture 2's real 30.0% accuracy and its different,
     Normal-collapse failure mode, with the practical caveat this implies
     for reading a future knee-point disagreement); and the real MACs,
     latency (with its own machine-load caveat and the MACs-vs-latency
     divergence finding), and SDR-vs-compute-plane methodology.

  Abstract, introduction roadmap, and the Limitations/Future-Work section
  were all updated to match — the stale "no classification stage yet"
  bullet was replaced with an accurate statement of what's built vs. what
  generation is still pending (the full six-baseline sweep and the
  resulting knee-point curve), and a new bullet flags Architecture 2's
  own need for tuning before its curve is trustworthy.

  **Verified, not assumed** (no LaTeX toolchain is available in this
  environment, so the PDF was not regenerated — same limitation the
  report's own text already discloses): brace and `\begin`/`\end` balance
  checked programmatically (1149/1149, 54/54), every `\label`/`\ref` pair
  checked for duplicates and dangling references (none found), and the
  newly added text scanned for unescaped `%`/stray `_` outside
  `\texttt{}`/math mode (none found beyond two pre-existing, legitimate
  line-continuation `%`s).

- **`src/sdr_compute_plane.py`** (new) — completed the SDR-vs-compute
  plane: the pending Baseline 6 native-additive SDR re-measurement
  finished (5.15 ± 2.74 dB, 95% CI, n=36, pooled convention), and
  reproduced the already-published 2026-08-25 figure exactly, a genuine
  independent cross-check rather than a coincidence of rounding
  (`baseline6_report.py` turns out to already use the same pooled-CI
  convention as `first_sdr_table.py`, not the fold-level mean±std
  convention some of the other PROTOCOL.md tables use). Hardcoded into
  `SDR_HEART_DB` alongside Baselines 1–5's own already-published numbers
  now that it's confirmed, rather than re-measured on every run.

  **Real result** (`results/sdr_compute_plane_report.html` +
  `results/plots/sdr_compute_plane.png`, sent to the user): on the MACs
  plane, bandpass (Baseline 1) dominates every other baseline outright —
  highest SDR *and* lowest MACs by nearly three orders of magnitude, so
  every other method is Pareto-dominated. On the measured-latency plane,
  that conclusion does **not** survive: bandpass and Conv-TasNet-lite
  (Baseline 6) are *both* non-dominated (bandpass has higher SDR,
  Conv-TasNet-lite has lower latency — a genuine trade-off), while every
  other baseline remains dominated on both planes. This is the concrete
  worked example of `latency.py`'s own MACs-vs-latency divergence finding:
  a plane built from MACs alone would have hidden Conv-TasNet-lite's real
  competitiveness entirely.

  `report/report.tex` §10.3 updated with the completed table and finding
  (previously stated as pending); 6 logic tests (`test_sdr_compute_plane.py`)
  already passing from before this completion.

## Done (2026-09-09 session)

- **`firmware/` — Baseline 1 ported to a microcontroller and measured on
  silicon** (not a Notion ref; the charter scopes C1/C2 only, and embedded
  deployment was never a chartered contribution — this is a deliberate
  extension, flagged as such rather than folded in as if it had been
  planned). Five commits: the port itself, two serial-usability fixes, the
  additive-clip fix and the tolerance fix below.

  **Target**: Arduino Nano 33 BLE Sense (nRF52840, 64 MHz Cortex-M4F, 256 kB
  RAM), CMSIS-DSP biquads. An ESP32-S3 target builds from the same core
  (ESP-DSP backend) but has not been run on hardware yet. Baseline 1 is the
  right first candidate not merely for being simplest — `sdr_compute_plane.py`
  puts it on the Pareto front of *both* compute planes, so it is one of the
  two methods a deployment decision would actually consider.

  **The one intentional divergence**: causal single-pass `sosfilt` instead of
  the Python baseline's zero-phase `sosfiltfilt`. Forced, not preferred — a
  15 s int16 clip is 120 kB (46 % of total RAM) and `filtfilt` needs a
  240 kB float32 working copy on top, on a 256 kB part. Measured cost over
  all 145 Mix.csv rows (`causal_vs_zerophase.py`, same coefficients, same BSS
  Eval, only the filter direction changed): SDR unchanged within ±0.2 dB,
  SIR −0.13/−0.50 dB. SAR *appears* to improve by 5–7 dB, which is a metric
  artifact and is documented as one: `mir_eval`'s projection fits a causal
  512-tap distortion filter, so a causal IIR's phase response is absorbed as
  allowed distortion while a zero-phase filter's acausal response is not.

  **Verification chain, four stages** — three need no hardware: portable C vs
  scipy on golden vectors (plus block sizes 1/7/64/333 bit-identical, which
  is what proves streaming state is carried); wire protocol against an
  emulated board over a pty; then on silicon, `*-selftest` (vendor kernel vs
  a 1024-sample flash-resident reference) before `*-bench` (full clips). The
  ordering is load-bearing: a wrong vendor kernel makes every bench number
  wrong in a way that reads as a filter-design problem.

  **The bench firmware has no microphone by design** — it replays real
  HLS-CMDS mixtures compiled into flash, bit-identical to what the Python
  baseline filters (the 1/32768 scaling reproduces `librosa.load()` exactly,
  verified), so board output is diffable against scipy sample by sample.

  **Real results, first hardware run**: selftest `5.3e-8` heart / `1.5e-8`
  lung on the golden slice. Bench on three additive mixtures — every SDR
  agrees with scipy to the last printed digit across all six clip/source
  measurements; **62.4 us/sample, 4.0x faster than real time**. That is the
  deployment-relevant number: this band split runs as a live stream on a
  64 MHz part with 4x headroom, and nothing in its RAM footprint scales with
  recording length.

- **`embed_clips.py` was embedding non-additive clips** (found while reading
  the first hardware run's output, not by a test). It selected with
  `mix_df.head(count)` → M0001–M0003, none of which is additive (§5.1's
  36-row finding), so `run_on_device.py`'s SDR column was measuring the
  dataset's 109-row defect rather than the separator. Selection now draws
  from the additive subset, computed from `verify_additive_triplets()` rather
  than hardcoded so it tracks the dataset; a non-additive `--ids` is refused
  unless `--any` is passed, and the generated header records which mode
  produced it. Board-vs-scipy was unaffected throughout — it is a numerical
  diff, not a separation metric.

- **The accuracy criterion was calibrated on the wrong amplitude regime.**
  The first run on additive clips reported FAIL at `1.045e-05` against an
  absolute `1e-5` tolerance — on a board whose self-test had just passed at
  `5.3e-8` and whose SDR matched scipy to four significant figures. The
  tolerance was wrong, not the kernel: the board computes in float32 against
  a float64 reference, and an IIR accumulates rounding through its own state,
  so the gap scales with amplitude. `1e-5` had been calibrated on clips
  peaking near 0.05; the 36 additive rows are peak-normalised to full scale
  (that normalisation *is* the clipping fingerprint identifying them), ~20x
  louder. Decisive check: scipy itself in float32 on M0112 deviates by
  `1.114e-05`, **more** than the board's `1.045e-05`.

  Fixed by measuring the floor per clip instead of assuming it — filter the
  same samples with scipy in float32, hold the board to 4x that. Observed
  ratios on hardware: 0.51–1.57. Injected faults for calibration: 0.01 % gain
  error → 14.3, one dropped sample → 15 691, sign inversion → 296 503. The
  threshold sits in the gap with ~9x margin above the real board.

- **`report/report.tex`**: new §11 "Embedded Deployment: Baseline 1 on a
  Microcontroller" (causal-filtering decision + measured cost, the
  four-stage verification chain, on-silicon results, and the tolerance
  finding written up as a general trap). §10.2 gained an explicit scope
  paragraph — its desktop figures are a one-machine cross-method ranking,
  not an embedded-cost claim, and are superseded for Baseline 1 alone.
  `src/latency.py`'s docstring and report dek carried a "no claim is made
  about edge-hardware latency" line that had become false for that method;
  both now state the exception and point at the on-device figure.

## Open / next up (as of 2026-09-09)

- **ESP32-S3 hardware run** — `esp32s3-selftest` then `esp32s3-bench`; both
  build clean but the ESP-DSP kernel has never executed. Needs the board.
- **S1-14**: confirm the descriptor's own stated sample rate with a page
  reference. Last item blocking `report/dataset_audit_comment.tex`.
- Carried over from the 2026-08-28 session: S6-02 sweep generation, S6-04
  knee-point write-up, S7-06 Architecture 2 tuning.

## Done (2026-09-09, later session)

- **`src/mcu_feasibility.py`** (new) — answers the question the Baseline 1
  port raised: is there anything better that would also fit? Models peak
  working RAM, flash for parameters, and compute time for all six baselines
  against both targets, every figure derived from the method's own imported
  constants. **The discriminator turns out not to be MACs but whether a
  method can be written as a stream at all.** Baseline 2 is the only
  non-trivial one that can (9.1 kB RAM, dictionary in flash); Baselines 4 and
  5 are batch *by construction* — SSA's diagonal averaging and VMD's ADMM
  both reference the whole signal every iteration — and overrun RAM by 167x
  and 71x. Baseline 6 is batch only *as written* (9.2 MB of activations from
  encoding the whole clip at once; its int8 weights are 318 kB and fit flash
  fine), so a chunked/causal variant would fit — a model redesign, not a
  deployment step.

  int8 rescues nothing: /4 leaves B3 at 1.2x over, B6 at 10.7x, B5 at 17.8x,
  B4 at 41.6x. And for most it is not an applicable representation anyway —
  EVMD_TOL is 1e-6, four orders below int8's ~4e-3 resolution, so the ADMM
  could not detect its own convergence; MSSA selects on a 2 % eigenvalue
  threshold across singular values spanning orders of magnitude; NMF's MU
  updates carry a 1e-12 epsilon floor.

  **The negative result worth stating**: on the native additive subset the
  only method beating Baseline 1 on either source is Baseline 4 on lung
  (+1.34 dB), and Baseline 4 is the least feasible entry. Everything that
  could run scores worse on both sources. The cheapest method is also the
  best-performing one here, which is why sdr_compute_plane.py already had it
  on the Pareto front of both planes.

- **The Cortex-M4F's FPU had never been enabled** — found because
  mcu_feasibility.py predicted ~400x real time against the measured 4.0x, and
  a 100x gap was too large to be modelling slack. The arduino-mbed builder
  appends `-nostdlib -mfloat-abi=soft` after the framework's own `softfp`,
  GCC honours the last occurrence, and every float operation was compiling to
  `__aeabi_*` software-float calls: zero VFP instructions in the disassembly
  of both hls_filter.c.o and CMSIS-DSP's biquad kernel. `build_flags` does not
  fix it (PlatformIO emits those *before* the trailing group — confirmed by
  re-reading the compile line rather than assuming the override worked);
  `build_unflags` does.

  **Re-measured on the same board: 3.47 us/sample, 72x real time — an 18x
  speedup** from a build flag, not an optimisation. Numerics unchanged and
  still passing (self-test 6.5e-8 / 1.7e-8, marginally different from the
  soft-float run because VFP and the soft-float library round intermediates
  differently; both at the float32 floor, ~150x inside tolerance). Flash also
  dropped 1 248 B.

  Model now calibrates properly: 400x predicted vs 72x measured = 5.6x
  optimism, the right order for real code paying for memory traffic. Applied
  to Baseline 2 that would mean 0.8x real time rather than 4.4x, so its
  margin is flagged as uncertain rather than reported as comfortable.

- **`report/report.tex`** — Section 10 gained two subsections: *Why the other
  five baselines do not run here* (the feasibility table, the batch-vs-
  streaming argument, the int8 analysis, and the negative result) and *A
  plausible measurement that was wrong by 18x* (the FPU finding, written up
  for its two transferable lessons: a plausible performance number has not
  been validated by looking plausible, and numerical verification says
  nothing about performance verification — the self-test passed correctly
  both before and after, since software floating point computes the right
  answer, only slowly). On-device table and timing updated throughout.
  43 pp., compiles clean.

## Open / next up (updated 2026-09-09)

- **Port Baseline 2** — the one feasible non-trivial method. Value is
  demonstrating the verification chain generalises to FFT + iterative matrix
  updates, not improving separation quality (its SDR is worse than B1's).
  Real-time margin needs measuring, not assuming: see the 5.6x calibration.
- **ESP32-S3 hardware run** — `esp32s3-selftest` then `esp32s3-bench`; both
  build clean, ESP-DSP kernel has never executed. Needs the board.
- **Re-measure the embedded clips' representativeness** — the three embedded
  rows are not a representative sample on the lung side: M0111 is the
  0th-percentile lung row of all 36 (−12.43 dB against a 4.09 dB mean). The
  bench's job is fidelity and timing, not SDR estimation, so this does not
  invalidate anything, but a reader seeing −12.43 could misread it.
- Carried over: S6-02 sweep generation, S6-04 knee-point write-up, S7-06
  Architecture 2 tuning.
