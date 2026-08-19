# Protocol Memo: Separated vs. Isolated Accuracy on HLS-CMDS

Status: draft — open items flagged inline. Date: 2026-08-17.

**2026-08-18 update:** Reviewed in TEEP2026_Sprint0_Review (Satya Adhiyaksa,
18 Aug 2026). Split, metrics wrapper, and test suite confirmed strong as-is.
One blocking finding: the dataset copy this memo and `src/baselines.py`'s
results were built against (`HLS_CMDS/`, sourced from
github.com/Torabiy/HLS-CMDS per S0-06/S1-01, not the Mendeley release the
charter specified) has 109/145 Mix.csv rows whose mixed recording is
acoustically unrelated to its named heart/lung sources — confirmed
independently both by this project's own clipping-based audio-quality audit
(§5.2) and by a direct additivity test (see `verify_additive_triplets()` in
`load_dataset.py`, added in response to the review). **All separation-quality
numbers in this memo and in `report/report.pdf` are provisional pending a
re-run on the Mendeley copy** — see §8 for the full list of items the review
raised and their status.

## 1. Background

HLS-CMDS (Torabi, Shirani & Reilly, *IEEE Data Descriptions*,
doi:10.1109/IEEEDATA.2025.3566012) is the dataset this project builds on: heart
and lung sounds recorded from a CAE Juno manikin across six chest landmarks
(Apex, RUSB, LUSB, LLSB, RC, LC), released as both **isolated** heart/lung
recordings and **mixed** heart+lung recordings with the isolated components
that built each mixture included as ground truth. That dual structure — same
dataset, same underlying recordings, available both isolated and mixed — is
what makes the comparison in this memo possible at all.

## 2. Related work

Six papers reviewed, organized by what each one actually measures:

| Paper | Dataset | Separation method | Separation metrics reported | Downstream task | Reported result | Split |
|---|---|---|---|---|---|---|
| HLS-CMDS descriptor (Torabi et al.) | — (source) | — | — | — | — | — |
| AI-Driven Cardiorespiratory Signal Processing (LingoNMF / PL-NMF) [aidriven] | HLS-CMDS | PL-NMF (parallel multi-layer NMF, heart/lung-tuned) + LLM-derived fundamental frequency | not reported (paired t-test on an unspecified quality proxy) | clustering + anomaly detection, **not** accuracy | "statistically significant gains," p<0.01 | not stated |
| Spectrotemporal Deep Learning for Heart Sound Classification [spectrotemporal] | PhysioNet 2016 | not fully characterized here (needs a primary-source read) — includes a computational-separation experiment, not raw-PCG-only as previously logged | n/a | binary normal/abnormal classification, **including on separated audio** | **Experiment 4: accuracy 89% (clean) → 41% (computationally separated) — per TEEP2026_Sprint0_Review, this is the single result this whole project exists to explain; not yet independently confirmed from the primary source here** | not stated |
| Respiratory Disease Classification (NMF-enhanced log-mel + CRNN) [nmfcrnn] | **Conflicting**: this memo previously logged ICBHI 2017 + Fraiwan CWLS; the project charter instead has Han et al. learning dictionaries from HLS-CMDS isolated recordings (and this is the stated source of the Baseline 2 (`src/baselines.py`) hyperparameters — Kr/Ki dictionary ranks, KL-divergence MU-NMF, 100/60 iterations). **Unresolved — needs primary-source read**, per review. | NMF-based respiratory enhancement | **none** (no SNR/SDR/SI-SDR/correlation/spectral-distortion/error-energy) | multi-class respiratory disease classification | 96.14±0.50% acc, 94.05±1.21 Macro-F1; denoised-ablation Macro-F1 84.98±22.0 over 10 seeds | **recording-level**, not subject-level (authors' own flagged limitation) |
| Edge-Enabled Portable Lung-Sound Classifier (ESP32 + PYNQ-ZU) [edgelung] | **HLS-CMDS** | proposed lung isolation from mixtures | **none** — no SNR, SDR, correlation, or spectral-error figure | lung sound classification | not detailed / separation step "reports no result" | not stated |
| Performance Measurement in Blind Audio Source Separation | — (methods paper) | — | defines SDR/SIR/SAR (BSS Eval) | n/a | n/a | n/a |
| Cardiorespiratory Sound Separation Using SSA [ssa] | not stated in summary — **still needs confirming** (citation now known; dataset still open) | two-stage Singular Spectrum Analysis (L=50) | SDR, STOI, ground-truth correlation, vs. 5 baselines (a Butterworth bandpass baseline among them is the likely source of the 5.7 dB order-of-magnitude figure referenced in `src/baselines.py`'s review) | **none** — no classification | respiratory SDR 5.34 dB (MSSA) vs. 4.64 dB (SSA); cardiac 26.44 dB both stages; correlation 80.5%/99.2% (MSSA) vs. 10.2%/50.0% (NMF baseline) | n/a |

**References** (from the Notion Reading List, via TEEP2026_Sprint0_Review — not yet cross-checked against the primary sources directly in this repo):
- `[aidriven]` Torabi, PhD thesis, McMaster, 2025.
- `[spectrotemporal]` Yaqub et al., *CMES*, 2025. doi:10.32604/cmes.2025.071571.
- `[nmfcrnn]` Han, Quan, Matuszewski & Corbett, *Sensors*, 2026. doi:10.3390/s26134268.
- `[edgelung]` Puneet, Shankar, Koluguri & Srivastava, *IEEE BioCAS*, 2025.
- `[ssa]` Han & Quan, *IEEE ICSP*, 2025. doi:10.1109/ICSPS66615.2025.11347745.

One cross-check worth noting: the Edge-Enabled paper's stated HLS-CMDS lung
class counts (Normal 28, Wheezing 28, Pleural Rub 25, Rhonchi 23, Fine
Crackles 22, Coarse Crackles 19) match `load_dataset.py`'s `PAPER_LUNG_COUNTS`
`mix_zip` column exactly — i.e. they're citing the *total-in-Mix.zip* count
from the HLS-CMDS descriptor paper's Table 2, not the isolated-LS.csv count
(which is smaller: `own_zip` is 5–12 per class). Worth being deliberate about
which of these two counts this memo's own tables use, since conflating them
is an easy mistake to reproduce.

Two load-bearing details from the BSS Eval methods paper, worth carrying into
this study's design rather than treating as background:
- **SAR catches what accuracy can't.** A classifier can stay accurate on a
  separated signal that's been badly distorted, if the distortion doesn't
  cross a decision boundary — SAR is sensitive to distortion accuracy alone
  won't reveal, which is the mechanistic reason to report both.
- **SDR is not perceptual and is gameable.** A slightly low-pass-filtered
  estimate can drive SDR toward +inf under a time-varying filter without
  the estimate actually sounding, or classifying, any better. SDR alone is
  not sufficient evidence of separation quality — SIR and SAR have to be
  reported alongside it, which is exactly what `metrics.py` already does.
- The paper also notes that for HLS-CMDS specifically, mixtures are heart +
  lung only with **no independent sensor-noise reference**, so the
  BSS Eval "noise" term is effectively zero and SAR captures pure
  algorithmic artifact, not real-world sensor/environmental noise. That
  bounds what this study can claim: it measures separation-then-classification
  degradation in a clean-mixture setting, not robustness to real acquisition
  noise (relevant given the Edge-Enabled paper's real ESP32 hardware).

## 3. The gap

**The paper this project most directly answers to is Yaqub et al.
(`[spectrotemporal]`), not the HLS-CMDS-specific papers below it.** Per
TEEP2026_Sprint0_Review, their experiment 4 reports a classification
accuracy collapse from 89% (clean heart sounds) to 41% (computationally
separated heart sounds) — the exact separated-vs-isolated accuracy gap this
project's research question (§4) is built to measure, on a different
dataset (PhysioNet 2016) and, per the review, without the leakage-safe
split this project's own §5.1 was built to guarantee. This memo previously
filed Yaqub under "no separation, binary classification" and missed this
entirely — **not yet independently re-confirmed from the primary source
here**, flagged as an open item in §8.

Of the six HLS-CMDS-adjacent papers, exactly one — **Edge-Enabled Portable
Lung-Sound Classifier** — evaluates on HLS-CMDS *and* runs a classifier on
mixture-separated audio. It reports a classification result but **zero**
separation-quality metrics, and (per the summary reviewed) does not compare
that accuracy against classifying the same recordings' isolated,
un-mixed ground truth. The SSA paper reports proper separation metrics
(SDR/STOI/correlation) but no classification task at all. The NMF+CRNN paper
reports classification accuracy with real numbers but on a dataset that is
itself now in question (see the `[nmfcrnn]` row above), with no separation
metrics, and its own authors flag a recording-level (leakage-prone) split.

No paper reviewed does all of:
1. reports separation quality via SDR/SIR/SAR,
2. reports classification accuracy on the separated outputs,
3. reports classification accuracy on the *isolated ground-truth* recordings
   for the *same* held-out cases, as a paired comparison, and
4. does so under a split where no recording used to fit the separation
   method (or the classifier) leaks into evaluation.

(4) is not a hypothetical concern for HLS-CMDS: our own content-hash audit of
this dataset (`src/split.py`) found that Mix.csv's 145 rows draw from only 89
distinct heart and 74 distinct lung recordings, roughly a third of which are
byte-identical to recordings also listed standalone in HS.csv/LS.csv. A
recording-level split — the exact mistake the NMF+CRNN paper's authors flag
in their own limitations section — would leak ground truth into training here
too.

## 4. Research question

**Superseded by the charter's C2 framing** (per TEEP2026_Sprint0_Review,
§ SCOPE — flagged as a drift from the chartered question, kept here
struck through rather than deleted so the correction is traceable):

~~Does separating heart/lung sound from a mixture, before classification,
measurably reduce normal/abnormal detection accuracy relative to classifying
the isolated recording directly — and does that reduction track separation
quality (SDR/SIR/SAR)?~~

The charter's C2 asks the sharper version: **sweep separation quality
continuously and find the knee point — the SDR below which separation
actively hurts classification accuracy, relative to not separating at
all.** The isolated-vs-separated binary comparison above is the two
endpoints of that sweep (perfect separation and no separation), not the
full question; Yaqub et al.'s 89%→41% result (§3) is one data point on the
same curve this project's Condition B (§5.3) is meant to trace out
continuously. Practical implication for §5.3: classification needs to run
across a *range* of separation quality (e.g. multiple separation methods
and/or degraded/blended versions of a method's output spanning a range of
SDR), not just the two endpoints "isolated ground truth" and "one
separation method's output," for a knee point to be identifiable at all.

## 5. Method

### 5.1 Split (built — `src/split.py`)

5-fold split at the leak-group level: mix rows are connected whenever they
share a heart or lung recording (by audio content, not ID string), and whole
connected components are assigned to folds so no recording crosses a fold
boundary. `dictionary_pool(hs_df, ls_df, mix_df, held_out_fold=k)` returns the
HS.csv/LS.csv rows safe to fit anything on for fold `k`. This split governs
**both** the separation method's fitting and the classifier's training —
not just evaluation.

### 5.2 Separation quality (built — `src/metrics.py`, `src/eval_harness.py`)

For each fold, `eval_harness.cross_validate(fit_and_separate_fn)` fits the
separation method on that fold's dictionary pool, separates the held-out
mixtures, and scores SDR/SIR/SAR against the true heart/lung components via
`metrics.evaluate_dataset`. The harness is method-agnostic
(`fit_and_separate_fn` is a plug-in), so which method to run is a per-run
choice, not a blocker.

**Baseline 1 — simple bandpass filtering (implemented, `src/baselines.py`).**
The zero-training sanity floor: any learned method that doesn't beat this
has a bug, not just a hard problem. Cutoffs were derived from this
dataset's own isolated recordings rather than borrowed from literature
defaults — averaged Welch PSD over all of HS.csv and LS.csv shows heart
energy dominant below ~150-200 Hz, lung dominant from ~200 Hz up to
~700-800 Hz, both falling into the noise floor above ~1 kHz (at this
dataset's 4000 Hz sample rate). Chosen bands: heart 20-200 Hz, lung
150-1000 Hz (the 150-200 Hz overlap is intentional — both sources have real
energy there, which is exactly the "spectral overlap" limitation a bandpass
baseline is supposed to expose, not an implementation bug). Zero-phase
Butterworth (`scipy.signal.sosfiltfilt`), fixed for every fold since it has
no learned parameters.

5-fold result (`python src/baselines.py`):

| source | SDR (dB) | SIR (dB) | SAR (dB) |
|---|---|---|---|
| heart | -12.65 ± 1.40 | 4.07 ± 1.16 | -8.10 ± 1.44 |
| lung  | -13.83 ± 1.12 | 1.73 ± 0.29 | -9.27 ± 1.42 |

Compare against the no-separation reference (`mixed` returned as both
estimates, via `eval_harness.py`'s own demo baseline): heart SDR -13.68±1.26
/ SIR 1.71±0.93 / SAR 6.46±3.71; lung SDR -15.89±0.84 / SIR -1.55±0.91 / SAR
6.46±3.71 (SAR is shared because there's no processing to tell heart's and
lung's artifacts apart when nothing was done). The pattern is exactly what
the theory predicts: bandpass buys real SIR (it does separate on frequency
grounds — both sources' SIR improves) at the cost of SAR collapsing (filter
ringing and destroyed harmonics), so SDR — which folds both together — only
nets a modest gain over doing nothing at all. That gap between "better SIR"
and "worse SAR" is the concrete demonstration of the BSS Eval paper's point
in §2: SDR alone would have made bandpass look like a bigger win than it is.

**Baseline 2 — supervised NMF, reproducing `[nmfcrnn]` (Han, Quan,
Matuszewski & Corbett, *Sensors*, 2026, doi:10.3390/s26134268 — identified
via TEEP2026_Sprint0_Review; not yet independently read from the primary
source) (implemented, `src/baselines.py`).** Two fixed dictionaries, one
per source, learned from isolated H/L recordings via KL-divergence
multiplicative-update NMF, then frozen and used to solve for per-mixture
activations (the classic supervised-NMF separation recipe, Smaragdis
2007). Hyperparameters as given during this project's chat-based handoff
(**not yet cross-checked against the paper directly — correction from an
earlier draft of this section, which overstated that check**): Kr = lung
dictionary rank = 20, Ki = heart dictionary rank = 10; 100 MU iterations to
fit each dictionary, 60 MU iterations to solve activations per mixture with
the dictionaries frozen. Separation is a soft (Wiener-style) mask on the
mixture's complex STFT built from the two sources' reconstructed
magnitudes. Note the still-open dataset discrepancy in the `[nmfcrnn]` row
of §2's table (ICBHI+Fraiwan vs. HLS-CMDS isolated recordings) — worth
resolving since it bears on whether "isolated H/L recordings" here means
this project's own HS.csv/LS.csv or a different corpus entirely.

**Leakage trap.** A third of HS.csv/LS.csv's recordings are byte-identical
to a heart/lung component of some Mix.csv row — that row's own ground
truth (§5.1/`split.py`). A naive reproduction that fits the dictionaries
on "all of HS.csv/LS.csv" therefore lets a mixture's own ground-truth
source into its separation dictionary. This implementation only ever sees
`hs_allowed`/`ls_allowed` — `eval_harness.cross_validate`'s fold-safe
pool, same triplet-level split as Baseline 1 — so the number below is
lower than a naive reproduction would report. That lower number is the
correct one; a higher one would mean leakage, not a better model.

5-fold result (`python src/baselines.py`):

| source | SDR (dB) | SIR (dB) | SAR (dB) |
|---|---|---|---|
| heart | -12.82 ± 1.31 | 4.22 ± 1.63 | -9.58 ± 0.90 |
| lung  | -15.09 ± 0.98 | 0.32 ± 0.58 | -9.23 ± 1.42 |

Compare against Baseline 1 (bandpass): heart SDR -12.65±1.40 / SIR
4.07±1.16 / SAR -8.10±1.44; lung SDR -13.83±1.12 / SIR 1.73±0.29 / SAR
-9.27±1.42. Leakage-safe supervised NMF comes in **essentially tied with
the trivial bandpass filter on heart** (SDR delta well within one fold-std
of either) and **~1.3 dB worse on lung** — not the clear win over a
zero-training baseline a paper's reported numbers would suggest. This
tracks with a sanity check run against a single mixture row using an
*oracle* mask built from that row's own true heart/lung spectrograms
(cheating on purpose, just to bound the method): even the oracle mask only
modestly beat the no-separation reference on that row, so the ceiling for
mask-based separation on this dataset's real (not synthetically summed)
mixture recordings looks lower than the bandpass-vs-no-separation
comparison alone would suggest — worth keeping in mind before trusting any
future method's reported gain at face value.

**Still open:** SSA (best-reported respiratory SDR among the papers
reviewed, 5.34 dB) or PL-NMF/LingoNMF if reproducible from the AI-Driven
paper's description, as a third separation method to compare against
both baselines.

### 5.3 Classification (new — not yet built)

Two conditions, same classifier, same fold split:

- **Condition A — Isolated (oracle/upper bound):** train and evaluate on
  HS.csv/LS.csv recordings only, respecting the same fold membership as 5.1
  (a recording's fold assignment is fixed once, whether it's evaluated via
  its isolated file or via a mixture that reuses it).
- **Condition B — Separated:** evaluate the *same trained classifier* (or a
  version trained under the identical split — needs deciding, see below) on
  the separated `heart_est`/`lung_est` outputs from 5.2, for the same
  held-out fold.

Report accuracy **and Macro-F1** for both conditions, per fold and
aggregated with mean±std across folds (matching `eval_harness`'s existing
aggregation pattern). Macro-F1 specifically because HLS-CMDS classes are
small and imbalanced (e.g. per our own `statistics/audio_quality.py` output,
heart AV Block n=3, S4 n=2 in the isolated set) — the NMF+CRNN paper's
84.98±22.0 Macro-F1 over 10 seeds is a concrete warning about what happens
when accuracy alone hides per-class collapse on exactly this kind of small,
imbalanced label set.

**Open items, need a decision before this section can be implemented:**
- Which classifier architecture. No confirmed HLS-CMDS classification
  baseline exists among the papers reviewed to standardize on — the closest
  in spirit is the Edge-Enabled paper's CNN, but it doesn't report enough
  detail (per the summary above) to reproduce as a baseline directly.
  Recommendation, absent other constraints: something small and
  well-understood (e.g. log-mel spectrogram + a shallow CNN, or MFCC + SVM)
  so the *comparison* is the contribution, not the classifier's novelty.
- Whether Condition B's classifier is literally the same trained weights as
  Condition A, or retrained on separated training-fold audio. The former
  isolates "does separation hurt inference" cleanly; the latter asks "can a
  classifier adapt to separation artifacts" — different questions, pick one
  as primary and note the other as a follow-up.

### 5.4 Comparison and significance

Primary result: `accuracy_isolated - accuracy_separated` (and the same for
Macro-F1), paired by fold, with a paired significance test across folds
(the AI-Driven paper's paired-t-test-at-p<0.01 precedent, adapted here to
5 folds rather than per-example pairs — note with only 5 folds, treat the
significance test as indicative, not decisive, and report the raw per-fold
deltas alongside it).

Secondary analysis: within Condition B, correlate per-fold (or, if volume
allows, per-recording) SDR/SIR/SAR against classification correctness — this
is the check for the BSS Eval paper's "SAR catches what accuracy can't"
claim: does a fold with high accuracy but low SAR exist, i.e. does the
classifier stay right despite bad artifacts, or does artifact severity
predict misclassification even when aggregate accuracy looks fine?

## 6. Threats to validity

- **Leakage** — addressed by 5.1; explicitly the mistake being avoided
  relative to the NMF+CRNN paper's recording-level split.
- **SDR is gameable, not perceptual** — never report SDR alone; SIR/SAR and
  the classification-accuracy comparison itself are the actual evidence of
  separation quality that matters here, per the BSS Eval paper's own caveat.
- **Small fold count (5) and small/imbalanced classes** — report per-fold
  deltas, not just the aggregate, and Macro-F1 alongside accuracy; treat any
  significance test as indicative given n=5 folds.
- **No sensor-noise term** — HLS-CMDS mixtures are heart+lung only; this
  study's SAR reflects algorithmic artifact from separation, not resilience
  to real acquisition noise (e.g. the Edge-Enabled paper's real ESP32
  hardware pipeline). Out of scope here; worth flagging as future work.
- **Separation-method choice is a confound** — the accuracy gap measured is
  specific to whichever separation method is chosen in 5.2; framing the
  result as "separation costs X% accuracy" should specify the method, not
  claim it generalizes to all separation approaches without re-running.

## 7. Deliverables

1. Separation-quality table (SDR/SIR/SAR, mean±std across folds) — mechanism
   already built.
2. Classification-accuracy table, Conditions A vs. B, per fold + aggregate.
3. Paired delta + significance test.
4. Separation-quality-vs-accuracy correlation analysis.
5. This memo, updated with the resolved open items (classifier choice,
   separation method choice, weight-sharing decision) once decided.

## 8. Open items (blocking implementation, not this memo)

**Blocking, per TEEP2026_Sprint0_Review — do first:**
- **Re-download HLS-CMDS from Mendeley**
  (data.mendeley.com/datasets/8972jxbpmp/3) into a directory outside this
  git repo; record version/date/URL/per-file SHA-256 in `README.md`.
  Attempted programmatically this session — Mendeley's actual file bytes
  sit behind a JS-driven download flow (public API metadata/linkset
  endpoints are reachable, but the file-content endpoints require an
  authenticated session, confirmed via direct probing) that plain
  HTTP tooling here can't complete. **Needs a manual download** (or a
  browser-automation session with access to wherever the files should
  land) before this item and the two below it can close.
- **Re-run `verify_additive_triplets()` on the Mendeley copy** and report
  how many of 145 rows are additive, plus the sample rate (charter/descriptor
  paper say 22,050 Hz; the current GitHub copy is 4000 Hz — a 5.5× mismatch
  that, per the review, is itself evidence the GitHub copy is a derived,
  downsampled distribution rather than the released dataset). This is the
  input to the charter's C1 rescoping decision and gates kickoff planning.
- **Re-issue Table 5 / `report/report.pdf`'s results on the confirmed-valid
  subset**, medians alongside means (see the statistics note below), once
  the above two land.

**Done this session:**
- `verify_additive_triplets()` added to `load_dataset.py` (+ `test_load_dataset.py`,
  6 tests) — tests mixed ≈ a·(heart+lung) per row via least-squares gain fit
  and relative residual, independent of `verify_mix_triplets()`'s
  file-existence/format-only checks. On the current GitHub copy: 36/145
  additive, matching both TEEP2026_Sprint0_Review's independent finding and
  this project's own clipping audit (§5.2) exactly (same 36 IDs: M0087,
  M0111–M0145).
- Five `[Author(s) needed]` citations filled in (§2), sourced from the
  review — not yet independently cross-checked against the primary sources.
- Yaqub et al. (`[spectrotemporal]`) correctly identified as the paper this
  project's research question most directly answers to (§3) — previously
  mischaracterized as an unrelated classification paper.
- Research question (§4) corrected to the charter's C2 "knee point" framing.
- Baseline 2's citation resolved to `[nmfcrnn]` (§5.2) — the `[nmfcrnn]`
  dataset discrepancy itself (ICBHI+Fraiwan vs. HLS-CMDS isolated
  recordings) remains open, see below.

**Still open:**
- Resolve the `[nmfcrnn]` dataset discrepancy (§2 table) — bears on what
  "isolated H/L recordings" means for Baseline 2's own reproduction.
- Confirm the SSA paper's (`[ssa]`) dataset — citation now known, dataset
  still not stated in the summary reviewed.
- **Statistics**: this memo's and `report/report.pdf`'s across-fold
  mean±std (e.g. Table 5's ±3.71 dB) understates pooled-row spread by an
  order of magnitude (pooled heart SAR is 6.30±39.38 dB, not 6.46±3.71) —
  report medians and/or bootstrap CIs alongside means once re-issued, and
  always state which level (per-row vs. per-fold) the dispersion is
  computed at.
- Independently re-read Yaqub et al. and `[nmfcrnn]` from the primary
  sources rather than relying on the review's summary of the Reading List.
- Decide classifier architecture and training protocol for 5.3, now scoped
  to a continuous separation-quality sweep (§4) rather than two fixed
  conditions.
- Decide whether Condition B reuses Condition A's trained classifier or
  retrains on separated audio.
- Repo hygiene: `HLS-CMDS` was an embedded git repo with no `.gitmodules`
  entry (already removed from tracking this session, pending the Mendeley
  copy landing outside the repo per the first item above — consistent with
  the review's recommendation, not a submodule); `src/baselines.py` is
  still untracked and uncommitted.
