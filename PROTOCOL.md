# Protocol Memo: Separated vs. Isolated Accuracy on HLS-CMDS

Status: draft — open items flagged inline. Date: 2026-08-17.

**2026-08-18 update:** Reviewed in TEEP2026_Sprint0_Review (Satya Adhiyaksa,
18 Aug 2026). Split, metrics wrapper, and test suite confirmed strong as-is.
One major finding: 109/145 Mix.csv rows have a mixed recording that is
acoustically unrelated to its named heart/lung sources — confirmed
independently both by this project's own clipping-based audio-quality audit
(§5.2) and by a direct additivity test (see `verify_additive_triplets()` in
`load_dataset.py`, added in response to the review).

**2026-08-19 update:** The review's blocking item — that `HLS_CMDS/` might be
a downsampled GitHub redistribution rather than the Mendeley release — was
tested and closed. A fresh Mendeley download (v3) is byte-identical to this
working copy on all six top-level artifacts; checksums are recorded in
`README.md`. The 4000 Hz sample rate and the 36/145 pairing are therefore
properties of the published dataset, not of a bad copy, and **the
separation-quality numbers in this memo and in `report/report.pdf` are
measurements on HLS-CMDS as released.** See §8 for what remains open.

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
| AI-Driven Cardiorespiratory Signal Processing (LingoNMF / PL-NMF) [aidriven] | **Resolved from primary source (2026-09-22), §§4.1.3/4.2.3/5.5/6.2.2**: HLS-CMDS, but **never `Mix.csv`'s 145 published mixtures**. §4.1.3 (p. 41) evaluates on "210 clinical manikin recordings" — not 145, and HLS-CMDS is not named there, so the set is *not stated*; §4.2.3 (p. 52) re-pairs HLS-CMDS segments at random into 25,000 mixtures of its own; §5.5 (p. 65) and §6.2.2 (p. 72) use the 50+50 isolated recordings only | PL-NMF (parallel multi-layer NMF, heart/lung-tuned) + LLM-derived fundamental frequency | **SDR/SIR/SAR after all** (defined Eqs. 4.9–4.11, p. 42; Fig. 4.4 plots them with 95% CIs and paired t-tests) — the earlier "not reported / unspecified quality proxy" reading came from the review summary and was wrong | clustering + anomaly detection, **not** accuracy | LingoNMF SIR 22.4 / SAR 25.2 / SDR 22.3 dB (Table A.1, p. 82), dataset for that table *not stated*; "statistically significant gains," p<0.01 | not stated |
| Spectrotemporal Deep Learning for Heart Sound Classification [spectrotemporal] | **Resolved from primary source (2026-08-24), Secs. 3/5**: PhysioNet 2016 for initial training, **HLS-CMDS itself** for external validation (not a separate/unrelated dataset) | **bandpass filter** (Experiment 4's stress test: heart sounds computationally separated from HLS-CMDS's own mixed heart-lung recordings) | none (no SDR/SIR/SAR — only downstream classification accuracy is reported) | 4-class PCG classification (Normal/Murmur/Extra Sound/Rhythm Disorder), **including on separated audio** | **Confirmed from the primary source directly, not the review's summary**: accuracy 89.0% (Exp. 3, clean HLS-CMDS) → 41.0% (Exp. 4, bandpass-separated) — Table 9: n=5365, weighted F1 0.89→0.39. This is the single result this whole project exists to explain, and (per their own Sec. 6.2) the collapse is attributed to bandpass-filter artifacts on spectrally-overlapping signals — the same method class as this project's own Baseline 1. | not stated |
| Respiratory Disease Classification (NMF-enhanced log-mel + CRNN) [nmfcrnn] | **Resolved from primary source (Sec. 3.2.3/4.1) — not a conflict, two datasets for two roles**: HLS-CMDS isolated heart/lung recordings are used *only* as an auxiliary corpus for offline NMF dictionary learning ("HLS-CMDS recordings were used exclusively for dictionary learning"); the actual *classification* dataset/results are ICBHI 2017 + Fraiwan CWLS, harmonized into a 7-class cohort (Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI). HLS-CMDS is what feeds Baseline 2's hyperparameters. | NMF-based respiratory enhancement (Kr=20 respiratory / Ki=10 interference dictionary ranks, 100 MU dict iters, 60 MU activation iters, 4th-order Butterworth 50–1800 Hz denoise before STFT/NMF — all confirmed directly, matches `src/baselines.py` Baseline 2 exactly as of this cross-check) | **none** (no SNR/SDR/SI-SDR/correlation/spectral-distortion/error-energy) — confirmed by direct read, not just the review's summary | multi-class respiratory disease classification | 96.14±0.50% acc, 94.05±1.21 Macro-F1; denoised-ablation Macro-F1 84.98±22.0 over 10 seeds | **recording-level**, not subject-level (authors' own flagged limitation) |
| Edge-Enabled Portable Lung-Sound Classifier [edgelung] | **HLS-CMDS** | EVMD (Enhanced Variational Mode Decomposition, Sec. II.B) + 150 Hz lowpass to isolate/subtract the cardiac mode | **none in the original paper** — no SNR, SDR, correlation, or spectral-error figure, despite running directly on HLS-CMDS's paired mixtures | lung sound classification (on the *HF_Lung V1* dataset, not HLS-CMDS — HLS-CMDS is only the EVMD-filtering demo/auxiliary set, Sec. II.B) | not detailed / separation step "reports no result" in the original paper — **now reproduced as this project's Baseline 5 (S4-01)**, the first SDR/SIR/SAR numbers computed for this method on HLS-CMDS (see `results/first_sdr_sir_sar_table.html`) | not stated. **Correction (2026-08-24)**: edge hardware is a PYNQ-ZU **FPGA board** (~15 W total system power, Sec. III), not an MCU — the ESP32 microcontroller only handles acquisition (stethoscope digitization + HTTP POST of raw PCM), not inference; earlier drafts here mischaracterized this as "ESP32 + PYNQ-ZU" implying a hybrid MCU inference target. |
| Performance Measurement in Blind Audio Source Separation | — (methods paper) | — | defines SDR/SIR/SAR (BSS Eval) | n/a | n/a | n/a |
| Cardiorespiratory Sound Separation Using SSA [ssa] | **Resolved from primary source (§III)**: HLS-CMDS isolated recordings (cites the same Torabi et al. descriptor paper this project uses), but *not* HLS-CMDS's own Mix.csv — the authors synthetically all-pairs-combine 10 cardiac x 5 respiratory recordings into 50 test mixtures, then add 2% RMS Gaussian noise. Not directly comparable to this project's own (real, additive) Baseline 1–3 results: the SSA paper's mixtures have an actual noise floor this project's don't, per PROTOCOL.md's own note below. | two-stage Singular Spectrum Analysis (L=50) | SDR, STOI, ground-truth correlation, vs. 5 baselines. **Confirmed from Table I directly**: the Butterworth bandpass baseline scores SDR 5.7 dB cardiac / **-5.7 dB respiratory** (i.e. negative — filtering alone fails on the respiratory side entirely), STOI 0.5/[unreported], correlation not tabulated for this row in the excerpt read. This confirms (not just "likely") the 5.7 dB figure previously guessed at in `src/baselines.py`'s review section. | **none** — no classification | respiratory SDR 5.34 dB (MSSA) vs. 4.64 dB (first-stage SSA); cardiac 26.44 dB both stages; respiratory correlation 80.5% (MSSA) vs. 77.2% (SSA) vs. 10.2% (their own NMF baseline, cardiac 50.0%) — all confirmed from Table I directly, not the review's summary | n/a |

**Caution on `[aidriven]`'s headline SDR.** The thesis's best-known figure,
26.8 dB, is **not** an HLS-CMDS result: it is VAE-WMT on "Dataset One"
(Kaggle Respiratory + CirCor + Chest Wall), per §4.2.3 (p. 52) and
Table A.2 (p. 83). On HLS-CMDS ("Dataset Two") the same model scores
15.1 dB. `report/paper.tex` §1 still attributes 26.8 dB to "this dataset"
and needs correcting.

**References** (from the Notion Reading List, via TEEP2026_Sprint0_Review — cross-checked against the primary sources in `papers/`; see each row for what that resolved):
- `[aidriven]` Torabi, PhD thesis, McMaster, 2025. Full text at
  `papers/2602.09210v1.pdf` (arXiv:2602.09210v1, xxi+123 pp.); the page
  numbers cited in this memo are that PDF's.
- `[spectrotemporal]` Yaqub et al., *CMES*, 2025. doi:10.32604/cmes.2025.071571.
- `[nmfcrnn]` Han, Quan, Matuszewski & Corbett, *Sensors*, 2026. doi:10.3390/s26134268.
- `[edgelung]` Puneet, Shankar, Koluguri & Srivastava, "Edge-Enabled Portable Classifier for Lung Sounds Using Convolutional Neural Networks," *IEEE BioCAS*, 2025. doi:10.1109/BioCAS67066.2025.00016.
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
- **SDR is not perceptual and is gameable.** The paper is explicit that SDR
  depends on which class of "allowed distortion" is used to project the
  estimate before scoring (§III) — a more permissive class (e.g. the
  time-invariant multi-tap filter `mir_eval.separation.bss_eval_sources`
  actually allows, confirmed from `src/metrics.py`; not literally
  "time-varying," corrected from an earlier draft of this line) can absorb
  more of the estimate's actual distortion into the "allowed" bucket and
  push SDR up without the estimate sounding, or classifying, any better.
  SDR alone is not sufficient evidence of separation quality — SIR and SAR
  have to be reported alongside it, which is exactly what `metrics.py`
  already does.
- **This project's own inference, not the paper's** (Vincent et al. is a
  2006 methods paper and predates HLS-CMDS by two decades, so it cannot and
  does not mention it by name — corrected misattribution from an earlier
  draft of this line): the paper's four-term decomposition (§I, `e_noise`
  representing "sensor noises" specifically, distinct from interference and
  artifacts) implies that since HLS-CMDS mixtures are heart + lung only with
  **no independent sensor-noise reference**, the BSS Eval "noise" term is
  effectively zero here, and SAR captures pure
  algorithmic artifact, not real-world sensor/environmental noise. That
  bounds what this study can claim: it measures separation-then-classification
  degradation in a clean-mixture setting, not robustness to real acquisition
  noise (relevant given the Edge-Enabled paper's real ESP32 hardware).

## 3. The gap

**The paper this project most directly answers to is Yaqub et al.
(`[spectrotemporal]`), not the HLS-CMDS-specific papers below it.**
**Resolved from the primary source directly (2026-08-24)** — Yaqub,
Orakzai, Qureshi, Mushtaq, Siddique & Radwan, *Computer Modeling in
Engineering & Sciences*, 145(2), 2025, doi:10.32604/cmes.2025.071571: a
ResNet-18 PCG classifier is trained on PhysioNet 2016, then **externally
validated on HLS-CMDS itself** (not a different dataset, as this memo
previously guessed) across four experiments — binary fine-tuning (88.0%),
4-class fine-tuning (86.0%), full retraining on HLS-CMDS's own 4-class
labels (89.0%, their best/most balanced result), and a stress test
(Experiment 4) that re-evaluates the Experiment-3 model on heart sounds
**computationally separated from HLS-CMDS's own mixed recordings using a
standard bandpass filter** — where accuracy collapses to 41.0% (Table 9:
n=5365, weighted F1 0.89→0.39, every class degrades). Their own Discussion
(§6.2) attributes the collapse specifically to bandpass artifacts on
spectrally-overlapping signals — the same class of method as this
project's own Baseline 1, whose separation-quality results (§5.2) already
show the identical SIR-gain/SAR-cost tradeoff on the separation-quality
side of exactly this mechanism. This memo previously filed Yaqub under "no
separation, binary classification" and missed this entirely; that error is
now corrected here and in `report/report.tex` (Sec. "Related Work"/
"Yaqub et al."), not just flagged as open.

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
**Designed and validated 2026-08-27, see §5.3.1**: the degraded/blended
mechanism (alpha-interpolation toward ground truth) and the decision that
the curve's x-axis is measured SDR, not synthetic SNR — the specific
choice that lets Yaqub's point actually land on this curve.

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

**Substrate change (2026-08-24, S1-09/S1-10/S1-13):** the per-baseline
results below (and `results/baselines_report.html`) are native-Mix.csv
results, kept as-is for history. The **primary evaluation substrate for the
headline separation-quality table is now the synthetic mixing set**
(`src/synthetic_mix.py`), not native Mix.csv pairs — only 36/145 native rows
are additive (§5.1/§8), too few to source a real SDR-vs-difficulty sweep for
C2's knee-point analysis. The synthetic set matches the native construction
model exactly (`mixed = a·(heart+lung) + noise`, gain calibrated from the
36 native rows' own fitted gains — validated in S1-10 to reproduce those 36
rows under `verify_additive_triplets()`), sweeps additive noise across
`SNR_SWEEP_DB` for a controllable difficulty axis (diverging from Han &
Quan's fixed 2% RMS noise level, matching its noise type), and uses a
source-file-level fold split (S1-13, distinct from S2-03's triplet-level
split — full combinatorial pairing collapses S2-03's leak-group logic into
one component). A fifth baseline, **EVMD** (S4-01, see below), was added
alongside this substrate change. The resulting three-column table (synthetic
set / native additive rows / Han & Quan's own Table I as a distinct,
non-recomputed column) is `results/first_sdr_sir_sar_table.html` — see
`src/first_sdr_table.py` and `code_description.md`'s `synthetic_mix.py` /
`Baseline 5` sections for full design and construction detail.

**Baseline 1 — simple bandpass filtering (implemented, `src/baseline/baseline1.py`).**
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

**Baseline 2 — supervised NMF, adapting `[nmfcrnn]` (Han, Quan,
Matuszewski & Corbett, *Sensors*, 2026, doi:10.3390/s26134268; now
**cross-checked directly against `papers/Respiratory_Disease_Classification_..._pdf`,
Sec. 3.2.1–3.2.4** — the earlier hedge here is resolved) (implemented,
`src/baseline/baseline2.py`).** Two fixed dictionaries, one per source, learned from
isolated H/L recordings via KL-divergence multiplicative-update NMF, then
frozen and used to solve for per-mixture activations (the classic
supervised-NMF separation recipe, Smaragdis 2007). Hyperparameters, all
confirmed against the paper's text (previously given only during a
chat-based handoff, unverified): Kr = respiratory (lung) dictionary rank =
20, Ki = interference (heart) dictionary rank = 10; 100 MU iterations to
fit each dictionary, 60 MU iterations to solve per-mixture activations with
the dictionaries frozen, H initialized non-negative and lower-bounded at
1e-3 (Sec. 3.2.4); STFT via a 512-sample window / 256-sample hop / 512-point
FFT (Sec. 3.2.3 — this project's prior 128-sample hop was unverified and has
been corrected to 256). **One real gap found and fixed**: the paper denoises
every snippet — both dictionary-training recordings and the mixture being
separated — with a 4th-order Butterworth bandpass (50–1800 Hz) *before*
STFT/NMF (Sec. 3.2.1), to strip baseline drift and acquisition noise; this
project's initial reproduction skipped that stage entirely. Now added
(`DENOISE_BAND` in `src/baseline/baseline2.py`), applied identically to Baseline 3
(§ below, which imports it directly rather than duplicating it) so the two
stay a controlled ablation of each other. Separation
extends the paper's own single-sided respiratory-only reconstruction to a
symmetric two-source mask (the paper never reconstructs or evaluates the
heart/interference side at all — see `src/baseline/baseline2.py`'s docstring for the
full adaptation note). The `[nmfcrnn]` dataset discrepancy flagged in §2's
table is now resolved, also from the primary source: HLS-CMDS isolated
recordings are the auxiliary dictionary-learning corpus (exactly this
project's own HS.csv/LS.csv), while ICBHI 2017 + Fraiwan CWLS is the
paper's separate classification dataset — not a conflict, two datasets for
two roles in the same paper.

**Leakage trap.** A third of HS.csv/LS.csv's recordings are byte-identical
to a heart/lung component of some Mix.csv row — that row's own ground
truth (§5.1/`split.py`). A naive reproduction that fits the dictionaries
on "all of HS.csv/LS.csv" therefore lets a mixture's own ground-truth
source into its separation dictionary. This implementation only ever sees
`hs_allowed`/`ls_allowed` — `eval_harness.cross_validate`'s fold-safe
pool, same triplet-level split as Baseline 1 — so the number below is
lower than a naive reproduction would report. That lower number is the
correct one; a higher one would mean leakage, not a better model.

5-fold result, post-denoising-fix (`python src/baselines.py`):

| subset | source | SDR (dB) | SIR (dB) | SAR (dB) |
|---|---|---|---|---|
| Full 145 rows | heart | -12.95 ± 1.34 | 4.82 ± 1.65 | -10.27 ± 0.89 |
| Full 145 rows | lung  | -15.09 ± 1.10 | 0.58 ± 0.33 | -10.11 ± 1.46 |
| Additive-only 36 rows | heart | 2.69 ± 2.96 | 7.42 ± 3.99 | 7.61 ± 0.87 |
| Additive-only 36 rows | lung  | 0.62 ± 3.14 | 2.25 ± 3.55 | 9.93 ± 0.91 |

On the full 145-row set, essentially unchanged from the pre-fix numbers
(heart was -12.82±1.31, lung -15.09±0.98) — the denoising step and corrected
hop size turn out **not** to be why this baseline underperforms; see below.
Compare against Baseline 1 (bandpass) on the same full set: heart SDR
-12.65±1.40 / SIR 4.07±1.16 / SAR -8.10±1.44; lung SDR -13.83±1.12 / SIR
1.73±0.29 / SAR -9.27±1.42. Leakage-safe supervised NMF comes in
**essentially tied with the trivial bandpass filter on heart** (SDR delta
well within one fold-std of either) and **~1.3 dB worse on lung** — not the
clear win over a zero-training baseline a paper's reported numbers would
suggest, though note `[nmfcrnn]` itself never reports a separation-quality
number at all (only downstream classification accuracy on its own,
different dataset), so there is no paper SDR/SNR figure to be "behind." This
tracks with a sanity check run against a single mixture row using an
*oracle* mask built from that row's own true heart/lung spectrograms
(cheating on purpose, just to bound the method): even the oracle mask only
modestly beat the no-separation reference on that row, so the ceiling for
mask-based separation on this dataset's real (not synthetically summed)
mixture recordings looks lower than the bandpass-vs-no-separation
comparison alone would suggest.

On the additive-only 36-row subset the picture changes substantially: both
sources swing to positive SDR (heart +2.69, lung +0.62 dB) — most of the
full-set's negative SDR is attributable to the 109 rows whose "mixed" file
isn't actually related to its named heart/lung sources (§0/`README.md`'s
Dataset section), not to a separation-method failure. This is the first
result this project has on the valid-only subset for Baseline 2 (previously
only Baseline 1/3 had been re-run there) — see Baseline 3 immediately below
for the apples-to-apples ablation this unlocks.

**Baseline 3 — standard NMF, no learned dictionary (ablation against
Baseline 2; ref S3-03, `src/baseline/baseline3.py`, `make_standard_nmf_baseline`).**
Same total rank (Ki+Kr=30), STFT params, and denoising pre-filter as
Baseline 2, but W and H are both factorized directly out of each held-out
mixture's own spectrogram — no dictionary-learning phase, so
`hs_allowed`/`ls_allowed` go unused. Components are unlabeled by
construction; assigned to heart/lung post-hoc by spectral centroid (heart
energy concentrated below ~200 Hz, per Baseline 1's own PSD survey).

5-fold result, post-denoising-fix:

| subset | source | SDR (dB) | SIR (dB) | SAR (dB) |
|---|---|---|---|---|
| Full 145 rows | heart | -13.07 ± 1.38 | 4.65 ± 1.08 | -10.08 ± 1.35 |
| Full 145 rows | lung  | -14.41 ± 1.23 | 1.42 ± 0.23 | -10.00 ± 1.47 |
| Additive-only 36 rows | heart | 3.44 ± 2.02 | 7.10 ± 3.23 | 9.38 ± 0.55 |
| Additive-only 36 rows | lung  | 2.22 ± 3.91 | 4.89 ± 4.78 | 9.64 ± 0.96 |

**This is now a genuine apples-to-apples ablation**, both baselines sharing
denoising/STFT/mask code and differing only in whether the dictionary is
pretrained (Baseline 2) or factorized fresh per mixture (Baseline 3). On the
additive-only subset, Baseline 3 (no learned dictionary) is **as good as or
slightly better than** Baseline 2 (learned dictionary) on both sources
(heart +3.44 vs. +2.69 dB; lung +2.22 vs. +0.62 dB) — i.e. on this dataset,
`[nmfcrnn]`'s pretrained-dictionary strategy is not earning its keep over
blind per-mixture NMF once leakage and the non-additive rows are both
controlled for. Worth treating as provisional (n=36 additive rows, wide
per-row spread — see the pooled std note in §8) rather than a settled
result, but it's a real, reproducible finding, not a sanity-check artifact.

**Baseline 4 — multi-stage SSA (MSSA), reproducing `[ssa]` (Han & Quan,
*2025 ICSPS*, doi:10.1109/ICSPS66615.2025.11347745; cross-checked directly
against `papers/Cardiorespiratory_Sound_Separation_Using_Singular_Spectrum_
Analysis.pdf`, Sec. II) (implemented, `src/baseline/baseline4.py`,
`fit_ssa_baseline`/`mssa_separate`).** Zero-training, two-stage decomposition
applied identically to every mixture — no dictionary or fold-fitting step,
so `hs_allowed`/`ls_allowed` go unused and there is no leakage trap to speak
of (nothing is fit on any recording). All four hyperparameters are stated
explicitly in the paper and used as-is: window length L=50 (Sec. II.A), a
250 Hz cardiac/respiratory frequency split (Sec. II.B, based on the S1/S2
heart-sound range), a 2% eigenvalue-contribution threshold for stage-2
"high-energy" respiratory RCs (Sec. II.B, `100/L` for L=50), and a 50%
cross-correlation threshold for including additional stage-2 RCs (Sec.
II.B). **Confirmed**: the 250 Hz split is a physiological frequency in
absolute Hz, not normalized to the paper's own sample rate, so it sits
correctly relative to this project's 4000 Hz sample rate (2000 Hz Nyquist,
8x above the split) with no rescaling needed; the paper's own dataset is the
same Torabi et al. HLS-CMDS descriptor source this project cites (their ref
[19]), so there's no cross-dataset sample-rate mismatch to resolve either —
this was the specific item flagged for confirmation when Baseline 4 was
scoped.

Stage 1 (cardiac) SSA-decomposes the raw mixture into 50 reconstructed
components (RCs) via trajectory-matrix embedding + SVD + diagonal averaging;
each RC's Welch-PSD peak frequency sorts it into the final heart_est
(≤250 Hz) or a residual pool (>250 Hz) that stage 2 further decomposes.
Stage 2 (respiratory) selects RCs whose relative eigenvalue contribution
clears the 2% threshold, then adds any remaining RC whose Pearson
correlation with that selected set's sum exceeds 50% — an interpretation
choice, since the paper doesn't fully spell out what "the remaining modes"
are correlated against; flagged in `src/baseline/baseline4.py`'s docstring as this
project's own reading, same as Baseline 2's two-sided-mask extension is
flagged.

**Synthetic-set comparison against Table I** (the paper's own evaluation
isn't run on HLS_CMDS's Mix.csv — see the `[ssa]` row in §2 — so this is the
only like-for-like comparison available; `build_synthetic_mixes()`
reproduces the paper's recipe from this project's own HS.csv/LS.csv: 10
heart x 5 lung recordings, all 50 combinatorial pairs, +2% RMS Gaussian
noise):

| | Paper's MSSA (Table I) | This reproduction |
|---|---|---|
| Cardiac SDR | 26.4 dB | 1.24 dB |
| Cardiac correlation | 99.2% | 64.8% |
| Respiratory SDR | 5.3 dB | 6.81 dB |
| Respiratory correlation | 80.5% | 44.4% |

Respiratory SDR is actually in the paper's range (slightly better);
everything else — cardiac SDR/correlation and respiratory correlation — is
substantially below the paper's own numbers. This is a larger, more
asymmetric gap than Baseline 2's turned out to be (which resolved to "harder
real-dataset setup," not a bug, once measured on the right subset) — **not
yet resolved**, flagged in §8 as an open item rather than assumed to be
either an implementation bug or an inherent reproduction gap.

5-fold result on this project's own real mixtures (via `eval_harness`, for
consistency with Baselines 1–3's reporting; not paper-comparable per the
`[ssa]` row in §2's synthetic-vs-real-mixture note):

| subset | source | SDR (dB) | SIR (dB) | SAR (dB) |
|---|---|---|---|---|
| Full 145 rows | heart | -12.80 ± 1.52 | 4.03 ± 1.18 | -7.80 ± 1.58 |
| Full 145 rows | lung  | -13.56 ± 1.26 | 2.61 ± 0.71 | -9.53 ± 1.38 |
| Additive-only 36 rows | heart | 5.17 ± 4.05 | 5.88 ± 3.97 | 18.97 ± 3.56 |
| Additive-only 36 rows | lung  | 5.32 ± 3.20 | 8.53 ± 4.55 | 12.04 ± 2.89 |

In the same ballpark as Baselines 1–3 on the full 145-row set. On the
additive-only 36-row subset, Baseline 4 is the **best of all four baselines
so far** on both sources (heart +5.17 dB vs. Baseline 3's +3.44 dB;
lung +5.32 dB vs. Baseline 3's +2.22 dB) — notably also with much higher SAR
(19.0/12.0 dB vs. Baselines 2/3's single-digit SAR), consistent with SSA's
own claim of preserving signal integrity better than filtering/NMF-based
masking. This real-mixture result is the opposite direction from the
synthetic-set comparison above (where Baseline 4's cardiac side badly
underperforms the paper's own number) — worth noting as a further reason to
treat the synthetic-set gap as a specific, unresolved discrepancy rather
than evidence Baseline 4 is broken generally.

**Still open:** PL-NMF/LingoNMF if reproducible from the AI-Driven paper's
description, as a fourth separation method; resolving Baseline 4's
cardiac-side synthetic-set gap (see §8).

**Baseline 5 — EVMD, reproducing `[edgelung]`'s separation stage (S4-01,
pulled forward from Sprint 4 into Sprint 3; implemented, `src/baseline/baseline5.py`).**
Zero-training, like Baselines 1/4. Sweeps VMD (Dragomiretskiy & Zosso 2014,
implemented directly — no VMD package available) over `K=2..10` at
`alpha=2000`, selecting the first `K` whose energy-loss coefficient and
every mode's per-mode check (permutation entropy / frequency band /
kurtosis-index cascade) clears the paper's own thresholds
(`mu1=0.01, mu2=0.4, mu3=0.3, mu4=0.05`); isolates the mode with the lowest
peak frequency as cardiac, lowpasses it at 150 Hz (the paper's cutoff) for
`heart_est`, subtracts it from the mixture for `lung_est`. **The paper's own
description of how these four criteria combine is thin** — this project's
best-faith reading is flagged throughout `code_description.md`'s Baseline 5
section as interpretation, not confirmed literal reproduction (per the
project lead's own framing when this baseline was scoped: an exact
reproduction may not be possible, and documenting where it breaks is a valid
result). **Measured, not assumed**: the full K-sweep costs ~15 s/mixture
(dominated by VMD's ADMM iterations at this dataset's 60,000-sample/15 s
length) — infeasible at the full synthetic-set n within this session's
timebox, so Baseline 5's synthetic-column result runs on a stratified
subsample (n stated explicitly in `results/first_sdr_sir_sar_table.html`,
smaller than the other four baselines' n on the same column — a disclosed
compute-driven reduction, not a silent one). Its native-additive column
(n=36) is cheap enough to run in full. `[edgelung]` itself reports no
SDR/SIR/SAR for this method at all (§2) — Baseline 5 is, as far as this
project has found, the first separation-quality measurement for it on
HLS-CMDS.

**Baseline 6 — Conv-TasNet-lite, the first neural separation baseline
(implemented, `src/convtasnet.py`).** Unlike Baselines 1-5 (zero-training or
a fixed-dictionary fit), this trains a fresh encoder/TCN-separator/decoder
network per fold from that fold's own leakage-safe dictionary pool, on
mixtures synthesized on the fly using `synthetic_mix.py`'s own mixing
recipe. **Sample-rate decision** (made explicitly before writing any
training code, since public Conv-TasNet/Sepformer checkpoints are trained
at 8-16 kHz and this dataset is natively 4000 Hz): train from scratch at
the native 4 kHz rather than resample up, which would invent no real
information. This is not a resolution limitation — heart (20-200 Hz) and
lung (100-1000 Hz) sit comfortably under 4 kHz's 2000 Hz Nyquist (§5.2's
Baseline 1 PSD survey). Architecture follows Conv-TasNet's own design
(Luo & Mesgarani 2019) and NeoSSNet (Poh et al. 2024 — the closest prior
work, also a masked Conv-TasNet-style model at 4 kHz), sized down to ~325K
parameters ("lite") for this dataset's much smaller training pool; fixed
source order (heart/lung are distinguishable classes, not interchangeable
speakers like Conv-TasNet's original speech-separation task), so no
permutation-invariant training. **Measured, not assumed**: full 5-fold CV
on both evaluation substrates took 568.2s total on the available GPU, with
every fold's training running the full 25 epochs and validation SI-SDR
improving monotonically. Result (`results/baseline6_report.html`):

| source | synthetic SDR, n=1500 | native-additive SDR, n=36 |
|---|---|---|
| heart | 3.16 ± 0.40 dB | 5.15 ± 2.74 dB |
| lung  | 0.37 ± 0.35 dB | 2.14 ± 2.14 dB |

This beats Baselines 1 and 2 on the synthetic set on both sources (Baseline
1: heart 2.10±0.43 / lung -1.24±0.44 dB; Baseline 2: heart -0.70±0.40 /
lung -3.73±0.40 dB — `results/baseline1_2_synthetic_report.html`), the
first baseline in this project to post a positive synthetic-set lung SDR
at all. On the native-additive column (n=36, wide CI) it's roughly tied
with Baseline 1 rather than a clear win — the synthetic column (n=1500) is
the more reliable comparison, same caveat every other baseline's
synthetic-vs-native table carries. This is a first working configuration,
not a tuned/converged model — no hyperparameter sweep was run.

### 5.3 Classification

**Condition A built and measured, 2026-08-27** (`src/heart_classifier.py`,
pulled forward from S5 into S2 — Sprint 1 review, three weeks early: the
separation track finished roughly three sprints ahead of the Gantt, and the
charter's own ordering rule spends a lead on C2, not more C1 depth. This is
the first *accuracy* number in the project — everything before it is
SDR/SIR/SAR in dB — and the Condition A anchor the rest of C2's sweep gets
compared against; reported on the same leak-group folds every separation
baseline already uses, with a CI.).

**Class-grouping decision (blocking, made before any classifier code).**
Full 10-class Heart Sound Type classification is out per the charter: under
the leak-group 5-fold split, S4 (n=2 recordings total) cannot appear in
every fold's training set at all — if both of its two recordings land in
the same held-out fold's leak group, three of five folds train with zero
S4 examples — and AV Block/Tachycardia (n=3 each) are barely better off.
Two options were considered: binary Normal/Abnormal, and a grouped
multi-class scheme. Grouped classes were chosen, using the exact four
groups Yaqub et al. (`[spectrotemporal]`) use for this same dataset
(confirmed directly from the primary source, Secs. 5.1–5.2: Tables 6–9's
class list — "Normal", "Murmur", "Extra Sound", "Rhythm Disorder" — the
same class list their own 89%→41% Experiment 3/4 collapse, the motivating
result this project exists to explain (§3), is measured on). Mapping
HLS-CMDS's 10 Heart Sound Types onto that scheme is unambiguous from
clinical terminology and the type names themselves: Normal → Normal (n=9);
the four murmur types (Mid/Late Systolic, Late Diastolic, Early Systolic)
→ Murmur (n=24); S3/S4 (extra heart sounds beyond S1/S2) → Extra Sound
(n=7); Atrial Fibrillation/Tachycardia/AV Block (rhythm and conduction
disorders) → Rhythm Disorder (n=10). This solves the small-class problem
(the smallest group, Extra Sound at n=7, is comfortably large enough to
appear in every fold's training set, unlike 10-class's n=2/3 classes) and,
unlike collapsing to binary Normal/Abnormal, keeps Condition A's number
directly comparable to the specific accuracy figures (86–89%) this
project's Yaqub-collapse motivation is anchored to — binary would solve the
imbalance problem too, but at the cost of measuring a different question
than the one the charter's motivating result is actually about.

**Fold assignment.** `split.py`'s new `assign_hs_folds()` extends the
existing leak-group split to HS.csv rows themselves (not just whether
they're excluded from a fold's dictionary pool, which `dictionary_pool()`
already handled): a recording byte-identical to a Mix.csv leak group's
heart component inherits that leak group's fold; recordings never reused
in any mixture carry no leak-group constraint and are assigned by balanced
round-robin, stratified by `class_group` so no fold is starved of a
minority group. Verified on the real data (2026-08-27 run): every one of
the 5 folds' training sets contains all 4 class groups.

**Architecture: one, deliberately** (per the charter — this is a measuring
instrument, not a contribution; S7-06 repeats the whole curve with a
second architecture as a robustness check once the curve itself exists).
13 MFCCs pooled to mean+std over time (26-dim feature vector) + an
RBF-kernel, class-balanced SVM. Chosen over a CNN or other deep spectrogram
model because n=50 recordings (as few as ~35–40/fold in training) is far
too little data to train one without the result being dominated by
overfitting noise rather than signal — MFCC+SVM was this section's own
stated fallback for exactly this regime (see the "Open items" note below,
now resolved). No hyperparameter search was run — `C=1.0`, `gamma='scale'`
are sklearn's own defaults, not tuned on this data, the same
disclosed-first-configuration framing as Baseline 6.

**Result** (`results/heart_classifier_report.html`, 5-fold CV,
`n_folds=5, seed=0`): accuracy **58.0% ± 7.3%** (95% CI across folds),
Macro-F1 **0.43 ± 0.07**. Per-class-group recall, pooled across all folds'
held-out predictions: Normal 44%, Murmur 88%, Extra Sound 29%, Rhythm
Disorder 20% — a majority-class bias toward Murmur (n=24/50, the largest
group), the expected shape of the class-imbalance problem the grouping
decision reduced (vs. 10-class's n=2/3 classes) but did not eliminate.
With only 5 folds, treat this CI as indicative rather than decisive, per
§6's own caveat on small fold counts — the same caveat every separation
baseline's across-fold std already carries.

**Failure mode analysis (confusion matrices), 2026-08-27**
(`heart_classifier.py`'s `confusion_counts()`/`confusion_recall_pct()`/
`top_confusions()`, `results/heart_classifier_report.html`). The pooled
confusion matrix (every recording predicted exactly once, in its own
held-out fold) shows the low per-class-group recalls above are not a
symmetric spread of confusion — they have a specific direction:

|                      | → Normal | → Murmur | → Extra Sound | → Rhythm Disorder |
|---|---|---|---|---|
| **Normal** (n=9)              | 44% | 11% | 11% | **33%** |
| **Murmur** (n=24)             | 0%  | **88%** | 4% | 8% |
| **Extra Sound** (n=7)         | 0%  | **43%** | 29% | 29% |
| **Rhythm Disorder** (n=10)    | 20% | **40%** | 20% | 20% |

Both minority classes' dominant error pulls toward **Murmur**: Extra Sound
is predicted Murmur more often than it's predicted correctly (43% vs. 29%
recall), and Rhythm Disorder is predicted Murmur nearly as often as any
single other outcome (40% vs. 20% recall) — while Murmur itself is almost
never mistaken for anything else (0% as Normal or Extra Sound, 8% as
Rhythm Disorder). Normal is the exception: its dominant confusion runs
toward Rhythm Disorder (33%), not Murmur. This is consistent with Murmur's
training-set dominance (n=24/50, the majority group) pulling the decision
boundary toward it despite `class_weight='balanced'` reweighting the SVM's
training loss — reweighting the loss doesn't guarantee balanced
*predictions* when the underlying 26-dim MFCC-summary-statistic feature
space gives the four classes limited separability to begin with. This is
a concrete, actionable failure mode for a future architecture iteration
(S7-06) to check specifically: does a richer feature representation (e.g.
full log-mel spectrograms rather than MFCC summary statistics) reduce the
Extra-Sound/Rhythm-Disorder → Murmur pull, or is it inherent to how
acoustically similar these classes are in this dataset regardless of
featurization?

### 5.3.1 Controlled degradation scheme (S6-01)

**Status: designed and validated, 2026-08-27** (`src/degradation.py`).
Ticket history, both pull-forwards deliberate: originally scoped for S6,
pulled forward once into S5 (S6 loses two days to holidays — 25 Sep Fri and
28 Sep Mon — leaving three working days to produce the paper's main figure;
moving the design work earlier is the fix the charter asked for by 17 Sep),
then pulled forward again in the Sprint 1 review, S5 into S2, alongside
Condition A above — the same "a lead is spent on C2, not more C1 depth"
reasoning. Ref code kept as S6-01 throughout.

**What this section is not**: the accuracy-vs-SDR curve itself. That needs
Condition B (classifying separated audio) below, which doesn't exist yet.
This section designs and validates the *mechanism* that curve will sweep
along — a continuous, controlled way to vary "how separated is this audio"
— so that when Condition B lands, the curve-building code has a
ready-made, already-checked tool rather than a fresh design problem.

**The scheme.** For a source with ground truth `g` and a real separation
baseline's estimate `s` (from any of §5.2's six methods),

    degraded(alpha) = (1 - alpha)·g + alpha·s = g + alpha·(s - g),   alpha in [0, 1]

alpha=0 reproduces the clean ground truth exactly; alpha=1 reproduces that
baseline's real separated output exactly. This has the same "signal plus a
scaled error term" shape as `synthetic_mix.py`'s own `SNR_SWEEP_DB`
(`mixed = a·(heart+lung) + noise`, noise scaled to a target SNR) — except
the injected term here, `s - g`, is a *real separation method's own
characteristic error* (its own SIR/SAR balance, per §2's BSS Eval notes),
not i.i.d. Gaussian noise.

**Relationship to the existing SNR sweep — both are kept, deliberately,
because they answer different questions.** `SNR_SWEEP_DB` degrades the
*mixture* before separation runs at all: it stresses "how hard is the
separation problem," and the resulting post-separation SDR is *measured*,
not chosen, confounded with each method's own robustness to noise at each
level. This scheme degrades the *already-separated output*, holding one
method's error signature fixed and dialing only its magnitude: it can hit
any target SDR *for that method's own artifact type* directly, decoupled
from mixture difficulty. Once Condition B exists, both should feed the same
curve — the SNR sweep supplies naturally-occurring points across a range of
real difficulty, and this scheme densifies the curve between them with
controlled, targeted points.

**The blocking design question this ticket exists to answer: what is the
curve's x-axis?** Three candidates were on the table — measured SDR from
the six real separation methods, synthetic input SNR, or both overlaid.
**Decision: SDR (dB), never SNR, with real-method points and this scheme's
dense sweep plotted on the same shared SDR axis.** SNR is a property of the
*mixture* fed into separation, not of the `heart_est`/`lung_est` audio the
classifier actually consumes — and it is undefined for the one external
point this whole curve needs to place: Yaqub et al.'s (`[spectrotemporal]`)
89%→41% collapse (§3) is reported against their bandpass-separated output,
with no SNR ever stated for their mixtures (they ran their pipeline on
HLS-CMDS's native mixed recordings as released). SDR is the one unit
computable for every point the curve needs, including theirs (via this
project's own Baseline 1 bandpass reproduction — confirmed the same method
class in §3) — the only choice that lets their result be placed on the
curve as a point rather than merely cited as an anecdote. "Both overlaid"
from the brief is still the right instinct, but the two things being
overlaid on that one SDR axis are (i) the six real methods' own sparse,
naturally-occurring (SDR, accuracy) points (§5.2, plus their behavior
across the SNR sweep) and (ii) this scheme's dense, controlled sweep per
method — not SDR and SNR as two competing x-axis units.

**Validated on real data, not assumed** (`results/degradation_scheme_
report.html`, one native additive row, Baselines 1 and 4 — two methods
with different artifact signatures, bandpass ringing vs. MSSA's
higher-SAR reconstruction): SDR(alpha) is monotonically non-increasing for
both, and alpha=1 exactly reproduces that method's own independently
measured SDR (sanity check on the plumbing, not just the concept).

**One real finding from that validation, not glossed over**: a *uniform*
alpha grid does not give an even spread of measured SDR. On the row
checked, SDR falls from ~262 dB (alpha=0, near-bit-identical to ground
truth) to ~10.6 dB by alpha=0.1 alone, then flattens out to ~4.8 dB by
alpha=1 — most of the informative range between "clean" and "that method's
real output" is compressed into the first tenth of the alpha axis, an
artifact of dB being logarithmic and most sensitive to small absolute
error near a perfect match. A uniform alpha grid would badly oversample
the flat tail and undersample the steep part. `find_alpha_for_target_sdr()`
(bisection over alpha, valid precisely because SDR(alpha) was checked
monotonic above) fixes this: given a target dB grid, it returns the alpha
that hits each target directly — verified on both methods to land within
~0.05 dB of every requested target across a 30 dB→each method's own floor
grid. This is the tool a future Condition B curve-builder should reach for
to lay out an evenly-spaced dB axis, not a uniform alpha grid.

`test/test_degradation.py` (11 tests): interpolation endpoints/midpoint,
length truncation, source isolation (degrading heart never disturbs the
lung estimate's score), the monotonicity property on real audio, and the
root-finder's accuracy plus its clamping behavior outside the achievable
range. All passing.

**Superseded, 2026-08-27**: Condition B is now built (`condition_b.py`) and
the accuracy-vs-SDR curve itself is now scripted (`sdr_accuracy_curve.py`,
S6-03) — see both below, replacing this "still open" note.

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

**Resolved 2026-08-27:**
- ~~Which classifier architecture~~ — MFCC + RBF-SVM, see above. No confirmed
  HLS-CMDS classification baseline exists among the papers reviewed to
  standardize on — the closest in spirit is the Edge-Enabled paper's CNN,
  but it doesn't report enough detail to reproduce as a baseline directly —
  so this section's own fallback recommendation (something small and
  well-understood, so the *comparison* is the contribution, not the
  classifier's novelty) was taken directly, choosing MFCC+SVM over a
  log-mel+CNN specifically because of n=50's overfitting risk.

**Resolved 2026-08-27 — weight-sharing decision (`condition_b.py`):** the
*same* trained classifier weights per fold, not retrained on separated
training-fold audio. This is exactly Yaqub et al.'s own methodology —
their Experiment 3 model (retrained on clean HLS-CMDS) *is* their
Experiment 4 model (evaluated on bandpass-separated audio), no retraining
between the two. Reproducing their finding "under controlled conditions"
means matching their causal claim (does clean-trained inference degrade on
separated input), not the different question of whether a classifier can
*adapt* to separation artifacts, which stays a noted follow-up, not
pursued here. Same-weights also isolates the separation method as the only
variable between Condition A and Condition B.

A real fold-basis trap surfaced and fixed while building this:
`assign_classifier_folds()` (Condition A) defaults to folding HS.csv
against the *full* 145-row Mix.csv, but `sdr_sweep.py` (S6-02) folds its
36-row native-additive substrate independently — a different row set
produces a different leak-group partition even with the same seed (the
same precedent `first_sdr_table.py`'s own `mix_df=valid_mix_df` override
already established). `assign_classifier_folds()` gained a
`mix_df_with_folds` parameter so Condition B can force both to share one
fold assignment, rather than two independently-computed ones that merely
share a seed.

**First real result** (`condition_b.py`, Baseline 1/bandpass only —
the other five baselines' separated audio was still being generated by
S6-02 at the time this ran; full six-method table pending): on the 36
native additive rows, isolated-ground-truth accuracy is **50.0%**,
bandpass-separated accuracy is **41.7%** — an 8.9-point drop, same
direction as Yaqub's collapse but far smaller in magnitude (their gap is
48 points), and **not statistically significant** at n=5 folds
(paired t-test, p=0.57 — per-fold deltas are noisy, one fold even shows
separated *beating* isolated). Read this as a real but inconclusive single
data point, not a reproduction or a refutation of Yaqub's finding — the
full six-baseline table and the continuous SDR sweep (S6-03 below) are
what actually let this question be answered properly, which is exactly
why this project's C2 framing (§4) asks for a curve, not one comparison.

`test/test_condition_b.py` (8 tests): the fold basis matches an
independent `assign_folds()` computation, the classifier never trains on
a recording in its own held-out fold, and — load-bearing — the isolated
and separated predictions for the same row come from literally the same
fitted classifier object. All passing.

### 5.3.2 Accuracy-vs-SDR curve (S6-03)

**Status: measurement scripted and unit-tested, 2026-08-27**
(`sdr_accuracy_curve.py`). Widened across the Sprint 6 holiday gap: Sprint
6 has only three working days (24 Sep Thu, 29 Sep Tue, 30 Sep Wed — 25/28
Sep are public holidays) since design (S6-01) and sweep generation (S6-02)
already happened in S5, leaving only this measurement step and S6-04.
Per the brief, the measurement itself is scripted to run **unattended**:
`measure_accuracy_at_each_sdr_point()` checkpoints to `results/
sdr_accuracy_curve.csv` every 25 rows and a re-run loads that file first,
skipping any `(baseline, mixed_id, target_sdr)` already measured — safe to
interrupt across the holiday gap and resume without redoing completed
work. It also carries a defensive check: each measured row's fold (from
S6-02's own provenance) must agree with an independently recomputed
classifier fold basis, or the script raises rather than silently risking a
wrong-fold classifier.

This is the actual x-axis §5.3.1 designed populated with real
measurements: for every `source == "heart"` row S6-02 generated (all
target-SDR points, not just each baseline's real alpha=1 output the way
Condition B above measures), the degraded audio is reconstructed
(`sdr_sweep.synthesize_sweep_row()`, cheap — no re-separation) and
classified with the same Condition B weight-sharing setup.
`summarize_accuracy_curve()` aggregates to accuracy per `(baseline,
target_sdr)`, x = mean *achieved* SDR (not the target, since achieved
varies row to row and clamped points sit at that baseline's own real-output
floor regardless of target — `sdr_sweep.py`'s `clamped_to_baseline_floor`).

`test/test_sdr_accuracy_curve.py` (10 tests): only heart-source rows are
measured, results round-trip through disk, a simulated interrupt-and-resume
produces identical predictions to a fresh full run while only
re-computing the missing points (checked via a call-counter on
`synthesize_sweep_row`, not just "the final numbers happen to match"), the
fold-consistency check actually raises on a tampered row, and the summary's
per-group accuracy matches direct computation. All passing.

**Still open**: the actual full-scale run (all six baselines × all seven
target-SDR points) — pending S6-02's generation finishing all six
baselines (Baseline 1 done, the rest still running as of this writing) —
and S6-04 (not scoped here; presumably the knee-point write-up once the
curve exists).

### 5.3.3 Second-architecture robustness check (S7-06)

**Status: designed, implemented, and unit-tested, 2026-08-27**
(`heart_classifier_cnn.py`, plus a `backend` parameter threaded through
`condition_b.py`/`sdr_accuracy_curve.py`/`sdr_knee_point.py`). No longer
optional now that C2's knee point — not classification accuracy on its
own — is the paper's headline result, per the Sprint 0 re-scope. The
charter's original "one architecture only, it's a measuring instrument"
rule (§5.3's own framing) held when C2 was a secondary result; once the
knee point IS the paper, the first reviewer question is whether it is a
property of separation quality or of the one architecture
(`heart_classifier.py`'s MFCC + RBF-SVM) that measured it. A second,
architecturally distinct classifier answers that question directly: if
both curves knee in the same place, the finding is about separation; if
they don't, that disagreement is itself a more interesting result than
the one originally planned.

**Architecture 2**: log-mel spectrogram (40 mel bins, same `N_FFT`/
`HOP_LENGTH` STFT window as Architecture 1) + a shallow CNN (2 conv
blocks, global average pooling, one linear head — a few thousand
parameters). This is §5.3's own other named small-model option (its
docstring names both "log-mel spectrogram + a shallow CNN" and "MFCC +
SVM"; Architecture 1 took the second, this takes the first) — chosen
specifically for maximal architectural distance from Architecture 1 (a 2D
time-frequency representation instead of pooled summary statistics, a
gradient-trained model instead of a kernel method), which is what a
robustness check needs: two flavors of the same idea wouldn't tell us
whether a disagreement is about separation quality or just "SVM vs. a
different SVM."

**Backend contract**: `condition_b.py`/`sdr_accuracy_curve.py`/
`sdr_knee_point.py` now accept a `backend` module parameter (default
`heart_classifier`), so the entire weight-sharing/fold-alignment/
knee-point pipeline runs unchanged against either architecture. A backend
must expose `train_fold_classifiers(hs_df, n_folds)` and
`predict_one(clf, y, sr)` — both existing modules satisfy this without
having to share a class hierarchy, the same "same call contract, no
shared base class" pattern the six separation baselines already use.

**First real result — a genuine, disclosed limitation, not glossed
over**: Architecture 2's own Condition A accuracy is **30.0% ± 8.8%**
(95% CI, 5-fold), well below Architecture 1's 58.0%, in this first
configuration (no hyperparameter search — same disclosed-first-
configuration framing as Baseline 6). Its confusion matrix shows a
collapse toward predicting **Normal** for 44 of 50 recordings (Normal
recall 100%, but Murmur/Extra Sound/Rhythm Disorder all route
overwhelmingly to Normal instead of their own class) — a different
majority-attractor than Architecture 1's own pull toward Murmur (the
earlier confusion-matrix finding), which is itself informative: the two
architectures don't just perform differently, they *fail* differently,
suggesting genuinely distinct decision boundaries rather than two
classifiers converging on the same shortcut.

This is exactly the risk `heart_classifier.py`'s own docstring flagged
when Architecture 1 avoided a CNN in the first place (n=50, ~35-40/fold
in training, "far too little data... without the result being dominated
by overfitting noise rather than signal") — Architecture 2 tests that
risk directly rather than dismissing it, and the risk is real. **Practical
consequence for the cross-architecture comparison**: a knee-point
disagreement found against this first configuration of Architecture 2
cannot yet cleanly distinguish "the knee is architecture-dependent" from
"Architecture 2 isn't yet a reliable enough classifier on its own to
support the comparison" — both remain live possibilities. `sdr_knee_
point.py`'s own report states this caveat automatically whenever
Architecture 2's isolated accuracy falls below 40%, rather than letting a
disagreement be over-read.

`compare_knee_points()` reports, per baseline, whether the two
architectures' estimated knee points agree within a stated tolerance
(default 3 dB) — `agrees=True/False` only when both found a clean
`"crossed"` knee; `agrees=None` (inconclusive, not a silent "no") when
either side found no clean crossing to compare, so a real disagreement is
never confused with "one architecture had nothing to compare against."

`test/test_heart_classifier_cnn.py` (8 tests): feature-shape consistency
across recordings, fit/predict round-trip and reproducibility, the
backend contract, and end-to-end cross-validation output shape — all
using a `max_epochs` override for speed, mirroring `convtasnet.py`'s own
testability pattern. `test/test_condition_b.py` and `test/
test_sdr_accuracy_curve.py` each gained a `TestBackendParameter` class
using a fake stub backend to prove the `backend` parameter is actually
threaded through (not silently ignored) without paying for a real CNN
training run. `test/test_sdr_knee_point.py` gained `TestCompareKneePoints`
(5 tests): agreement within tolerance, disagreement beyond it, the
inconclusive case, baseline-intersection behavior, and the
at-least-two-backends guard. All passing.

**Still open**: the actual cross-architecture curve comparison with real
SDR-sweep data — pending S6-02's generation finishing (same blocker as
§5.3.2) — and, separately, whether Architecture 2's first configuration
needs tuning (regularization, more epochs, a different learning rate)
before its own curve is trustworthy enough to draw the robustness
conclusion from. Both are tracked to a later session update.

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
   already built. **First version landed 2026-08-24** as
   `results/first_sdr_sir_sar_table.html` (5 methods including the new
   EVMD baseline; synthetic-set / native-additive / Han & Quan columns, each
   with n and 95% CI stated — see §5.2's substrate-change note).
2. Classification-accuracy table, Conditions A vs. B, per fold + aggregate.
3. Paired delta + significance test.
4. Separation-quality-vs-accuracy correlation analysis.
5. This memo, updated with the resolved open items (classifier choice,
   separation method choice, weight-sharing decision) once decided.

## 8. Open items (blocking implementation, not this memo)

**Closed, per the 2026-08-19 provenance check:**
- ~~Re-download HLS-CMDS from Mendeley~~ **Done.** v3 retrieved 2026-08-19;
  all six top-level MD5s match the working copy. Version, date, URL and both
  MD5/SHA-256 sets are recorded in `README.md`.
- ~~Re-run `verify_additive_triplets()` on the Mendeley copy~~ **Moot** — the
  Mendeley copy is the same bytes, so the 36/145 count and the 4000 Hz rate
  stand as measured. The C1 rescoping decision this gated was taken on
  2026-08-19: C2 becomes the headline contribution, C1 the instrument.
- ~~Re-issue Table 5 / `report/report.pdf`'s results on the confirmed-valid
  subset~~ **Done** — every results table now reports the 36-row native
  subset and the synthetic substrate as separate columns, with pooled
  row-level median alongside the across-fold mean (the across-fold std
  understates pooled-row spread by roughly 10×).

**Still open:**
- Verify the descriptor paper's own stated sample rate against its text and
  record the page reference (task S1-14). This decides whether the audit
  comment says the descriptor is wrong or that the figure was mis-transcribed
  downstream of it.

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

**Done in the papers/ cross-check session (2026-08-20):** primary-source
PDFs landed in `papers/`, enabling direct verification instead of relying on
TEEP2026_Sprint0_Review's summary:
- **`[nmfcrnn]` dataset discrepancy resolved** (§2 table, §5.2) — HLS-CMDS is
  the auxiliary NMF dictionary-learning corpus (matches this project's own
  HS.csv/LS.csv exactly); ICBHI 2017 + Fraiwan CWLS is the paper's separate
  classification dataset. Not a conflict.
- **`[nmfcrnn]` hyperparameters and pipeline order fully cross-checked**
  (§5.2) — Kr=20/Ki=10, 100/60 MU iterations, and H's 1e-3 init floor all
  confirmed exact matches; STFT hop corrected from an unverified 128 to the
  paper's actual 256; a missing pre-NMF waveform-denoising stage (4th-order
  Butterworth 50–1800 Hz) found and added to `src/baselines.py`, applied
  identically to Baseline 3 for a controlled ablation.
- **`[ssa]` dataset resolved** (§2 table) — HLS-CMDS isolated recordings,
  synthetically all-pairs-combined (10 cardiac x 5 respiratory = 50 test
  mixtures) plus 2% RMS Gaussian noise; *not* HLS-CMDS's own Mix.csv, so its
  SDR numbers aren't directly comparable to this project's Baseline 1–3
  results on real mixtures.
- **`[ssa]`'s Table I confirmed directly** (§2 table) — the "likely" 5.7 dB
  Butterworth-baseline guess in an earlier `src/baselines.py` comment is
  confirmed exactly: 5.7 dB cardiac, **-5.7 dB respiratory** (negative).
- **BSS Eval paper's SDR-gameability claim corrected** (§2) — the actual
  mechanism is allowed-distortion-class permissiveness (§III of the paper),
  not literally "time-varying filter"; a separate misattributed claim ("the
  paper notes... for HLS-CMDS specifically") is corrected — the 2006 paper
  predates HLS-CMDS and never mentions it; that inference is this project's
  own, applied from the paper's general noise/interference/artifact
  decomposition.
- Baseline 2 and 3 re-run post-fix on both the full 145 rows and the
  additive-only 36-row subset (§5.2) — first time Baseline 2 has numbers on
  the valid subset, making the ablation against Baseline 3 apples-to-apples
  for the first time. Result: Baseline 3 (no learned dictionary) matches or
  slightly beats Baseline 2 (learned dictionary) there — the pretrained
  dictionary isn't earning its keep on this dataset, a genuine finding, not
  a bug.

**Done in the Baseline 4 / MSSA session (2026-08-20, continued):**
- Baseline 4 implemented and cross-checked directly against `[ssa]`'s primary
  source (§5.2): multi-stage SSA, all four hyperparameters (L=50, 250 Hz
  split, 2%/50% thresholds) confirmed exact matches to the paper's §II. The
  250 Hz cardiac/respiratory split confirmed to sit correctly at this
  project's 4000 Hz sample rate (physiological Hz value, well under the
  2000 Hz Nyquist, no rescaling needed) — the item explicitly flagged for
  confirmation when this baseline was scoped.
- **New open item, not resolved**: Baseline 4's synthetic-set reproduction
  (§5.2) falls far short of the paper's own reported cardiac SDR/correlation
  (1.24 dB/64.8% vs. 26.4 dB/99.2%), while landing in the right range on
  respiratory SDR (6.81 vs. 5.3 dB) — an asymmetric gap. Candidate
  explanation not yet confirmed: Stage 1's literal per-RC peak-frequency
  threshold (this implementation's reading of "components with dominant
  frequencies below or equal to 250 Hz are classified as cardiac-related")
  may be more permissive than the paper's own "periodic structure analysis"
  phrasing implies, letting low-energy/noise-like RCs into the cardiac sum
  on both sides of the split. Needs further digging before trusting Baseline
  4's numbers as a faithful reproduction rather than a partial one.

**Still open:**
- **Statistics**: this memo's and `report/report.pdf`'s across-fold
  mean±std (e.g. Table 5's ±3.71 dB) understates pooled-row spread by an
  order of magnitude (pooled heart SAR is 6.30±39.38 dB, not 6.46±3.71) —
  report medians and/or bootstrap CIs alongside means once re-issued, and
  always state which level (per-row vs. per-fold) the dispersion is
  computed at.
- ~~Independently re-read Yaqub et al. (`[spectrotemporal]`) from the primary
  source rather than relying on the review's summary of the Reading List~~
  **Resolved (2026-08-24)** — see §2/§3's now-updated rows and
  `report/report.tex`'s new "Yaqub et al." subsection. `[nmfcrnn]`, `[ssa]`,
  and now `[spectrotemporal]` are all independently confirmed from primary
  sources. ~~`[aidriven]` (a Ph.D. dissertation) remains the one exception.~~
  **Resolved (2026-09-22)** — the thesis is posted in full as
  arXiv:2602.09210v1 and is now in `papers/`; §2's row is rewritten from it,
  and it corrected two things the review summary had wrong (the separation
  metrics, and which dataset the 26.8 dB belongs to). Every cited work is
  now cross-checked against its primary source.
- Resolve Baseline 4's cardiac-side reproduction gap (see above) — either by
  refining Stage 1's RC classification criterion or by confirming the paper
  is genuinely silent on this and the gap is inherent to the ambiguity.
- PL-NMF/LingoNMF (`[aidriven]`) remains the one separation method from §2's
  table not yet reproduced as a baseline.
- Decide classifier architecture and training protocol for 5.3, now scoped
  to a continuous separation-quality sweep (§4) rather than two fixed
  conditions.
- Decide whether Condition B reuses Condition A's trained classifier or
  retrains on separated audio.
- Repo hygiene: `HLS-CMDS` was an embedded git repo with no `.gitmodules`
  entry; it was removed from tracking and the dataset now lives flat in
  `HLS_CMDS/`. Two items remain: `HLS_CMDS/` is listed in `.gitignore` but its
  1,128 files are still tracked from before that rule existed (`git rm -r
  --cached` never ran), and `results/`, `report/` and `self_learning.md` are
  gitignored, so no number, figure or draft in this project is reachable from
  a clean checkout.
