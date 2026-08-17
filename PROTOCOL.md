# Protocol Memo: Separated vs. Isolated Accuracy on HLS-CMDS

Status: draft — open items flagged inline. Date: 2026-08-17.

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
| AI-Driven Cardiorespiratory Signal Processing (LingoNMF / PL-NMF) | HLS-CMDS | PL-NMF (parallel multi-layer NMF, heart/lung-tuned) + LLM-derived fundamental frequency | not reported (paired t-test on an unspecified quality proxy) | clustering + anomaly detection, **not** accuracy | "statistically significant gains," p<0.01 | not stated |
| Spectrotemporal Deep Learning for Heart Sound Classification | PhysioNet 2016 | none (raw PCG) | n/a | binary normal/abnormal classification | "high accuracy" | not stated |
| Respiratory Disease Classification (NMF-enhanced log-mel + CRNN) | ICBHI 2017 + Fraiwan CWLS | NMF-based respiratory enhancement | **none** (no SNR/SDR/SI-SDR/correlation/spectral-distortion/error-energy) | multi-class respiratory disease classification | 96.14±0.50% acc, 94.05±1.21 Macro-F1; denoised-ablation Macro-F1 84.98±22.0 over 10 seeds | **recording-level**, not subject-level (authors' own flagged limitation) |
| Edge-Enabled Portable Lung-Sound Classifier (ESP32 + PYNQ-ZU) | **HLS-CMDS** | proposed lung isolation from mixtures | **none** — no SNR, SDR, correlation, or spectral-error figure | lung sound classification | not detailed / separation step "reports no result" | not stated |
| Performance Measurement in Blind Audio Source Separation | — (methods paper) | — | defines SDR/SIR/SAR (BSS Eval) | n/a | n/a | n/a |
| Cardiorespiratory Sound Separation Using SSA | not stated in summary — **needs confirming** | two-stage Singular Spectrum Analysis (L=50) | SDR, STOI, ground-truth correlation, vs. 5 baselines | **none** — no classification | respiratory SDR 5.34 dB (MSSA) vs. 4.64 dB (SSA); cardiac 26.44 dB both stages; correlation 80.5%/99.2% (MSSA) vs. 10.2%/50.0% (NMF baseline) | n/a |

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

Of the six papers, exactly one — **Edge-Enabled Portable Lung-Sound
Classifier** — evaluates on HLS-CMDS *and* runs a classifier on
mixture-separated audio. It reports a classification result but **zero**
separation-quality metrics, and (per the summary reviewed) does not compare
that accuracy against classifying the same recordings' isolated,
un-mixed ground truth. The SSA paper reports proper separation metrics
(SDR/STOI/correlation) but no classification task at all. The NMF+CRNN paper
reports classification accuracy with real numbers but on a different dataset
(ICBHI+Fraiwan, not HLS-CMDS), with no separation metrics, and its own authors
flag a recording-level (leakage-prone) split.

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

**Does separating heart/lung sound from a mixture, before classification,
measurably reduce normal/abnormal detection accuracy relative to classifying
the isolated recording directly — and does that reduction track separation
quality (SDR/SIR/SAR)?**

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
`metrics.evaluate_dataset`. **Open item:** which separation method — the
harness is method-agnostic (`fit_and_separate_fn` is a plug-in), so this is a
choice, not a blocker. Candidates from the literature above: NMF (baseline,
several papers compare against it), SSA (best-reported respiratory SDR
in the papers reviewed, 5.34 dB), or PL-NMF/LingoNMF if reproducible from
the AI-Driven paper's description.

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

- Confirm full citations (author/venue/year) for all six papers above — only
  titles and content summaries were available when drafting this memo.
- Confirm the SSA paper's dataset (not stated in the summary reviewed).
- Decide separation method for 5.2 (NMF / SSA / other).
- Decide classifier architecture and training protocol for 5.3.
- Decide whether Condition B reuses Condition A's trained classifier or
  retrains on separated audio.
