# Code descriptions

Long module-level explanations that used to live as big comment blocks at the
top of each source file. Each file now keeps a short docstring (summary +
`Usage:`) and points here for the full context. Section headers match the
file paths.

## baselines.py

Baseline 0 (raw mixture, no separation -- the reference every other
baseline's SDR is read against) plus the report-generation glue that ties
Baselines 0-4 together into `results/baselines_report.html` and builds the
SSA-paper-style synthetic evaluation set. Baselines 1-5 each live in their
own module under `baseline/` (split out of what used to be one large
baselines.py -- see below); Baseline 6 lives in `convtasnet.py`.

## baseline/common.py

The handful of pieces literally reused by more than one baseline: the two
frequency bands (`HEART_BAND`, `LUNG_BAND`), the zero-phase Butterworth
bandpass (`_bandpass`), and Welch-PSD peak-frequency picking
(`_peak_frequency`). Anything used by only one baseline stays defined in
that baseline's own module -- see baseline3.py below for the one exception
(it imports Baseline 2's NMF machinery directly rather than duplicating it,
since it's explicitly an ablation of Baseline 2).

## baseline/baseline1.py

**Baseline 1: simple bandpass filtering.** Zero-training, non-adaptive: a fixed
Butterworth bandpass per source, applied identically to every mixture. Any
learned method (NMF, SSA, ...) should comfortably beat this; if it doesn't,
that's a sign of a bug, not a hard separation problem.

Cutoffs were chosen from the average Welch PSD of HS.csv/LS.csv's isolated
recordings (not just literature defaults): heart energy dominates below
~150-200 Hz, lung takes over from ~200 Hz up to ~700-800 Hz, and both fall
into the noise floor above ~1 kHz at this dataset's 4000 Hz sample rate. The
heart/lung bands below therefore overlap on purpose in the 150-200 Hz
region -- that overlap is real (both sources have genuine energy there) and
is exactly the "spectral overlap" failure mode this baseline is meant to
demonstrate, not an implementation bug.

## baseline/baseline2.py

**Baseline 2: supervised NMF**, adapting Han, Quan, Matuszewski & Corbett,
"Respiratory Disease Classification Using NMF-Enhanced Log-Mel Spectrograms
and Convolutional Recurrent Neural Networks," Sensors 2026, 26(13):4268,
doi:10.3390/s26134268 (papers/Respiratory_Disease_Classification_...pdf --
now actually cross-checked against the primary source, Sec. 3.2.3/3.2.4,
resolving PROTOCOL.md 5.2's earlier hedge). Two fixed dictionaries -- one per
source -- are learned from isolated H/L recordings via KL-divergence
multiplicative-update NMF, then frozen and used to solve for per-mixture
activations, per the classic supervised-NMF separation recipe (Smaragdis
2007). Paper notation, confirmed: Kr = respiratory (lung) dictionary rank =
20, Ki = interference (heart) dictionary rank = 10; 100 MU iterations to fit
each dictionary, 60 MU iterations to solve activations against a held-out
mixture with the dictionaries frozen. The paper's own auxiliary dictionary
corpus *is* HLS-CMDS's isolated heart/lung recordings (Sec. 3.2.3: "HLS-CMDS
recordings were used exclusively for dictionary learning") -- this also
resolves PROTOCOL.md 2's "conflicting" `[nmfcrnn]` dataset row: HLS-CMDS is
the auxiliary dictionary source, while ICBHI 2017 + Fraiwan CWLS (Sec. 4.1)
is the *classification* dataset the paper's accuracy numbers are reported
on. Not a conflict -- two different datasets for two different roles in the
same paper.

ADAPTATION, not literal reproduction: the paper's own pipeline is
single-sided -- it only ever reconstructs the *respiratory* (lung) signal
via M_r = V-hat_r / (V-hat_r + V-hat_i) and discards the interference
(heart) component entirely (Sec. 3.2.4); it never reports a separated heart
signal or any separation-quality metric (SDR/SNR/etc.) for either source,
only downstream classification accuracy/Macro-F1 on the enhanced respiratory
output. This project needs both sources back out (its own research question
is heart+lung separation quality, not respiratory-only enhancement for
classification), so this implementation extends the paper's mask to a
symmetric two-source split: mask_heart = heart_mag / (heart_mag +
lung_mag), mask_lung = 1 - mask_heart. The dictionary-learning and
activation-solving math is faithful to the paper; the two-sided
reconstruction is this project's own extension of it.

PIPELINE ORDER (Sec. 3.2.1, initially missed): the paper denoises every
snippet -- both the isolated dictionary-training recordings and the mixture
being separated -- with a 4th-order Butterworth bandpass (50-1800 Hz) *before*
STFT/NMF, to strip baseline drift and high-frequency acquisition noise.
That's a single broadband pre-filter shared by both sources, distinct in
purpose from Baseline 1's per-source 20-200/150-1000 Hz bands (which *are*
the separation, not a pre-filter for one). Added here (`DENOISE_BAND`,
applied in `_fit_dictionary` and both `separate()` closures in baselines.py)
to match the paper's actual pipeline order, not just its NMF math -- the
initial reproduction skipped this stage entirely. (`DENOISE_BAND` and the
NMF machinery now live in `baseline/baseline2.py`; `baseline/baseline3.py`
imports them directly rather than duplicating -- see below.)

LEAKAGE TRAP (see PROTOCOL.md 5.1/split.py): a third of HS.csv/LS.csv's
recordings are byte-identical to a heart/lung component of some Mix.csv
row -- the row's own ground truth. A naive reproduction that fits the
dictionaries on "all of HS.csv/LS.csv" therefore lets a mixture's own
ground-truth source into its separation dictionary. This implementation
only ever sees `hs_allowed`/`ls_allowed` -- eval_harness.cross_validate's
fold-safe pool, with every recording that appears in the held-out fold's
mixtures already excluded (same triplet-level split as S2-03) -- so its
score is expected to come in lower than a naive reproduction's. That lower
number is the correct one; a higher one would mean leakage, not a better
model.

## baseline/baseline3.py

**Baseline 3: standard NMF**, no learned dictionary (ablation against Baseline 2 /
S3-02; ref S3-03). Same total rank (Ki+Kr=30), STFT params, and DENOISE_BAND
pre-filter as Baseline 2, but W and H are both factorized directly out of
each held-out mixture's own (denoised) spectrogram -- no dictionary-learning
phase from isolated recordings, so hs_allowed/ls_allowed go unused, same as
Baseline 1. Keeping the denoising step identical to Baseline 2 matters here:
the point of this ablation is to isolate the effect of the *learned
dictionary* specifically, so every other stage of the pipeline (denoise,
STFT params, mask/reconstruction recipe) is held fixed between the two. This
isolates how much of Baseline 2's score comes from the learned/frozen
dictionary versus NMF factorization alone. The resulting components are
unlabeled by construction; they're assigned to heart/lung post-hoc by
spectral centroid (heart energy concentrated below ~200 Hz, per Baseline 1's
own PSD survey) -- a fixed physical prior, not a source label, so the method
stays genuinely blind.

## baseline/baseline4.py

**Baseline 4: multi-stage Singular Spectrum Analysis (MSSA)**, reproducing Han &
Quan, "Cardiorespiratory Sound Separation Using Singular Spectrum Analysis,"
2025 17th IEEE Int'l Conf. on Signal Processing Systems (ICSPS), doi:
10.1109/ICSPS66615.2025.11347745 (`papers/Cardiorespiratory_Sound_Separation_
Using_Singular_Spectrum_Analysis.pdf`). Zero-training like Baseline 1 --
`hs_allowed`/`ls_allowed` go unused, no leakage trap applies (there is no
fitting step at all). All four hyperparameters are stated explicitly in the
paper and used here as-is:

- `SSA_WINDOW_LENGTH = 50` (Sec. II.A: "a window length of 50 was selected
  to achieve a balance between processing speed and signal separability").
- `SSA_CARDIAC_SPLIT_HZ = 250.0` (Sec. II.B: "components with dominant
  frequencies below or equal to 250 Hz are classified as cardiac-related
  signals... S1 primarily occupies 100-200 Hz, while S2 extends up to
  250 Hz"). This is a physiological frequency in absolute Hz, not a value
  normalized to the paper's own sample rate, so it's valid at any sample
  rate whose Nyquist clears it -- including this project's 4000 Hz HLS_CMDS
  copy (Nyquist 2000 Hz, 8x above the split). The paper's own dataset is
  sourced from the same Torabi et al. HLS-CMDS descriptor paper this project
  cites (their ref [19] = this project's own PROTOCOL.md Sec. 1 citation),
  so there is no cross-dataset sample-rate mismatch here either.
- `SSA_EIGENVALUE_THRESHOLD_PCT` (Sec. II.B: "the SSA decomposition produces
  50 RC layers, an average contribution per mode is calculated as 2%...
  a threshold of 2% is set"). Derived as `100/SSA_WINDOW_LENGTH` rather than
  hardcoded, so it stays principled if the window length ever changes; it
  equals exactly 2.0 for L=50, matching the paper.
- `SSA_CORRELATION_THRESHOLD = 0.50` (Sec. II.B: "modes showing a
  correlation above 50% are additionally included").

Two-stage algorithm (Sec. II.A "Basic Algorithm" for the SSA math, II.B
"Proposed Method" for the two stages and thresholds):

- Stage 1 (cardiac): SSA-decompose the raw mixture (L=50) into 50
  reconstructed components (RCs) via trajectory-matrix embedding + SVD +
  diagonal averaging (Hankelization). Each RC's dominant frequency (Welch
  PSD peak) sorts it into cardiac (<=250 Hz, summed directly into the final
  heart_est -- Fig. 1 routes this branch straight to output, no stage-2
  refinement) or into a residual pool (>250 Hz) that feeds stage 2.
- Stage 2 (respiratory): SSA-decompose the residual (L=50 again) into a
  fresh 50 RCs. RCs whose relative eigenvalue contribution clears the 2%
  threshold are the initial "high-energy" respiratory set; each remaining
  RC is then Pearson-correlated against that set's sum and added if the
  correlation exceeds 50%. lung_est is the sum of the final selected set.
  (Interpretation choice, since the paper doesn't fully spell out what "the
  remaining modes" are correlated against: correlating each leftover RC
  against the sum of the already-selected high-energy modes is the natural
  reading of "the identified high-energy modes vs. the remaining modes" --
  flagged here the same way Baseline 2's two-sided-mask extension is
  flagged as this project's own interpretation, not literal paper text.)

## baseline/baseline5.py

**Baseline 5: EVMD (Enhanced Variational Mode Decomposition)**, reproducing
Sec. II.B of Puneet, Shankar, Koluguri & Srivastava, "Edge-Enabled Portable
Classifier for Lung Sounds Using Convolutional Neural Networks," IEEE
BioCAS 2025, doi:10.1109/BioCAS67066.2025.00016
(`papers/Edge-Enabled_Portable_Classifier_for_Lung_Sounds_Using_
Convolutional_Neural_Networks.pdf` -- `[edgelung]` in PROTOCOL.md). Sprint
ref **S4-01**, pulled forward from Sprint 4 into Sprint 3 (same pattern as
Baseline 3's S3-03 pull-forward). That paper runs EVMD-based lung isolation
directly on HLS-CMDS mixtures but reports no separation metric for either
source at all -- this baseline computing SDR/SIR/SAR for it is a direct
instance of the gap this project exists to close.

Zero-training, like Baselines 1/4 -- `hs_allowed`/`ls_allowed` unused, no
leakage trap applies.

**VMD itself** (`_vmd`): Variational Mode Decomposition (Dragomiretskiy &
Zosso, IEEE Trans. Signal Processing 2014) -- no VMD package is installed or
in `requirements.txt`, so this is implemented directly, the same way
Baseline 4's SSA was implemented from scratch. Standard frequency-domain
ADMM solve with mirror-padding at both signal ends (suppresses boundary
artifacts in the mode estimates, standard VMD practice). This part is fully
specified by the cited paper and not in question -- `test/test_baseline5.py`
checks its reconstruction fidelity directly.

**K-selection (Sec. II.B, `_evmd_select_k`)**: sweeps `K=2..10` at
`alpha=2000` (`EVMD_ALPHA`, `EVMD_K_MIN`/`EVMD_K_MAX` -- both stated
explicitly in the paper), stopping at the first `K` where the energy loss
coefficient (`_energy_loss_coefficient`, `||signal - sum(modes)||^2 /
||signal||^2`) clears `mu1=0.01` (`EVMD_MU1`) and every mode passes its
per-mode check.

**INTERPRETATION, flagged the same way Baseline 2's mask extension and
Baseline 4's Stage 2 correlation target are flagged** -- the paper's own
text ("must also satisfy the constraints on its frequency domain signature
and NPE ratio... in case of failure, the IMF's Kurtosis Index is assessed")
doesn't fully specify how the four criteria combine. This project's
best-faith reading, per the project lead's own framing when this baseline
was scoped ("the selection logic is described thinly, so an exact
reproduction may not be possible... if you hit the limit, write up what you
tried and where it broke"):

- A mode with Normalized Permutation Entropy (`_normalized_permutation_entropy`,
  Bandt-Pompe, embedding dimension `EVMD_NPE_EMBED_DIM=5` -- not stated in
  the paper) `<= mu2=0.4` (`EVMD_MU2`) passes outright (low-complexity,
  clearly structured).
- A mode with NPE `> mu2` additionally passes if BOTH (a) its dominant
  frequency (Welch PSD peak, reusing Baseline 4's `_peak_frequency`) falls
  inside a recognized heart or lung band (this project's own `HEART_BAND`/
  `LUNG_BAND`, from Baseline 1's PSD survey -- read as the paper's
  unspecified "frequency domain signature" check) AND (b) its NPE ratio
  (this mode's NPE over the highest NPE among the K modes at this candidate
  K) is `<= mu3=0.3` (`EVMD_MU3`).
- Failing that, a low (`<= mu4=0.05`, `EVMD_MU4`) normalized Kurtosis Index
  (`_kurtosis_index`) is treated as a fallback pass. Raw Fisher excess
  kurtosis is unbounded and routinely >>1 for physiological transients, so
  `mu4=0.05` can't sensibly threshold it directly -- this project reads
  `mu4` as applying to a normalized index instead
  (`1/(1+|excess kurtosis|)`, near 1 for Gaussian-like modes, near 0 for
  strongly impulsive ones), flagged as this project's own reading, not
  stated in the paper.
- If no `K` in `[2, 10]` gets every mode through every check, this falls
  back to `K=10` and reports non-convergence rather than forcing a pass --
  a genuine, documented result per the go-ahead above, not a bug to paper
  over.

**Reconstruction** (`evmd_separate`): the mode "corresponding to the
cardiac frequency band" is read as singular -- the one mode with the lowest
peak frequency (distinct from Baseline 4's MSSA, which sums *all*
cardiac-band RCs) -- lowpassed at 150 Hz (`EVMD_HEART_LOWPASS_HZ`, the
paper's own cutoff, via the new `_lowpass` helper) for `heart_est`;
`lung_est = mixed - heart_est` (paper: "the sound component of the heart is
subtracted from the original signal, leaving behind the residual lung
sound").

**Measured cost, not assumed**: the full K-sweep at this dataset's
15 s/4000 Hz recordings (60,000 samples/mixture) costs ~15 s/mixture,
dominated by VMD's ADMM iterations (`EVMD_MAX_ITER=100` -- this project's
own bounded choice, not stated in the paper; VMD literature commonly
converges within 100-200 iterations at tol=1e-6, and running the full sweep
over a synthetic set with n in the thousands makes runtime a real
constraint). This is why `first_sdr_table.py` evaluates Baseline 5's
synthetic-set column on a stratified subsample rather than the full
synthetic set every other baseline uses -- see that script and
`synthetic_mix.py` below.

## convtasnet.py

**Baseline 6: Conv-TasNet-lite** -- the first neural, end-to-end learned
separation baseline. Lives here rather than under `baseline/` since it's a
different kind of module (trains a model) rather than a fixed
separate_fn/fit_fn pair. See this module's own docstring for the full
sample-rate decision and architecture derivation; summarized here.

*Sample-rate decision*, made explicitly before any training code was
written: this dataset is natively 4000 Hz, so public Conv-TasNet/Sepformer
checkpoints (trained at 8-16 kHz) don't apply without resampling, and
resampling up invents no real information. Decision: train from scratch at
the native 4000 Hz. This is not a resolution limitation for this task --
heart energy sits at 20-200 Hz and lung at 100-1000 Hz (Baseline 1's own
Welch-PSD survey), comfortably under 4 kHz's 2000 Hz Nyquist.

*Architecture*: the same encoder/TCN-separator/decoder design as Conv-TasNet
itself (Luo & Mesgarani, IEEE/ACM TASLP 2019) and NeoSSNet (Poh et al., IEEE
OJEMB 2024 -- the closest prior work: also a masked Conv-TasNet-style model
separating heart/lung sound from one chest channel at 4 kHz), sized down
("lite") for this dataset's much smaller training pool: ~325K parameters vs.
Conv-TasNet's own ~5M-parameter speech config or NeoSSNet's 8.4M-parameter
transformer-augmented model. No transformer mask generator (NeoSSNet's own
addition) -- just the original stacked-dilated-TCN separator. Sigmoid mask
activation per source, independent (no unit-sum constraint) -- Conv-TasNet's
own ablation (Sec. IV-A) found sigmoid at least as good as a softmax
constraint, and this dataset's mixtures have a real noise/residual
component so heart+lung need not reconstruct the mixture exactly (see
`load_dataset.verify_additive_triplets`). Fixed source order (heart =
channel 0, lung = channel 1), not permutation-invariant training --
unlike Conv-TasNet's interchangeable speakers, heart and lung are
distinguishable classes here, matching NeoSSNet's own (non-PIT) choice.

*Training loop*: unlike Baselines 1-5 (zero-training or a fixed-dictionary
fit), this baseline trains a fresh network per fold from that fold's
leakage-safe `hs_allowed`/`ls_allowed` pool (same contract every baseline's
`fit_and_separate_fn` receives). Since Conv-TasNet needs paired (mixed,
heart, lung) examples rather than a spectral dictionary, training mixtures
are synthesized on the fly, reusing `synthetic_mix.py`'s own mixing recipe
(`mixed = a*(heart+lung) + noise`, gain log-uniform over the native-additive
gain range, noise drawn to a uniformly-sampled target SNR within
`SNR_SWEEP_DB`'s range) rather than inventing a separate one -- keeps
train-time mixtures drawn from the same distribution the model is evaluated
against. A file-level 80/20 split inside the allowed pool (never touching
the outer CV fold's held-out data) gives an internal validation set for
early stopping and LR scheduling (AdamW, LR halved after 4 epochs without
validation-SI-SDR improvement, best-checkpoint restore -- matching
NeoSSNet's own training recipe) rather than just returning the final
epoch's weights.

*Honest scope*: this is the first neural baseline, establishing the
training-loop infrastructure -- not a tuned, converged model. The dataset's
~50 recordings/class (fewer per fold once leakage exclusions apply) is
small for a from-scratch neural separator even with on-the-fly mixing
augmentation; `baseline6_report.py` prints each fold's actual training
diagnostics (epochs run, best validation SI-SDR, train/val pool sizes) and
measured wall-clock rather than assuming the shipped config
(`MAX_EPOCHS=25`, `STEPS_PER_EPOCH=40`, `BATCH_SIZE=8`) trained to
convergence.

`test/test_convtasnet.py` (12 tests): model forward-pass shape/finiteness
for arbitrary input lengths, SI-SDR loss sanity (scale invariance, identity
ceiling, uncorrelated-estimate floor), the augmented-batch sampler's output
shapes, and an end-to-end training-loop smoke test (tiny synthetic pool,
few epochs) that the `fit_and_separate_fn` contract actually works and a
too-small allowed pool raises rather than silently training on nothing.

## synthetic_mix.py

Synthetic mixing set (S1-09/S1-10/S1-13) -- replaces native Mix.csv pairs
as the primary evaluation substrate for `first_sdr_table.py`'s headline
SDR/SIR/SAR table. Motivation: only 36/145 native Mix.csv rows are actually
additive (`load_dataset.verify_additive_triplets`), too few to source a
real SDR-vs-difficulty sweep for the charter's C2 "knee point" analysis.

**Construction (S1-09) matches the native dataset's own model**: rather
than an arbitrary synthetic recipe, `mixed = a*(heart+lung) + noise` --
exactly the form `verify_additive_triplets` already shows native additive
rows follow (one scalar gain on the *sum*, not independent per-source
levels). `a` is drawn log-uniformly per (heart, lung) source pair from the
range empirically observed on the 36 native additive rows' own fitted gains
(`_native_gain_range`, computed at build time from real data, not
hardcoded) -- not the SSA paper's arbitrary 10x5 subsample
(`build_synthetic_mixes` in `baselines.py` is a separate, already-documented
reproduction of that paper's own Table I comparison and isn't reused here).

**Noise**: additive Gaussian, RMS-relative to `a*(heart+lung)`'s RMS --
matches Han & Quan's noise *type*. Diverges on *level*: they fix 2% RMS
(~34 dB SNR); this project sweeps `SNR_SWEEP_DB` (hard->easy) instead,
because the C2 knee-point analysis needs a continuous, controllable
difficulty axis a single fixed level can't give -- the sweep's top level is
flagged (`SNR_DB_CLOSEST_TO_HAN_QUAN`) as the closest point to their
condition. **Trimmed from 5 to 3 levels** (`-5, 15, 35` dB) after measuring
per-row cost across all 5 baselines showed a 5-level sweep would make the
full run multi-hour -- a disclosed, compute-driven scope reduction (see
`first_sdr_table.py`'s EVMD-subsample note for the same category of
tradeoff), not a silent one. Gain `a` is fixed per source pair across its
noise realizations, so difficulty varies along the noise axis only.

**Split (S1-13) is source-file-level, not triplet-level (supersedes S2-03
for this substrate)**: with full combinatorial pairing (every HS.csv
recording paired with every LS.csv recording), every heart file
transitively connects to every lung file through *some* mixture, collapsing
`split.py`'s connected-component/leak-group logic into one giant component
-- it simply doesn't produce meaningful folds here. Instead,
`assign_source_folds` assigns each of the 50 heart files and 50 lung files
independently to one of `n_folds` folds; a synthetic mixture `(h, l)` is
only used for held-out evaluation under fold `k` when
`fold(h) == fold(l) == k` -- required so a fold's dictionary pool (Baseline
2's leakage safety) excludes *both* sources of every mixture it's evaluated
on, not just one.

**Provenance**: every row records `heart_id`, `lung_id`, `gain_a`,
`snr_db`, `fold` -- required for the leakage tests in `test/test_synthetic_mix.py`
and so the generator itself is auditable. Audio arrays are *not* stored in
the provenance table (`build_synthetic_set` returns scalars only); rows are
synthesized on demand (`synthesize_row`) from a small (50+50 recording)
audio cache, keeping ~1500 rows cheap to hold in memory.

**Seed policy**: one top-level `SYNTHETIC_SEED`; per-row seeds derived
deterministically via `np.random.SeedSequence` from `(heart_id, lung_id)`
(gain) or `(heart_id, lung_id, snr_db)` (noise) through a stable MD5-based
integer hash (`_stable_seed` -- Python's built-in `hash()` is
process-randomized, so it can't be used for a seed that needs to reproduce
across runs).

**Validation (S1-10) -- does the construction actually reproduce the
native dataset's own additive mixtures, not just claim to by construction?**
`validate_against_native` takes each of the 36 native additive rows' *real*
heart/lung audio and *that row's own* fitted gain (from
`verify_additive_triplets`), builds `mixed_synth = a*(heart+lung)` (zero
noise -- the reproduction claim, not a noisy realization), writes it to a
temp WAV, and calls **`load_dataset.verify_additive_triplets()` itself**
(reused directly, per instruction, not reimplemented) against a mix_df
pointing at the real heart/lung files and this new synthetic mixed file.
Result: all 36 rows reproduce, with synthetic residuals landing in the same
~1e-4 cluster the genuine native rows occupy (not just barely under the
pass/fail threshold) -- the evidence that lets `first_sdr_table.py`'s
report claim the synthetic substrate is faithful to the dataset's own
additive mixtures, not an arbitrary process.

## eval_harness.py

K-fold evaluation harness for heart/lung separation models on the mix set.

Wires split.py's leakage-safe fold assignment into metrics.py's BSS Eval:
for each fold, a caller-supplied function fits a dictionary/model on that
fold's allowed HS/LS recordings only (i.e. excluding anything that also
appears in that fold's held-out mixtures) and returns a separation
function, which is then evaluated on the held-out mix rows. Results are
tagged by fold and source, then aggregated two ways:

- per-fold means (one row per fold x source)
- across-fold mean +/- std (the CV estimate of generalization performance)

## metrics.py

BSS Eval separation metrics (SDR / SIR / SAR) for the heart/lung mix set.

Thin wrapper around `mir_eval.separation.bss_eval_sources` [Vincent et al.,
2006], specialized for this dataset's 2-source case: every Mix.csv row mixes
one heart recording and one lung recording, so a separation model's output
is evaluated against those two ground-truth sources.

mir_eval 0.8 deprecated bss_eval_sources/_images in favor of a museval-style
API that isn't published yet; bss_eval_sources is still the correct,
actively-used implementation of the classic Vincent et al. 2006 metric, so
the module silences the (currently unactionable) FutureWarning it raises on
every call.

## split.py

Leakage-safe, triplet-level train/eval split for the HLS-CMDS mix set.

Why this exists: Mix.csv's 145 rows only draw from 89 distinct heart
recordings and 74 distinct lung recordings (some reused up to 6x across
different mixtures), and a third of those recordings are byte-identical to
a file also listed standalone in HS.csv/LS.csv -- the same recordings you'd
learn a separation dictionary from. Splitting by Mixed Sound ID, or fitting
a dictionary on "all of HS.csv/LS.csv", leaks ground truth into training:
a recording can sit in one mix row's evaluation fold while its literal
duplicate (or a mix row sharing it) sits in the dictionary-fitting pool.

The split unit here is therefore a *leak group*: mix rows are connected
whenever they share a heart or a lung recording (by audio content, not by
ID string -- HS.csv and Mix.csv use different ID namespaces for the same
underlying files), and the resulting connected components -- not individual
rows -- are what gets assigned to folds. Any HS.csv/LS.csv recording whose
content matches a recording in a held-out fold is excluded from that fold's
dictionary-fitting pool.

`assign_hs_folds()` (new, for `heart_classifier.py`'s Condition A) extends
this to assign HS.csv rows *themselves* a fold, not just decide whether
they're excluded from a fold's dictionary pool: a recording byte-identical
to a Mix.csv leak group's heart component inherits that leak group's fold
(the same boundary `dictionary_pool()` already enforces, reused rather than
redefined); recordings never reused in any mixture have no leak-group
constraint and are assigned by balanced round-robin, optionally stratified
by a caller-supplied column (the classifier's `class_group`) so no fold
ends up starved of a minority class.

## heart_classifier.py

PROTOCOL.md Sec. 5.3 Condition A: classification accuracy on HS.csv's own
clean/isolated heart sound recordings, on the same leak-group 5-fold split
every separation baseline already uses (`split.py`'s `assign_hs_folds()`).
The first accuracy number in the project -- everything before this is
SDR/SIR/SAR in dB -- and the anchor the charter's C2 sweep (PROTOCOL.md
Sec. 4) will be compared against once Condition B (classifying separated
audio) exists.

**Class-grouping decision** (blocking, made before any classifier code):
full 10-class Heart Sound Type classification is out -- S4 has only 2
recordings total, so under the leak-group split it can't appear in every
fold's training set at all, and AV Block/Tachycardia (n=3 each) are barely
better off. Grouped into Yaqub et al.'s (`[spectrotemporal]`) own 4-class
scheme for this same dataset (confirmed from the primary source, not the
review's summary): Normal (n=9), Murmur (n=24, the 4 murmur types),
Extra Sound (n=7, S3+S4), Rhythm Disorder (n=10, AFib+Tachycardia+AV Block).
This was chosen over collapsing to binary Normal/Abnormal specifically so
Condition A's number stays comparable to the 86-89% figures the project's
own motivating result (Yaqub's 89%->41% collapse, PROTOCOL.md Sec. 3) is
reported on -- see `heart_classifier.py`'s module docstring for the full
paragraph, which is also PROTOCOL.md Sec. 5.3's Methods text verbatim.

**Architecture**: 13 MFCCs pooled to mean+std over time (26-dim feature
vector) + an RBF-kernel, class-balanced SVM -- one architecture,
deliberately (this is a measuring instrument, not a contribution; see
BACKLOG.md's S7-06 note for the planned second-architecture robustness
check). Chosen over a CNN because n=50 recordings (as few as ~35-40/fold
in training) is far too little data for a deep model without the result
being dominated by overfitting noise, and MFCC+SVM is PROTOCOL.md Sec.
5.3's own stated fallback for exactly this regime. No hyperparameter
search was run (`C=1.0`, `gamma='scale'` are sklearn's defaults).

`cross_validate_classifier()` fits a fresh classifier per fold on that
fold's training split and reports accuracy + Macro-F1 per fold plus a
95%-CI aggregate across folds (n=5 -- indicative, not decisive, per
PROTOCOL.md Sec. 6). First measured result (`results/heart_classifier_
report.html`): accuracy 58.0% ± 7.3% (95% CI), Macro-F1 0.43 ± 0.07,
with per-class-group recall Normal 44%, Murmur 88%, Extra Sound 29%,
Rhythm Disorder 20% -- a majority-class bias toward Murmur (n=24/50), the
expected shape of the small-n class-imbalance problem the grouping
decision reduced but didn't eliminate.

**Failure mode analysis (confusion matrices)**: `confusion_counts()` (raw,
pooled across all folds' held-out predictions -- every recording is
predicted exactly once in its own held-out fold, so pooling never
double-counts), `confusion_recall_pct()` (row-normalized, the same
convention Yaqub et al.'s own confusion matrices use, Figs. 13-16, for
direct visual comparability), and `top_confusions()` (the single most
common *wrong* prediction per true class_group, excluding the correct
diagonal cell -- the specific failure mode behind a low recall number, not
just the rate). `plot_confusion_heatmap()` renders a row-normalized-color
heatmap with count + row-% annotations to `results/plots/heart_classifier_
confusion_matrix.png`.

Measured result (`results/heart_classifier_report.html`): the dominant
failure mode is **both minority classes' errors pulling toward Murmur**,
not a symmetric confusion spread -- Extra Sound is predicted Murmur more
often than it's predicted correctly (43% vs. 29% recall), and Rhythm
Disorder is predicted Murmur nearly as often as any other single outcome
(40% vs. 20% recall), while Murmur itself is barely ever predicted for
something else (0% confused as Normal or Extra Sound, only 8% as Rhythm
Disorder). Normal is the one class whose dominant confusion runs the other
direction, toward Rhythm Disorder (33%) rather than Murmur. Consistent
with Murmur's training-set dominance (n=24/50) pulling the decision
boundary toward it despite `class_weight='balanced'` reweighting the SVM's
loss -- reweighting the loss doesn't guarantee balanced predictions when
the underlying 26-dim MFCC-summary feature space gives the four classes
limited separability to begin with.

`test/test_heart_classifier.py` (18 tests): class-group mapping totals
match the documented decision, `assign_hs_folds()`'s leak-safety property
(a recording reused in Mix.csv gets the same fold as that mixture's leak
group -- mirrors `test_split.py`'s load-bearing dictionary-pool test),
every fold's training set covers all 4 class groups, feature-vector shape,
cross-validation output shape/range, and the confusion-matrix functions
(counts sum to class sizes, diagonal matches per-class recall, row-%
sums to 100, dominant-confusion lookup excludes the diagonal, plus a
small hand-checked example). All passing.

Also exposes `predict_one(clf, y, sr)` and (renamed from `_aggregate_ci95`)
`aggregate_ci95()` as public functions -- part of the "backend contract"
(alongside `train_fold_classifiers`) `condition_b.py`/
`sdr_accuracy_curve.py`/`sdr_knee_point.py` call generically via a
`backend` parameter (S7-06, see `heart_classifier_cnn.py` below), so
either architecture plugs into the same downstream pipeline unchanged.

## heart_classifier_cnn.py

S7-06: the second classifier architecture for the C2 knee-point
robustness check -- no longer optional once C2's knee point, not
classification accuracy on its own, is the paper's headline result. The
charter's "one architecture only" rule held when the classifier was a
measuring instrument for a secondary result; once the knee point IS the
paper, a reviewer's first question is whether it's a property of
separation quality or of the one architecture that measured it.

**Architecture**: log-mel spectrogram (40 mel bins, same `N_FFT`/
`HOP_LENGTH` as `heart_classifier.py`) + a shallow CNN (2 conv blocks,
global average pooling, one linear head -- a few thousand parameters).
Deliberately as different from Architecture 1 as possible while staying
"small and well-understood" (§5.3's own phrase): a 2D time-frequency
representation instead of pooled MFCC summary statistics, a
gradient-trained model instead of a kernel method -- two flavors of SVM
wouldn't isolate whether a knee-point disagreement is about separation or
about one specific decision boundary.

Reuses everything architecture-agnostic from `heart_classifier.py`
directly: `CLASS_GROUPS`, `HEART_TYPE_TO_GROUP`, `add_class_group`,
`assign_classifier_folds`, `aggregate_ci95`, and the confusion-matrix
functions (`confusion_counts`/`confusion_recall_pct`/`top_confusions`/
`plot_confusion_heatmap`, generic over any true/pred `predictions_df`).
Only feature extraction, the model (`_ShallowCNN`, a raw `nn.Module`),
and the fit/predict loop (`CNNClassifier`, a thin `.fit`/`.predict`
adapter matching sklearn's duck-typed interface) are architecture-specific.

**Backend contract**: exposes `train_fold_classifiers(hs_df, n_folds)`
and `predict_one(clf, y, sr)`, the same two functions
`heart_classifier.py` exposes -- lets `condition_b.py`/
`sdr_accuracy_curve.py`/`sdr_knee_point.py` run their entire weight-
sharing/fold-alignment/knee-point pipeline against either architecture
via a `backend` module parameter (default `heart_classifier`), the same
"same call contract, no shared base class" pattern the six separation
baselines already use.

**Real measured result, a genuine disclosed limitation** (`results/
heart_classifier_cnn_report.html`, first configuration, no hyperparameter
search): Condition A accuracy 30.0% ± 8.8% (95% CI, 5-fold) vs.
Architecture 1's 58.0% -- and its confusion matrix shows a collapse
toward predicting **Normal** for 44/50 recordings (100% Normal recall,
but Murmur/Extra Sound/Rhythm Disorder all routing overwhelmingly to
Normal instead of their own class). This is a different majority-attractor
than Architecture 1's own pull toward Murmur -- the two architectures
don't just perform differently, they fail differently, which is itself
informative for the robustness check (genuinely distinct decision
boundaries, not the same shortcut twice). It's also exactly the
overfitting risk `heart_classifier.py`'s own docstring predicted when it
chose SVM over CNN for n=50 -- tested directly here, not dismissed, and
the risk turned out real. Practical consequence: a knee-point
disagreement measured against this first CNN configuration can't yet
cleanly separate "the knee is architecture-dependent" from "this CNN
config isn't a reliable enough classifier to support the comparison" --
`sdr_knee_point.py`'s own report states this caveat automatically
whenever Architecture 2's isolated accuracy is below 40%.

`test/test_heart_classifier_cnn.py` (8 tests, using a `max_epochs`
override for speed -- mirrors `convtasnet.py`'s own testability pattern):
feature-shape consistency across recordings, `CNNClassifier` fit/predict
round-trip, predicting before fitting raises, seeded reproducibility, the
backend contract (`train_fold_classifiers`/`predict_one`), and end-to-end
cross-validation output shape. All passing.

## degradation.py

S6-01 (pulled forward twice: S6 -> S5 -> S2, see BACKLOG.md for both
decisions). Designs and validates the controlled separation-degradation
scheme PROTOCOL.md Sec. 5.3.1 uses to sweep separation quality
continuously, for the charter's C2 knee-point curve.

**The scheme**: `degrade_toward_ground_truth(ground_truth, estimate,
alpha)` linearly blends a real separation baseline's own estimate toward
its ground truth -- `degraded = g + alpha*(s - g)` -- so alpha=0 is clean
and alpha=1 is that baseline's real output. `degrade_row()` applies this to
one source (heart or lung) of a mix row while holding the *other* source's
estimate fixed at its real separated value, and scores the result through
`metrics.evaluate_heart_lung()` directly -- no parallel metric definition.
`sdr_sweep()` runs a grid of alphas and returns the *measured* SDR/SIR/SAR
at each, since alpha itself has no physical meaning and different rows/
methods reach a given alpha at different real SDRs.

**The x-axis decision** (the actual point of this ticket, see PROTOCOL.md
Sec. 5.3.1 for the full memo): the curve's x-axis is measured SDR, not
synthetic SNR (SNR is undefined for Yaqub et al.'s own bandpass-separated
point, so it can't be the unit that places their result on the curve) --
real-method points and this scheme's dense sweep share one SDR axis.

**Real finding from validating on real data** (not glossed over): a
uniform alpha grid gives a badly uneven spread of measured SDR -- most of
the curve's informative range falls inside alpha in [0, 0.1], since dB is
most sensitive to small absolute error near a perfect match. Added
`find_alpha_for_target_sdr()` (bisection over alpha, valid because
SDR(alpha) is monotonic -- checked, not assumed) so a caller can request an
evenly-spaced target-SDR grid directly instead.

Also `find_alphas_for_target_sdrs()` -- a batched version added for
`sdr_sweep.py` (S6-02): generating the sweep needs several target SDRs per
(row, source), and a from-scratch 40-iteration bisection per target
multiplies BSS-Eval's own per-call cost (mir_eval's permutation search
over full-length audio) by the number of targets for no reason. This
builds one coarse, log-spaced probe table per (row, source) -- reusing the
same monotonicity property, not a new assumption -- then refines each
target from an already-narrow bracket instead of the full [0, 1] range.
Measured ~1.8x wall-clock speedup on `test_sdr_sweep.py`'s tiny case
(agreement with the unbatched function verified directly, not assumed --
`TestFindAlphasForTargetSdrsBatched` in `test_degradation.py`).

`test/test_degradation.py` (14 tests): interpolation endpoints/midpoint,
truncation, source isolation, the monotonicity property on real dataset
audio (Baseline 1's actual output on a native additive row), the
root-finder's accuracy plus its endpoint-clamping behavior, and the
batched version's agreement with the unbatched one plus its own clamping.
All passing.

Scope: this module designs and validates the degradation *mechanism*
itself; it does not build the accuracy-vs-SDR curve, which needs
Condition B (PROTOCOL.md Sec. 5.3, classifying separated audio) to exist
first -- still open.

## sdr_sweep.py

S6-02, pulled forward from S6 into S5 alongside S6-01 (same holiday
reasoning -- see `degradation.py`'s section above and BACKLOG.md). Actually
*generates* the controlled SDR sweep dataset S6-01 designed: runs every one
of the six separation baselines (S4-03) once per fold over the 36 native
additive rows (same substrate `degradation.py`'s own validation used),
caches each baseline's real `heart_est`/`lung_est` to `results/
sdr_sweep_cache/<baseline>/<mixed_id>_<source>.wav`, then uses
`degradation.py`'s `find_alphas_for_target_sdrs()` to hit a fixed
`TARGET_SDR_GRID_DB = (25, 20, 15, 10, 5, 0, -5)` dB grid for every
(baseline, row, source).

Does not touch the classifier at all -- confirmed by import graph, not
just description: this module imports `degradation`, `load_dataset`,
`metrics`, `report_utils`, `split`, and the six baseline modules, nothing
from `heart_classifier.py` -- so it ran independently of, and in parallel
with, S5-01 (the classifier).

**Design, mirroring `synthetic_mix.py`'s own split** between a cheap
provenance table and audio reconstructed on demand: separation itself (the
expensive part, especially Baselines 2/5/6) is cached once per (baseline,
fold), never re-run per target-SDR grid point; `synthesize_sweep_row()`
reconstructs the actual degraded waveform on demand from that cache plus a
provenance row's own alpha (a single cheap `degrade_toward_ground_truth`
call) -- the swept dataset a future classification step consumes is this
provenance table (`results/sdr_sweep_provenance.csv`) plus the cache
directory, not gigabytes of pre-materialized degraded audio.

`test/test_sdr_sweep.py` (7 tests, deliberately cheap -- a 4-row subset and
only Baseline 1): schema/row-count, alpha validity, the
`clamped_to_baseline_floor` flag agreeing with alpha, cached `.wav` files
existing where expected, target-vs-alpha monotonicity surfaced through the
generated table itself, and `synthesize_sweep_row()`'s reconstruction
matching what `build_sdr_sweep()` recorded (including an alpha=0 case that
must reproduce ground truth exactly regardless of the cached estimate).
All passing.

**Measured, not assumed**: see BACKLOG.md's 2026-08-27 entry for the actual
generation run's coverage (rows generated, wall-clock, per-baseline
target-vs-achieved accuracy, and how many grid points clamped to each
baseline's own real SDR floor) -- `results/sdr_sweep_report.html` has the
full breakdown.

## condition_b.py

Condition B of PROTOCOL.md Sec. 5.3 -- evaluates the Condition A classifier
on real separated audio (S6-02's cache) instead of isolated ground truth.
This is the controlled-conditions reproduction of Yaqub et al.'s
89%->41% collapse (PROTOCOL.md Sec. 3), run with all six of this project's
separation baselines instead of their one bandpass filter.

**Weight-sharing decision** (the open item PROTOCOL.md Sec. 5.3 flagged,
resolved here): the *same* trained classifier weights per fold, not
retrained on separated audio -- matching Yaqub et al.'s own methodology
exactly (their Experiment 3 model *is* their Experiment 4 model). Isolates
the separation method as the only variable between the two conditions.

**Fold-basis alignment** (a real correctness trap this module exists to
avoid): `heart_classifier.assign_classifier_folds()` defaults to folding
HS.csv against the *full* 145-row Mix.csv, but `sdr_sweep.py` folds its
36-row native-additive substrate independently (a different row set
produces a different leak-group partition, even with the same seed --
matching the precedent already established in `first_sdr_table.py`'s own
`mix_df=valid_mix_df` override of `cross_validate`). Evaluating "the same
trained weights" requires both to agree on what fold k *is* --
`build_condition_b_fold_basis()` computes the native-additive fold
assignment once and passes it into `assign_classifier_folds()`'s
`mix_df_with_folds` parameter (added to `heart_classifier.py` for this),
so the classifier and the cached separated audio are guaranteed to share
one leak-safe partition, not two independently-computed ones that happen
to use the same seed.

`evaluate_condition_b()` classifies, for every native-additive row: the
row's own isolated ground-truth heart recording (`Mix.csv`'s own
`heart_audio_path` -- not a lookup back into HS.csv, since not every
mix-heart recording has a standalone HS.csv counterpart at all) once, and
each baseline's real separated `heart_est` (loaded straight from
`sdr_sweep.estimate_path()`'s cache, not through the degradation machinery
-- Condition B evaluates each method's actual output, not an interpolated
point) -- both through `heart_classifier.train_fold_classifiers()`'s
fold-k weights. `summarize_by_baseline()`/`summarize_by_fold()` give
accuracy/Macro-F1 per condition; `paired_delta_vs_isolated()` computes
PROTOCOL.md Sec. 5.4's primary result (`accuracy_isolated -
accuracy_separated`, paired by fold) with a paired t-test per baseline
(n=5 folds -- indicative, not decisive, per Sec. 6). The `__main__` block
also renders a confusion matrix (reusing `heart_classifier.py`'s
`confusion_counts`/`confusion_recall_pct`/`top_confusions`/
`plot_confusion_heatmap` directly -- these are generic over any
true/pred `predictions_df`, not HS.csv-specific) per condition.

`test/test_condition_b.py` (11 tests, deliberately cheap -- a 4-row subset,
Baseline 1's real bandpass output cached the same way `sdr_sweep.py`
would): the fold basis matches `assign_folds()` computed the identical
way, the classifier never trains on a recording in its own held-out fold,
schema/row-count, isolated rows carry no SDR while separated rows do, and
-- the load-bearing check -- the isolated and separated predictions for
the same row come from literally the same fitted classifier object. All
passing.

**`backend` parameter (S7-06)**: `evaluate_condition_b()`/
`evaluate_no_separation()` accept a `backend` module (default
`heart_classifier`) and call `backend.train_fold_classifiers()`/
`backend.predict_one()` instead of hardcoding `heart_classifier`'s own --
lets `heart_classifier_cnn.py` (Architecture 2) run the identical
isolated/separated/no-separation comparison unchanged.
`TestBackendParameter` (3 new tests) proves this with a fake stub backend
that always predicts a fixed label, confirming the passed backend is
actually used rather than silently ignored, without paying for a real
CNN training run.

**Measured, not assumed**: see BACKLOG.md's 2026-08-27 entry for the real
run's accuracy numbers (isolated vs. each baseline's separated accuracy,
the paired delta and significance test, and the dominant failure mode per
condition) -- `results/condition_b_report.html` has the full breakdown.

## sdr_accuracy_curve.py

S6-03: measures downstream classification accuracy at *every* point in
S6-02's controlled SDR sweep, not just each baseline's real (alpha=1)
output the way `condition_b.py` does -- this is the actual accuracy-vs-SDR
curve PROTOCOL.md Sec. 5.3.1 designed the x-axis for. Scoped to Sprint 6's
three working days (24/29/30 Sep -- 25/28 Sep are holidays); design (S6-01)
and generation (S6-02) already happened in S5, so this script's only job
is measurement, reusing everything else as-is.

Consumes S6-02's outputs directly (`results/sdr_sweep_provenance.csv` +
`results/sdr_sweep_cache/`) and Condition B's weight-sharing setup
(`condition_b.build_condition_b_fold_basis()` / `heart_classifier.
train_fold_classifiers()`) -- same trained weights per fold, never
retrained on degraded audio, consistent with every other measurement in
this family. Only `source == "heart"` rows are measured; the sweep's
lung-source rows exist for BSS Eval's own 2-source bookkeeping and have no
classification counterpart.

**Scripted to run unattended**, per the brief: `measure_accuracy_at_each_
sdr_point()` checkpoints to `results/sdr_accuracy_curve.csv` every
`CHECKPOINT_EVERY` rows, and a re-run loads that file first and skips any
`(baseline, mixed_id, target_sdr)` triple already present -- safe to
interrupt (e.g. across the Sprint 6 holiday gap) and resume without
redoing completed work. Also carries a defensive consistency check: each
row's own recorded fold (from S6-02's provenance) must agree with an
independently recomputed classifier fold basis, or the script raises
rather than silently using a possibly-wrong fold's classifier.

`summarize_accuracy_curve()` aggregates to one row per `(baseline,
target_sdr)`: accuracy, n, mean achieved SDR (the actual x-axis value --
target_sdr is a request, achieved_sdr is what was measured), and the
fraction of points clamped to that baseline's own real-output floor.
`plot_accuracy_curve()` renders accuracy vs. measured SDR, one line per
baseline, with the isolated-ground-truth accuracy as a dashed reference.

`test/test_sdr_accuracy_curve.py` (12 tests, deliberately cheap -- a 4-row
subset, Baseline 1 only): only heart rows get measured, output round-trips
through disk, a simulated interrupt-and-resume produces identical
predictions to a fresh full run while only re-computing the missing rows
(checked via a monkeypatched call-counter on `synthesize_sweep_row`, not
just "the final answer happens to match"), the fold-consistency assertion
actually raises on a tampered row, and the summary's per-group accuracy
matches direct computation. All passing.

**`backend` parameter (S7-06)**: `measure_accuracy_at_each_sdr_point()`
accepts a `backend` module (default `heart_classifier`) and calls
`backend.train_fold_classifiers()`/`backend.predict_one()`; `output_path()`
is backend-aware too (`results/sdr_accuracy_curve_<backend>.csv` for
anything other than the default), so the two architectures' checkpoints
never collide. `TestBackendParameter` (2 new tests) checks both with a
fake stub backend.

**Measured, not assumed**: see BACKLOG.md's 2026-08-27 entry for the real
run's curve (accuracy at each SDR point, per baseline, and whether a knee
point is visible) -- `results/sdr_accuracy_curve_report.html` has the full
breakdown.

## sdr_knee_point.py

S6-04, the project's headline figure per the Sprint 0 re-scope: plots the
accuracy-vs-SDR curve and identifies the knee point. Treats every upstream
module (`split.py`, the six baselines, `heart_classifier.py`,
`degradation.py`, `sdr_sweep.py`, `condition_b.py`,
`sdr_accuracy_curve.py`) as scaffolding for this one plot -- the first
place this dataset's classification accuracy and separation-quality SDR
live on the same axis (Yaqub et al. report accuracy with no SDR attached;
this project's own Sec. 5.2 tables report SDR with no downstream accuracy
attached).

**Knee-point definition** -- operationalizes PROTOCOL.md Sec. 4's own
wording ("the SDR below which separation actively hurts classification
accuracy, relative to not separating at all") directly, rather than an
arbitrary curvature/elbow heuristic: the SDR at which a baseline's
accuracy(SDR) curve crosses `condition_b.evaluate_no_separation()`'s
accuracy (classifying the raw, unseparated mixture directly), walking from
high SDR to low. `find_knee_point()` returns one of four statuses
(`crossed` -- interpolated between the two bracketing measured points;
`always_above`/`always_below` -- no knee in the measured range;
`noisy_crossing` -- a crossing exists but not in the expected high->low
direction, flagged distinctly rather than silently treated as clean) so a
reader can tell a real knee from small-n noise. No smoothing is applied --
interpolating a curve through `len(TARGET_SDR_GRID_DB)` points (each
averaging a handful of rows) would manufacture precision the data doesn't
support, and n per point is always reported alongside the estimate.

`condition_b.py` gained `evaluate_no_separation()` (+`NO_SEPARATION_LABEL`)
for this: the third reference point PROTOCOL.md Sec. 4's C2 framing needs
verbatim, classifying each row's raw mixed recording directly with the
same fold-appropriate weights Condition A/B use.
`condition_b.summarize_by_baseline()`'s output ordering was extended
(`NO_SEPARATION_LABEL`, `ISOLATED_LABEL`, then each separation baseline)
so all three condition types summarize through one function.

**Honesty about partial data**: the `__main__` block checks which of the
six baselines actually have cached separated audio (`sdr_sweep.py`'s
`estimate_path()`) before plotting, and if `results/sdr_sweep_
provenance.csv` doesn't exist yet (S6-02 still running), falls back to
`sdr_sweep.build_provenance_from_cache()` for whichever baselines *are*
ready rather than blocking on the full six-baseline generation run. The
rendered figure's own title and report explicitly state how many
baselines are present vs. still pending -- a real but partial figure says
so, rather than silently looking complete.

`test/test_sdr_knee_point.py` (12 tests, synthetic curves -- pure
arithmetic, no audio I/O): a clean interpolated crossing, exact-equality
edge case, always-above/always-below, the noisy-crossing case detected
distinctly, row-order independence, multi-baseline dispatch, and (S7-06)
`TestCompareKneePoints` -- agreement within tolerance, disagreement beyond
it, the inconclusive case when one side has no clean crossing,
baseline-intersection behavior, and the at-least-two-backends guard. All
passing.

**Cross-architecture robustness check (S7-06)**: `run_pipeline_for_
backend()` runs the entire curve-and-knee-point pipeline (S6-03
measurement + both reference points + knee detection) for one classifier
backend; `present_baselines_for()` factors out the "which baselines have
cached audio" check `__main__` used inline before. `compare_knee_points()`
is the actual verdict -- per baseline, whether every architecture's
`knee_sdr` falls within `tolerance_db` (default 3 dB) of each other.
`agrees` is `True`/`False` only when *every* architecture found a clean
`"crossed"` knee; `None` (inconclusive, not a silent "no") when any side
found no clean crossing -- a real disagreement is never confused with "one
architecture had nothing to compare." The `__main__` block now runs both
`heart_classifier` and `heart_classifier_cnn`, and states a caveat
automatically whenever Architecture 2's isolated accuracy falls below
40% (true as of this session's first CNN configuration -- see
`heart_classifier_cnn.py`'s own section above), so a knee-point
disagreement isn't over-read while the second architecture itself is
still unreliable.

**Measured, not assumed**: see BACKLOG.md's 2026-08-27 entry for the
actual curve, knee points, and how many of the six baselines were present
when it ran -- `results/sdr_knee_point_report.html` has the full
breakdown.

## compute_cost.py

MACs (multiply-accumulate operations) and parameter counts per method --
the computational-cost dimension no table in this project had reported
yet, matching Yaqub et al.'s own Table 5 (Params/GFLOPs/model size) and
PROTOCOL.md's Edge-Enabled-paper deployment-hardware discussion (PYNQ-ZU
FPGA, ~15 W), applied to this project's own six separation baselines plus
the Condition A classifier, on this dataset's real 15s/4000Hz (60,000-
sample) signal length. Independent of every other module built this
session -- doesn't touch the sweep, the classifier's fold basis, or any
background generation job, so it ran to completion immediately regardless
of what else was in flight.

**Two explicit precision tiers**, never blurred into one confident-looking
number: "exact" (literal matrix-multiply/conv-layer shapes taken directly
from each method's own real code, or measured from an actual forward
pass) for Baseline 1 (bandpass), Baselines 2/3 (NMF -- exact MU-update
matmul shapes), Baseline 6 (Conv-TasNet-lite -- measured via forward hooks
on every real `Conv1d`/`ConvTranspose1d`, not analytically re-derived
stride/padding/dilation arithmetic), and the classifier (a real fold's
fitted SVM, support-vector count read directly, not guessed); "estimate"
(standard textbook complexity formulas -- reduced-SVD FLOPs, ADMM
elementwise-loop counts) for Baseline 4 (MSSA) and Baseline 5 (EVMD),
whose real cost has no closed-form matmul shape to count exactly.

**Real measured result** (`results/compute_cost_report.html`, per-mixture
inference cost on a real 60,000-sample recording):

| method | params | MACs/inference | precision |
|---|---|---|---|
| Baseline 1 (bandpass) | 0 | 2.4M | exact |
| Condition A classifier (MFCC+SVM) | 40 support vectors | 8.1M | exact |
| Baseline 2 (supervised NMF) | 7,710 (frozen dictionary) | 219M | exact |
| Baseline 3 (standard NMF) | 0 (transductive) | 1.16G | exact |
| Baseline 4 (MSSA) | 0 | 1.20G | estimate |
| Baseline 6 (Conv-TasNet-lite) | 325,465 | 1.96G | exact |
| Baseline 5 (EVMD) | 0 | 5.22G | estimate |

Two findings worth calling out: (1) Baseline 6's measured param count
(325,465) independently reproduces the ~325K figure BACKLOG.md's own
2026-08-25 session already reported from PyTorch's own summary -- a real
cross-check, not a coincidence of rounding. (2) Baseline 3 costs ~5.3x
more MACs per mixture than Baseline 2 despite being the "simpler" ablation
(no learned dictionary) -- because it is fully transductive (every one of
its 160 MU iterations updates both W and H fresh per mixture, vs.
Baseline 2's 60 activation-only iterations against an already-frozen
dictionary), giving BACKLOG.md's existing "B2 is inductive, B3 is
transductive" fairness caveat a concrete computational-cost number to go
with it, not just a qualitative distinction. EVMD (already known to be
the slowest baseline by wall-clock, ~15s/mixture) is confirmed the most
expensive by MACs too, by a wide margin (~2,175x more than the bandpass
floor) -- consistent with, not contradicting, its own measured wall-clock
ranking.

`test/test_compute_cost.py` (14 tests): each exact formula checked against
a hand-computable small case or an independent recomputation from the same
real constants; each estimate formula checked against its own stated
inputs; Conv-TasNet's param count cross-checked against a direct
`model.parameters()` sum and against the previously-reported ~325K
ballpark; the classifier's support-vector count cross-checked against a
freshly-fitted model; and the real, expected Baseline-3-costs-more-than-
Baseline-2 and EVMD-is-the-most-expensive-classical-baseline orderings
both checked directly rather than assumed. All passing.

## latency.py

The empirical wall-clock/CPU-time counterpart to `compute_cost.py`'s
analytic MACs/params table -- completes this project's own version of
Yaqub et al.'s Table 5 (Params/GFLOPs/Model size/**Inference time**) for
the six separation baselines plus the Condition A classifier. "Desktop"
contrasts deliberately with the Edge-Enabled paper's PYNQ-ZU FPGA
deployment target (PROTOCOL.md Sec. 2) -- this measures inference latency
on the workstation this project runs on, with no claim about how these
numbers translate to edge hardware.

**Uniform protocol** (`time_calls()`), identical across every method
regardless of what else is running on the machine: one untimed warm-up
call, `N_TRIALS` timed repetitions (fewer for EVMD specifically --
`N_TRIALS_SLOW` -- since its own K=2..10 sweep already costs ~15s/call by
itself, and adding full repetitions to an already-expensive method
disproportionately loads the machine for no extra protocol rigor), median
+ IQR reported for both wall-clock (`time.perf_counter()`) and process
CPU time (`time.process_time()`, summed across the calling process's
threads). One-time setup cost (NMF dictionary fitting on a small 5-file
pool, Conv-TasNet-lite training with a reduced epoch count -- training
quality doesn't affect *inference* speed, matching `test_convtasnet.py`'s
own testability pattern) is fit once, untimed, before the timed region.

**Machine-load honesty, built in from the start, not an afterthought**:
`os.getloadavg()` is recorded at measurement time and stated in the
report; when load exceeds CPU count, the report's own language switches
to an explicit caveat (relative rankings between methods stay meaningful
since every method is timed by the same code; absolute wall-clock values
should be read as noisy upper bounds, not clean single-tenant numbers;
CPU-time is offered as the more trustworthy comparison basis under
contention -- less sensitive to *other processes'* scheduling, though not
entirely immune to cache/memory-bandwidth contention).

`test/test_latency.py` (7 tests): `time_calls()`'s own protocol logic
checked with fast, deterministic synthetic functions (correct call count
across warm-up + trials, non-negative medians/IQRs, a known-duration
`time.sleep()` case confirming the wall-clock/CPU-time distinction
actually holds), plus two real, cheap methods (bandpass, the classifier)
exercised end-to-end to confirm `measure_*_latency()` wires real calls
into `time_calls()` correctly -- the four expensive methods (both NMF
baselines, MSSA, EVMD, Conv-TasNet) share the exact same tested protocol
function, so they aren't re-exercised in the automated suite to avoid
adding load during an already heavily-contended session. All passing.

**Measured, not assumed**: see BACKLOG.md's 2026-08-27 entry for the real
per-method latency numbers and the recorded machine load at measurement
time -- `results/latency_report.html` has the full breakdown.

## statistics/audio_quality.py

Audio quality + per-class/per-location statistics for the HLS-CMDS dataset.

Scans every .wav file referenced by HS.csv, LS.csv, and Mix.csv (via
load_dataset.py) and reports, per file:

- duration (s)
- sample rate (Hz)
- channel count
- clipping (fraction of samples pinned at full-scale +/-32767)

Then aggregates those per-file properties by class (Heart/Lung Sound Type)
and by recording Location.

## visualization/audio_plotter.py

Audio Waveform Plotter -- visualizes and plots time-domain waveforms of
audio files and plays them back via IPython. Designed for heart/lung sound
data but works with any audio file.

Features:

- Plot multiple waveforms in one figure.
- Automatically adjust for the number of audio files.
- Save the figure as an image file.
- Play the audio file directly using IPython.

## visualization/audio_spectrogram.py

Mel-Spectrogram Plotter -- plots the Mel-spectrograms of multiple audio
files using `librosa` and `matplotlib`, processing audio files and
displaying them in a vertically oriented figure.

Features:

- Generate and plot Mel-spectrograms for multiple audio files.
- Customize frequency range and Mel band limits.
- Include a color bar for better interpretation of dB scale values.

## visualization/donut_chart.py

Sound Types Donut Chart Visualization -- generates a multi-layered donut
chart to visualize the number of different heart, lung, and mixed sounds in
a dataset, using `matplotlib` to create concentric donut layers with a
legend showing the total counts for each sound type.

Features:

- Three concentric donut charts: inner (lung sounds), middle (heart
  sounds), outer (mixed sounds).
- Consistent color mapping across all layers for each sound type.
- Displays the count for each sound type within the chart.
