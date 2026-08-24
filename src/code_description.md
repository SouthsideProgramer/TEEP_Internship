# Code descriptions

Long module-level explanations that used to live as big comment blocks at the
top of each source file. Each file now keeps a short docstring (summary +
`Usage:`) and points here for the full context. Section headers match the
file paths.

## baselines.py

Baseline separation methods for the heart/lung mix set -- the "does the real
method even beat a filter" sanity floor referenced in PROTOCOL.md 5.2/8.

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
initial reproduction skipped this stage entirely.

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
specified by the cited paper and not in question -- `test_baselines.py`
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
`snr_db`, `fold` -- required for the leakage tests in `test_synthetic_mix.py`
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
