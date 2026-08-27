\
# TEEP_Internship — heart/lung sound separation pipeline
#
# All targets assume `make install` has been run once (creates .venv and
# installs requirements.txt into it). Override PYTHON to use a different
# interpreter (e.g. `make test PYTHON=python3` for the conda audio_env).

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
SRC        := src
TORCH_INDEX := https://download.pytorch.org/whl/cu121

.PHONY: help all venv install test validate split eval-harness baselines \
        baseline1 baseline2 baseline3 baseline4 baseline5 baseline6 synthetic-set \
        baseline12-synthetic first-sdr-table heart-classifier degradation-scheme sdr-sweep condition-b \
        sdr-accuracy-curve sdr-knee-point compute-cost heart-classifier-cnn latency sdr-compute-plane \
        stats plots clean clean-pyc clean-all

.DEFAULT_GOAL := help

help:
	@echo "Setup:"
	@echo "  make install       create $(VENV) and install requirements.txt"
	@echo ""
	@echo "  make all           validate + split + eval-harness + test + stats + plots + baselines"
	@echo "                     (~15-20 min, dominated by baseline2's per-fold NMF dictionary fitting,"
	@echo "                     now run twice each -- full 145 rows and the additive-only subset)"
	@echo ""
	@echo "Dataset / pipeline sanity checks:"
	@echo "  make validate      load HS/LS/Mix CSVs, validate audio paths, write HTML reports to src/"
	@echo "  make split         leakage-safe fold assignment + dictionary pool sizes -> results/split_report.html"
	@echo "  make eval-harness  no-op baseline through the full k-fold harness (plumbing smoke test) -> results/"
	@echo ""
	@echo "Separation baselines (src/baselines.py: Baseline 0 + report glue; src/baseline/: Baselines 1-5) -> results/baselines_report.html:"
	@echo "  make baselines     Baseline 1 + 2 + 3 + 4, each on full 145 rows AND the additive-only subset,"
	@echo "                     plus Baseline 1/4 on the SSA-paper-style synthetic set, ~20 min"
	@echo "  make baseline1     Baseline 1 only (bandpass filter, fast)"
	@echo "  make baseline2     Baseline 2 only (supervised NMF, slow — dictionary fitting per fold)"
	@echo "  make baseline3     Baseline 3 only (standard NMF, no learned dictionary — ablation vs. baseline2)"
	@echo "  make baseline4     Baseline 4 only (multi-stage SSA, reproducing Han & Quan ICSPS 2025)"
	@echo "  make baseline5     Baseline 5 only (EVMD, reproducing the edge-lung paper's separation stage)"
	@echo ""
	@echo "Baseline 6 (src/convtasnet.py) -> results/baseline6_report.html:"
	@echo "  make baseline6     Conv-TasNet-lite, first neural model -- trains from scratch per fold"
	@echo "                     (native 4000 Hz, no resampling), synthetic + native columns, ~5-10 min"
	@echo ""
	@echo "Synthetic mixing set (S1-09/S1-10/S1-13) -> results/synthetic_mix_report.html:"
	@echo "  make synthetic-set   build the synthetic set + validate it reproduces the 36 native additive rows"
	@echo ""
	@echo "First SDR/SIR/SAR table (all 5 baselines, synthetic + native + Han & Quan columns):"
	@echo "  make first-sdr-table -> results/first_sdr_sir_sar_table.html (slow -- dominated by"
	@echo "                          Baseline 4/5's per-row cost at ~1500 synthetic rows, ~1-2 hrs)"
	@echo "  make baseline12-synthetic -> results/baseline1_2_synthetic_report.html (Baselines 1+2"
	@echo "                          only, synthetic + native columns, no Han & Quan column, faster)"
	@echo ""
	@echo "Condition A classifier (src/heart_classifier.py) -> results/heart_classifier_report.html:"
	@echo "  make heart-classifier  MFCC + RBF-SVM on isolated HS.csv audio, same leak-group 5-fold"
	@echo "                         split as the separation baselines; 4 grouped classes (Normal/"
	@echo "                         Murmur/Extra Sound/Rhythm Disorder), accuracy + Macro-F1 with a"
	@echo "                         95% CI, per-fold and per-class-group breakdowns"
	@echo ""
	@echo "Controlled degradation scheme (src/degradation.py, S6-01) -> results/degradation_scheme_report.html:"
	@echo "  make degradation-scheme  validate the alpha-interpolation SDR sweep on real audio"
	@echo "                           (monotonicity + target-SDR root-finder), see PROTOCOL.md Sec. 5.3.1"
	@echo ""
	@echo "Generated SDR sweep dataset (src/sdr_sweep.py, S6-02) -> results/sdr_sweep_report.html:"
	@echo "  make sdr-sweep     run all 6 separation baselines once each over the 36 native additive"
	@echo "                     rows, cache their real heart_est/lung_est (results/sdr_sweep_cache/),"
	@echo "                     and apply the S6-01 degradation scheme to hit a 25..-5 dB target grid --"
	@echo "                     the dataset Sprint 6's classification-and-plotting step consumes."
	@echo "                     Does not touch the classifier. Slow (bisection-heavy + EVMD/Conv-TasNet"
	@echo "                     separation cost) -- run in the background."
	@echo ""
	@echo "Condition B (src/condition_b.py, reproduces Yaqub's 89%->41% collapse) -> results/condition_b_report.html:"
	@echo "  make condition-b   requires results/sdr_sweep_cache/ (make sdr-sweep first). Evaluates the"
	@echo "                     SAME per-fold trained classifier from Condition A on isolated ground"
	@echo "                     truth vs. each of the 6 baselines' real separated output; paired delta"
	@echo "                     + significance test + confusion matrices per condition."
	@echo ""
	@echo "Accuracy-vs-SDR curve (src/sdr_accuracy_curve.py, S6-03) -> results/sdr_accuracy_curve_report.html:"
	@echo "  make sdr-accuracy-curve   requires make sdr-sweep first. Measures the Condition B"
	@echo "                            classifier at every point in the SDR sweep (not just each"
	@echo "                            baseline's real output) -- the actual C2 knee-point curve."
	@echo "                            Checkpointed to results/sdr_accuracy_curve.csv -- safe to"
	@echo "                            interrupt and re-run; already-measured points are skipped."
	@echo ""
	@echo "Headline figure (src/sdr_knee_point.py, S6-04) -> results/sdr_knee_point_report.html:"
	@echo "  make sdr-knee-point   accuracy-vs-SDR curve + knee point per baseline (the SDR below"
	@echo "                        which separation stops beating not separating at all). Works off"
	@echo "                        whichever baselines have cached separated audio so far -- renders"
	@echo "                        a real but explicitly partial figure if S6-02 isn't fully done."
	@echo ""
	@echo "Second architecture / robustness check (src/heart_classifier_cnn.py, S7-06):"
	@echo "  make heart-classifier-cnn   Architecture 2 (log-mel + shallow CNN) Condition A, matching"
	@echo "                              heart_classifier.py's own report -> results/"
	@echo "                              heart_classifier_cnn_report.html. sdr-knee-point (above) runs"
	@echo "                              both architectures and compares their knee points."
	@echo ""
	@echo "Desktop latency (src/latency.py, uniform protocol) -> results/latency_report.html:"
	@echo "  make latency   wall-clock + CPU-time inference latency per method (warm-up + timed"
	@echo "                 repetitions, median+IQR); records and discloses ambient machine load."
	@echo ""
	@echo "SDR vs. compute plane (src/sdr_compute_plane.py) -> results/sdr_compute_plane_report.html:"
	@echo "  make sdr-compute-plane   heart SDR (native additive rows) vs. MACs and vs. desktop"
	@echo "                           latency, one point per separation baseline, with Pareto-"
	@echo "                           dominance marked on each axis. Reuses this project's own"
	@echo "                           already-measured SDR/MACs/latency numbers -- fast."
	@echo ""
	@echo "Compute cost (src/compute_cost.py) -> results/compute_cost_report.html:"
	@echo "  make compute-cost   MACs and parameter counts per method (6 separation baselines +"
	@echo "                      the Condition A classifier), on this dataset's real 15s/4000Hz"
	@echo "                      signal length -- fast, independent of every other target above."
	@echo ""
	@echo "Reports (all HTML, written to results/ — terminal only prints progress):"
	@echo "  make stats         per-class/per-location duration, sample rate, clipping stats"
	@echo "  make plots         waveform/spectrogram/donut-chart figures (headless, MPLBACKEND=Agg)"
	@echo ""
	@echo "Tests:"
	@echo "  make test          pytest over src/test/ (all test_*.py: metrics, split, load_dataset,"
	@echo "                     synthetic_mix, baseline5/EVMD, convtasnet)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         remove generated reports (results/, src/*_validation.html)"
	@echo "  make clean-pyc     remove __pycache__/.pytest_cache"
	@echo "  make clean-all     clean + clean-pyc + remove $(VENV)"

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt --extra-index-url $(TORCH_INDEX)

all: validate split eval-harness test stats plots baselines

test:
	cd $(SRC) && ../$(PYTHON) -m pytest test/ -v

validate:
	cd $(SRC) && ../$(PYTHON) load_dataset.py

split:
	cd $(SRC) && ../$(PYTHON) split.py

eval-harness:
	cd $(SRC) && ../$(PYTHON) eval_harness.py

baselines:
	cd $(SRC) && ../$(PYTHON) baselines.py

baseline1:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baseline.baseline1 import fit_bandpass_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 1 (bandpass)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline1_report.html', 'Baseline 1', 'Baseline 1 (bandpass)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline2:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baseline.baseline2 import make_supervised_nmf_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 2 (supervised NMF)...'); \
results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline2_report.html', 'Baseline 2', 'Baseline 2 (supervised NMF)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline3:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baseline.baseline3 import make_standard_nmf_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 3 (standard NMF)...'); \
results_df, fold_summary, cv_summary = cross_validate(make_standard_nmf_baseline(seed=0), n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline3_report.html', 'Baseline 3', 'Baseline 3 (standard NMF)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline4:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baseline.baseline4 import fit_ssa_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 4 (MSSA)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_ssa_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline4_report.html', 'Baseline 4', 'Baseline 4 (MSSA)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline5:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baseline.baseline5 import fit_evmd_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 5 (EVMD)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_evmd_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline5_report.html', 'Baseline 5', 'Baseline 5 (EVMD)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline6:
	cd $(SRC) && ../$(PYTHON) baseline6_report.py

synthetic-set:
	cd $(SRC) && ../$(PYTHON) synthetic_mix.py

first-sdr-table:
	cd $(SRC) && ../$(PYTHON) first_sdr_table.py

baseline12-synthetic:
	cd $(SRC) && ../$(PYTHON) baseline12_synthetic_report.py

heart-classifier:
	cd $(SRC) && ../$(PYTHON) heart_classifier.py

degradation-scheme:
	cd $(SRC) && ../$(PYTHON) degradation.py

sdr-sweep:
	cd $(SRC) && ../$(PYTHON) sdr_sweep.py

condition-b:
	cd $(SRC) && ../$(PYTHON) condition_b.py

sdr-accuracy-curve:
	cd $(SRC) && ../$(PYTHON) sdr_accuracy_curve.py

sdr-knee-point:
	cd $(SRC) && ../$(PYTHON) sdr_knee_point.py

latency:
	cd $(SRC) && ../$(PYTHON) latency.py

sdr-compute-plane:
	cd $(SRC) && ../$(PYTHON) sdr_compute_plane.py

compute-cost:
	cd $(SRC) && ../$(PYTHON) compute_cost.py

heart-classifier-cnn:
	cd $(SRC) && ../$(PYTHON) heart_classifier_cnn.py

stats:
	cd $(SRC)/statistics && ../../$(PYTHON) audio_quality.py

plots:
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) audio_plotter.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) audio_spectrogram.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) donut_chart.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) plot_per_class.py

clean:
	rm -f $(SRC)/dataset_validation.html $(SRC)/mix_pairing_validation.html
	rm -rf results

clean-pyc:
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
	rm -rf .pytest_cache

clean-all: clean clean-pyc
	rm -rf $(VENV)
