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
        baseline1 baseline2 baseline3 baseline4 baseline5 synthetic-set \
        baseline12-synthetic first-sdr-table stats plots clean clean-pyc clean-all

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
	@echo "Separation baselines (src/baselines.py) -> results/baselines_report.html:"
	@echo "  make baselines     Baseline 1 + 2 + 3 + 4, each on full 145 rows AND the additive-only subset,"
	@echo "                     plus Baseline 1/4 on the SSA-paper-style synthetic set, ~20 min"
	@echo "  make baseline1     Baseline 1 only (bandpass filter, fast)"
	@echo "  make baseline2     Baseline 2 only (supervised NMF, slow — dictionary fitting per fold)"
	@echo "  make baseline3     Baseline 3 only (standard NMF, no learned dictionary — ablation vs. baseline2)"
	@echo "  make baseline4     Baseline 4 only (multi-stage SSA, reproducing Han & Quan ICSPS 2025)"
	@echo "  make baseline5     Baseline 5 only (EVMD, reproducing the edge-lung paper's separation stage)"
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
	@echo "Reports (all HTML, written to results/ — terminal only prints progress):"
	@echo "  make stats         per-class/per-location duration, sample rate, clipping stats"
	@echo "  make plots         waveform/spectrogram/donut-chart figures (headless, MPLBACKEND=Agg)"
	@echo ""
	@echo "Tests:"
	@echo "  make test          pytest over src/test_metrics.py src/test_split.py src/test_load_dataset.py"
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
	cd $(SRC) && ../$(PYTHON) -m pytest test_metrics.py test_split.py test_load_dataset.py -v

validate:
	cd $(SRC) && ../$(PYTHON) load_dataset.py

split:
	cd $(SRC) && ../$(PYTHON) split.py

eval-harness:
	cd $(SRC) && ../$(PYTHON) eval_harness.py

baselines:
	cd $(SRC) && ../$(PYTHON) baselines.py

baseline1:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import fit_bandpass_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 1 (bandpass)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline1_report.html', 'Baseline 1', 'Baseline 1 (bandpass)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline2:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import make_supervised_nmf_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 2 (supervised NMF)...'); \
results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline2_report.html', 'Baseline 2', 'Baseline 2 (supervised NMF)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline3:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import make_standard_nmf_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 3 (standard NMF)...'); \
results_df, fold_summary, cv_summary = cross_validate(make_standard_nmf_baseline(seed=0), n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline3_report.html', 'Baseline 3', 'Baseline 3 (standard NMF)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline4:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import fit_ssa_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 4 (MSSA)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_ssa_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline4_report.html', 'Baseline 4', 'Baseline 4 (MSSA)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

baseline5:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import fit_evmd_baseline; from report_utils import results_dir, write_cv_report; \
print('Running Baseline 5 (EVMD)...'); \
results_df, fold_summary, cv_summary = cross_validate(fit_evmd_baseline, n_folds=5, seed=0); \
p = write_cv_report(results_dir() / 'baseline5_report.html', 'Baseline 5', 'Baseline 5 (EVMD)', fold_summary, cv_summary); \
print(f'Report written to {p}')"

synthetic-set:
	cd $(SRC) && ../$(PYTHON) synthetic_mix.py

first-sdr-table:
	cd $(SRC) && ../$(PYTHON) first_sdr_table.py

baseline12-synthetic:
	cd $(SRC) && ../$(PYTHON) baseline12_synthetic_report.py

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
