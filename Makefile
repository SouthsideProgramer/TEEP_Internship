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
        baseline1 baseline2 stats plots clean clean-pyc clean-all

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
	@echo "  make validate      load HS/LS/Mix CSVs, validate audio paths, write HTML reports"
	@echo "  make split         print the leakage-safe fold assignment + dictionary pool sizes"
	@echo "  make eval-harness  run the no-op baseline through the full k-fold harness (plumbing smoke test)"
	@echo ""
	@echo "Separation baselines (src/baselines.py):"
	@echo "  make baselines     Baseline 1 + 2, each on full 145 rows AND the additive-only subset, ~15 min"
	@echo "  make baseline1     Baseline 1 only (bandpass filter, fast)"
	@echo "  make baseline2     Baseline 2 only (supervised NMF, slow — dictionary fitting per fold)"
	@echo ""
	@echo "Reports:"
	@echo "  make stats         per-class/per-location duration, sample rate, clipping stats"
	@echo "  make plots         waveform/spectrogram/donut-chart figures (headless, MPLBACKEND=Agg)"
	@echo ""
	@echo "Tests:"
	@echo "  make test          pytest over src/test_metrics.py src/test_split.py src/test_load_dataset.py"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         remove generated reports/plots/CSVs (gitignored outputs)"
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
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import fit_bandpass_baseline; \
results_df, fold_summary, cv_summary = cross_validate(fit_bandpass_baseline, n_folds=5, seed=0); \
print(fold_summary.to_string(index=False)); print(); print(cv_summary.to_string())"

baseline2:
	cd $(SRC) && ../$(PYTHON) -c "from eval_harness import cross_validate; from baselines import make_supervised_nmf_baseline; \
results_df, fold_summary, cv_summary = cross_validate(make_supervised_nmf_baseline(seed=0), n_folds=5, seed=0); \
print(fold_summary.to_string(index=False)); print(); print(cv_summary.to_string())"

stats:
	cd $(SRC)/statistics && ../../$(PYTHON) audio_quality.py

plots:
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) audio_plotter.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) audio_spectrogram.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) donut_chart.py
	cd $(SRC)/visualization && MPLBACKEND=Agg ../../$(PYTHON) plot_per_class.py

clean:
	rm -f $(SRC)/dataset_validation.html $(SRC)/mix_pairing_validation.html
	rm -f $(SRC)/visualization/combined_plots.png
	rm -rf $(SRC)/visualization/plots
	rm -rf $(SRC)/statistics/audio_quality_reports

clean-pyc:
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
	rm -rf .pytest_cache

clean-all: clean clean-pyc
	rm -rf $(VENV)
