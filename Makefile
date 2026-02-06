# Root Makefile (repo-level)
# Usage:
#   make venv
#   make lock
#   make all
#   make ci
#   make test
#   make demo
#   make demo-plot
#   make clean

SUBDIR := fin-aero-ee-repo
ROOT   := $(abspath $(CURDIR))

VENV      := $(ROOT)/$(SUBDIR)/.venv
PYROOT    := $(VENV)/bin/python
PIPROOT   := $(VENV)/bin/pip

# After: cd fin-aero-ee-repo
PYSUB := .venv/bin/python

REQ_TXT  := $(SUBDIR)/requirements.txt
REQ_LOCK := $(SUBDIR)/requirements.lock.txt
REQ_FILE := $(if $(wildcard $(REQ_LOCK)),$(REQ_LOCK),$(REQ_TXT))

# Stamp so deps re-install when requirements change
VENV_STAMP := $(VENV)/.installed

.DEFAULT_GOAL := help

.PHONY: help venv install lock core timestep bootstrap all ci clean test demo demo-plot

help:
	@echo "Targets:"
	@echo "  make venv       - create venv + install deps (prefers requirements.lock.txt if present)"
	@echo "  make lock       - generate fin-aero-ee-repo/requirements.lock.txt from current venv"
	@echo "  make core       - generate core figures + fit drag model + model compare"
	@echo "  make timestep   - run timestep sensitivity study (DT, DT/2, DT/4)"
	@echo "  make bootstrap  - run bootstrap_k_eff (configurable via env vars)"
	@echo "  make test       - run pytest"
	@echo "  make demo       - run a quick CLI demo for fin_ratio=0.75"
	@echo "  make demo-plot  - run demo + save trajectory plot"
	@echo "  make all        - run core + bootstrap + timestep"
	@echo "  make ci         - faster settings for CI (small bootstrap)"
	@echo "  make clean      - remove generated outputs (keeps raw data)"

venv: $(VENV_STAMP)

$(VENV_STAMP): $(REQ_FILE)
	python3 -m venv $(VENV)
	$(PYROOT) -m pip install --upgrade pip
	$(PIPROOT) install -r $(REQ_FILE)
	$(PYROOT) -m pip check
	@touch $(VENV_STAMP)

install: venv

# Generate / refresh the lockfile (run after venv is ready)
lock: venv
	cd $(SUBDIR) && $(PYSUB) -m pip freeze --all | sort > requirements.lock.txt
	cd $(SUBDIR) && $(PYSUB) -m pip check
	@echo "Wrote: $(REQ_LOCK)"

core: venv
	cd $(SUBDIR) && $(PYSUB) -m src.make_figures
	cd $(SUBDIR) && $(PYSUB) -m src.fit_drag_model
	cd $(SUBDIR) && $(PYSUB) -m src.model_compare

timestep: venv
	cd $(SUBDIR) && DT=0.003 $(PYSUB) -m src.timestep_sensitivity

bootstrap: venv
	cd $(SUBDIR) && N_BOOT=200 DT=0.003 N_BISECT=25 $(PYSUB) -m src.bootstrap_k_eff

test: venv
	cd $(SUBDIR) && $(PYSUB) -m pytest -q

demo: venv
	cd $(SUBDIR) && $(PYSUB) -m src.demo --fin-ratio 0.75

demo-plot: venv
	cd $(SUBDIR) && $(PYSUB) -m src.demo --fin-ratio 0.75 --plot

all: core bootstrap timestep

ci: venv
	cd $(SUBDIR) && $(PYSUB) -m src.make_figures
	cd $(SUBDIR) && $(PYSUB) -m src.fit_drag_model
	cd $(SUBDIR) && $(PYSUB) -m src.model_compare
	cd $(SUBDIR) && N_BOOT=40 DT=0.003 N_BISECT=25 $(PYSUB) -m src.bootstrap_k_eff
	cd $(SUBDIR) && DT=0.003 $(PYSUB) -m src.timestep_sensitivity
	cd $(SUBDIR) && $(PYSUB) -m pytest -q
	cd $(SUBDIR) && $(PYSUB) -m src.demo --fin-ratio 0.75

clean:
	rm -rf $(SUBDIR)/outputs/figures
	rm -rf $(SUBDIR)/figures/timestep_sensitivity
	rm -f  $(SUBDIR)/data/processed/*.csv
	rm -f  $(VENV_STAMP)
