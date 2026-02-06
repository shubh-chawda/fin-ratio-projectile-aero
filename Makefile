# Repo-root Makefile
# Usage:
#   make venv
#   make lock
#   make all
#   make ci
#   make test
#   make demo
#   make demo-plot
#   make clean

ROOT := $(abspath $(CURDIR))

VENV := $(ROOT)/.venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

REQ_TXT  := requirements.txt
REQ_LOCK := requirements.lock.txt
REQ_FILE := $(if $(wildcard $(REQ_LOCK)),$(REQ_LOCK),$(REQ_TXT))

VENV_STAMP := $(VENV)/.installed

.DEFAULT_GOAL := help

.PHONY: help venv install lock core timestep bootstrap all ci clean test demo demo-plot

help:
	@echo "Targets:"
	@echo "  make venv       - create venv + install deps (prefers requirements.lock.txt if present)"
	@echo "  make lock       - generate requirements.lock.txt from current venv"
	@echo "  make core       - generate core figures + fit drag model + model compare"
	@echo "  make timestep   - run timestep sensitivity study (DT, DT/2, DT/4)"
	@echo "  make bootstrap  - run bootstrap_k_eff (configurable via env vars)"
	@echo "  make test       - run pytest"
	@echo "  make demo       - run a quick CLI demo for fin_ratio=0.75"
	@echo "  make demo-plot  - run demo + save trajectory plot"
	@echo "  make all        - run core + bootstrap + timestep"
	@echo "  make ci         - faster settings for CI (small bootstrap)"
	@echo "  make clean      - remove generated outputs + processed data (keeps raw data)"

venv: $(VENV_STAMP)

$(VENV_STAMP): $(REQ_FILE)
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r $(REQ_FILE)
	$(PY) -m pip check
	@touch $(VENV_STAMP)

install: venv

lock: venv
	$(PY) -m pip freeze --all | sort > $(REQ_LOCK)
	$(PY) -m pip check
	@echo "Wrote: $(REQ_LOCK)"

core: venv
	$(PY) -m src.make_figures
	$(PY) -m src.fit_drag_model
	$(PY) -m src.model_compare

timestep: venv
	DT=0.003 $(PY) -m src.timestep_sensitivity

bootstrap: venv
	N_BOOT=200 DT=0.003 N_BISECT=25 $(PY) -m src.bootstrap_k_eff

test: venv
	$(PY) -m pytest -q

demo: venv
	$(PY) -m src.demo --fin-ratio 0.75

demo-plot: venv
	$(PY) -m src.demo --fin-ratio 0.75 --plot

all: core bootstrap timestep

ci: venv
	$(PY) -m src.make_figures
	$(PY) -m src.fit_drag_model
	$(PY) -m src.model_compare
	N_BOOT=40 DT=0.003 N_BISECT=25 $(PY) -m src.bootstrap_k_eff
	DT=0.003 $(PY) -m src.timestep_sensitivity
	$(PY) -m pytest -q
	$(PY) -m src.demo --fin-ratio 0.75

clean:
	rm -rf outputs
	rm -rf data/processed
	rm -f  $(VENV_STAMP)

