# Makefile for Research Monorepo

VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

.PHONY: venv
venv: 
	source $(VENV_NAME)/bin/activate

.PHONY: init-venv
init-venv: clean-venv
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install wheel
	$(PIP) install -e .

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV_NAME)
