.PHONY: test format lint

SHELL := /bin/bash
.ONESHELL:

PYENV_ROOT = $(HOME)/.pyenv
PYTHON_VERSION ?= 3.12.2
VENV = .venv
PYTHON = ./$(VENV)/bin/python3
PIP = ./$(VENV)/bin/pip3
PRE_COMMIT = ./$(VENV)/bin/pre-commit

DEV?=true
FETCH?=true

test:
	@$(PYTHON) -m pytest tests -s -x --cov=plant_segmenter --cov-config=.coveragerc -vv

lint:
	@$(PYTHON) -m flake8 plant_segmenter/*
	@$(PYTHON) -m flake8 dataset_handler/*
	# @$(PYTHON) -m flake8 --ignore=D,W503,E712 tests/*
	@$(PYTHON) -m mypy plant_segmenter --follow-imports=skip
	@$(PYTHON) -m mypy dataset_handler --follow-imports=skip

format:
	@$(PYTHON) -m black plant_segmenter/ dataset_handler/
	@$(PYTHON) -m isort plant_segmenter/* dataset_handler/*

pyenv:
	@export PYENV_ROOT=$(PYENV_ROOT)
	@bash ./setup_pyenv.sh
	@export PATH=$(PYENV_ROOT)/bin:$(PATH)
	@source ~/.bashrc
	@eval "$(pyenv init -)"
	@pyenv install -s $(PYTHON_VERSION)
	@pyenv global $(PYTHON_VERSION)

venv/bin/activate: pyenv requirements
	@echo "Using $(shell python -V)"
	@source ~/.bashrc
	@python3 -m venv $(VENV)
	@chmod +x $(VENV)/bin/activate
	@source ./$(VENV)/bin/activate
	@$(PIP) install --upgrade setuptools wheel
	@$(PIP) install numpy==1.26.4 Cython==3.0.10
	@$(PIP) install -r requirements/requirements.txt

venv: venv/bin/activate
	@source ./$(VENV)/bin/activate
	@echo "VIRTUAL ENVIRONMENT LOADED"

install: venv
