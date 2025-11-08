# Makefile for Employee Attrition Prediction project

.PHONY: help install clean lint format notebook
.DEFAULT_GOAL := help

VENV_NAME := venv
PYTHON := $(VENV_NAME)/Scripts/python
PIP := $(VENV_NAME)/Scripts/pip

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies and setup environment
	python -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

clean: ## Clean cache files and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

lint: ## Run linting
	$(PYTHON) -m flake8 src/
	$(PYTHON) -m mypy src/

format: ## Format code
	$(PYTHON) -m black src/
	$(PYTHON) -m isort src/

notebook: ## Start Jupyter notebook
	$(PYTHON) -m jupyter notebook notebooks/

requirements: ## Update requirements.txt from current environment
	$(PIP) freeze > requirements.txt

setup-dev: install ## Setup development environment
	$(PIP) install black flake8 mypy isort

all: clean install test ## Clean, install, and test everything