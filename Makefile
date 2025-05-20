.PHONY: clean clean-test clean-pyc clean-build clean-docs help test lint build dist install publish

# Find Python path
PYTHON := $(shell which python3)

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "build - build the package (no clean)"
	@echo "dist - clean and package for distribution"
	@echo "install - install the package to the active Python's site-packages"
	@echo "publish - package and upload a release to PyPI"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -fr .pytest_cache/
	rm -fr .coverage
	rm -fr htmlcov/

lint:
	$(PYTHON) -m flake8 src tests

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

dist: clean build

install: clean
	$(PYTHON) -m pip install -e .

publish: dist
	$(PYTHON) -m twine check dist/*
	$(PYTHON) -m twine upload dist/*

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"