.DEFAULT_GOAL := all

.PHONY: install
install:
    pipenv run pip install -U Cython
    pipenv install --dev

.PHONY: lint
lint:
 	pipenv run python -m flake8 .

.PHONY: test
test:
	pipenv run python -m unittest

.PHONY: all
all: test lint
