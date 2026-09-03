.PHONY: check format lint setup_pre_commit test

PYTHON ?= python3

format:
	$(PYTHON) infra/pre-commit.py --all-files --fix

lint:
	$(PYTHON) infra/pre-commit.py --all-files

test:
	$(PYTHON) infra/ci/run_tests.py

check: lint test

setup_pre_commit:
	git config core.hooksPath .githooks
	@echo "Configured core.hooksPath=.githooks"
