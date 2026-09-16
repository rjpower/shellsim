.PHONY: check format lint review setup_pre_commit test

PYTHON ?= python3

format:
	$(PYTHON) infra/pre-commit.py --all-files --fix

lint:
	$(PYTHON) infra/pre-commit.py --all-files

review:
	$(PYTHON) infra/pre-commit.py --review

test:
	$(PYTHON) infra/ci/run_tests.py

check: lint test

setup_pre_commit:
	git config core.hooksPath .githooks
	@echo "Configured core.hooksPath=.githooks"
