clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr .eggs/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/
	rm -fr .ruff_cache/

clean: clean-build clean-pyc clean-test

install:
	uv sync --dev

lint:
	uv run ruff check phylib

format:
	uv run ruff format phylib

format-check:
	uv run ruff format --check phylib

lint-fix:
	uv run ruff check --fix phylib

test: lint format-check
	uv run pytest --cov-report term-missing --cov=phylib phylib

test-fast:
	uv run pytest phylib

coverage:
	uv run coverage html

apidoc:
	uv run python tools/api.py

build:
	uv build

upload:
	uv publish

upload-test:
	uv publish --publish-url https://test.pypi.org/legacy/

dev: install lint format test

ci: lint format-check test build

.PHONY: clean-build clean-pyc clean-test clean install lint format format-check lint-fix test test-fast coverage apidoc build upload upload-test dev ci