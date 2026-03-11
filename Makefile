clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-build clean-pyc

lint:
	uv run flake8 --jobs=1 phylib

test: lint
	uv run pytest --cov-report term-missing --cov=phylib phylib

coverage:
	uv run coverage html

apidoc:
	uv run python tools/api.py

build:
	uv build

upload:
	@echo "Build artifacts with 'uv build' and upload them with twine."
