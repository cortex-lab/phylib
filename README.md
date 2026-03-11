# phylib
[![Build Status](https://img.shields.io/travis/cortex-lab/phylib.svg)](https://travis-ci.org/cortex-lab/phylib)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phylib.svg)](http://codecov.io/github/cortex-lab/phylib?branch=master)

Electrophysiological data analysis library used by [phy](https://github.com/kwikteam/phy/), a spike sorting visualization software, and [ibllib](https://github.com/int-brain-lab/ibllib/).


## Contribution

- create a local environment with `uv venv --python 3.12` and `uv sync --extra dev`
- run all tests using `uv run pytest phylib`
- PR to main
- update `CHANGELOG.md` and version in `phylib\__init__.py`
- publish to pypi:
```shell
uv build
twine upload dist/*
```
