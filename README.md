# phylib
[![CI](https://github.com/cortex-lab/phylib/actions/workflows/ci.yml/badge.svg)](https://github.com/cortex-lab/phylib/actions/workflows/ci.yml)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phylib.svg)](http://codecov.io/github/cortex-lab/phylib?branch=master)

Electrophysiological data analysis library used by [phy](https://github.com/kwikteam/phy/), a spike sorting visualization software, and [ibllib](https://github.com/int-brain-lab/ibllib/).


## Raw data through SpikeInterface

With `pip install phylib[spikeinterface]`, `dat_path` in `params.py` can name a
[SpikeInterface](https://github.com/SpikeInterface/spikeinterface) recording instead of a flat
binary file: a provenance `.json` (from `recording.dump_to_json()`) or a `.zarr` folder (from
`recording.save(format='zarr')`). The recording's preprocessing chain is computed lazily, so
remote data is read without being downloaded or exported. Traces are raw and unscaled, in the
recording's dtype. The recording sets the sample rate and channel count, so keep `sample_rate`
and `n_channels_dat` in `params.py` consistent with it.

## Contribution

- participation in phylib is governed by the project [Code of Conduct](CODE_OF_CONDUCT.md)
- run all tests using pytest `pytest phylib`
- PR to master
- keep `phylib/__init__.py` on the next development version (for example,
  `2.7.1.dev0`)
- for a release, follow the steps below
- after publishing, immediately bump `phylib/__init__.py` to the next
  `.dev0` version

## Release

Before merging the release PR, confirm maintainer approval and green CI. Update
`CHANGELOG.md` with all changes since the previous release, replace the pending
release date with the actual release date, and remove the `.dev0` suffix from
`phylib/__init__.py`.

Validate the final release checkout in a dedicated environment:

```shell
uv venv --python 3.12 .venv-release
uv pip install --python .venv-release -r requirements-dev.txt -e . setuptools build twine
.venv-release/bin/python -m flake8 phylib
.venv-release/bin/python -m pytest phylib
```

After the approved PR is merged, create an annotated tag from the merged release
commit (for example, `git tag -a 2.7.1 -m "phylib 2.7.1"`). Build fresh artifacts
from that tagged checkout into an empty output directory:

```shell
.venv-release/bin/python -m build --outdir dist/2.7.1
.venv-release/bin/python -m twine check dist/2.7.1/*
```

Install the wheel in a separate clean environment and verify its version and
imports. Verify the downstream phy template-less waveform regression against
this wheel before publishing. Upload only the artifacts just checked:

```shell
.venv-release/bin/python -m twine upload dist/2.7.1/*
```

Verify installation of the published version from PyPI in another clean
environment. Then bump `master` to `2.7.2.dev0` and add an `Unreleased` changelog
section. Substitute the appropriate version numbers for subsequent releases.
