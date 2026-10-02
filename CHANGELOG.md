# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.1] - Unreleased

### Added
- Added a Code of Conduct for project participation and incident reporting.

### Changed
- Optimized spike selection and stored sparse-waveform extraction for large datasets,
  including bounded display sampling and batched memory-mapped reads.
- Unknown raw-data file extensions now emit a warning instead of preventing the dataset
  from loading.

### Fixed
- Prevented geometry range searches from hanging with `float32` input.
- Development checkouts now report a development version with a Git commit suffix.
- Fixed reloading curated template-less datasets after cluster assignments change.
- Derived the waveform sample count from stored spike-waveform subsets when templates are
  unavailable.
- Treated a blank `dat_path` in `params.py` as no raw-data file instead of the dataset
  directory (#57).
- Made cluster-assignment, TSV, JSON, text, and `params.py` writes atomic to prevent
  truncation if saving is interrupted.
- Corrected wheel metadata to advertise Python 3 only.
- Included the changelog and test dependency list in source distributions.

### Development
- Enforced flake8 checks in CI and fixed existing lint violations.
- Made copied test datasets writable when the source cache is read-only, supporting
  sandboxed distribution builds (#66).

## [2.7.0] 2025-12-10

### Added
- `phylib.alf.io` reverse Alf to Phy conversion to get AU amplitudes ala kilosort from true units amplitudes.

## [2.6.4] 2025-12-07

### Fixed
- #53 channel_ids is often provided as a list, cast numpy arrayW

## [2.6.3]  2025-11-03

### Added

- Add `iblsorter_parameters.yaml` as a dataset to be copied during ALF export. 

## [2.6.2]  2025-10-21

### Added

- Contribution instructions (pypi, changelog, tests...)

### Fixed

- Compatibility with numpy 2
