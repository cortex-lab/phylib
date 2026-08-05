# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Code of Conduct for project participation and incident reporting.

### Fixed
- Development checkouts now report a development version with a Git commit suffix.
- Template datasets without waveform templates can be reloaded after cluster assignments change,
  and stored spike-waveform subsets provide their waveform sample count.
- TSV, JSON, text and `params.py` files are written atomically, so a crash during a save no longer
  truncates the file it was replacing, for instance `cluster_group.tsv`.

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
