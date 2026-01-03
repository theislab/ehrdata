# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.11]

### Fixed
- {func}`~ehrdata.io.read_h5ad` fixed issues when `backed=True`. ([#199](https://github.com/theislab/ehrdata/pull/199)) @eroell

### Maintenance
- Address `FutureWarning`s across multiple places ([#200](https://github.com/theislab/ehrdata/pull/200)) @eroell

## [0.0.10]

{class}`~ehrdata.EHRData` drops the `.R` field, and now supports 3D data storage in any slot of `.layers`. See the {doc}`tutorials/getting_started` tutorial for an introduction to this behaviour. In the future, `.X` will be enabled soon for 3D data storage as well.

### Maintenance
- Enhanced {doc}`tutorials/getting_started` ([#184](https://github.com/theislab/ehrdata/pull/184)) @eroell
- Move from zarr<3 to zarr>=3 ([#185](https://github.com/theislab/ehrdata/pull/185)) @eroell

### Fixed

### Modified
- `EHRData` drops the `.R` field in favor of using `.layers` for any 3D data arrays ([#184](https://github.com/theislab/ehrdata/pull/184)) @eroell
- `EHRData`'s shape property will always return a 3 dimensional shape. If an `EHRData` object has flat arrays only, the third dimension will be 1. ([#184](https://github.com/theislab/ehrdata/pull/184)) @eroell
- The following functions now take a `layer` argument: {func}`~ehrdata.io.read_csv`, {func}`~ehrdata.io.from_pandas`, {func}`~ehrdata.io.to_pandas`, {func}`~ehrdata.io.omop.setup_variables`, {func}`~ehrdata.io.omop.setup_interval_variables`, {func}`~ehrdata.dt.ehrdata_blobs`, {func}`~ehrdata.dt.physionet2012`. If it is let to its default, `None`, the `.X` field of `EHRData` is used. Since `.X` is 2D in this release, in cases with 3D data, the `layer` argument needs to be used. ([#184](https://github.com/theislab/ehrdata/pull/184)) @eroell
- {func}`~ehrdata.io.write_zarr` now writes an `EHRData` specific store encoding, with `anndata` as a substore. This change allows to use `AnnData` with its change to consolidated Zarr metadata, and better isolates `AnnData`'s io. ([#185](https://github.com/theislab/ehrdata/pull/185)) @eroell
- {func}`~ehrdata.io.read_zarr` is adapted to read the new store encoding, and can also deal with `AnnData` stores. ([#185](https://github.com/theislab/ehrdata/pull/185)) @eroell


## [0.0.9]

### Maintenance
- Use custom logger & remove pydata sparse ([#176](https://github.com/theislab/ehrdata/pull/176)) @Zethson
- Replace figshare with scverse S3 ([#177](https://github.com/theislab/ehrdata/pull/177)) @Zethson
- Update template to v0.6.0 ([#166](https://github.com/theislab/ehrdata/pull/166)) @Zethson

### Fixed
- Fix order of `var` created in `ed.io.omop.setup_variables` and `ed.io.omop.setup_interval_variables` ([#179](https://github.com/theislab/ehrdata/pull/179)) @eroell

### Modified
- Rename `ed.pl.vitessce.gen_config` to `ed.integrations.vitessce.gen_config` ([#181](https://github.com/theislab/ehrdata/pull/181)) @eroell
- Rename `ed.tl.omop.EHRDataset` to `ed.integrations.torch.OMOPEHRDataset` ([#181](https://github.com/theislab/ehrdata/pull/181)) @eroell


## [0.0.8]

### Fixed
- Update duckdb imports for future ([#157](https://github.com/theislab/ehrdata/pull/157)) @eroell

### Maintenance
- Private subset method for `EHRData` ([#160](https://github.com/theislab/ehrdata/pull/160)) @eroell
- Remove `omop` package dependency ([#160](https://github.com/theislab/ehrdata/pull/160)) @eroell

## [0.0.7]

### Fixed
- Fix tests and Getting Started Notebook ([#155](https://github.com/theislab/ehrdata/pull/155)) @eroell

### Maintenance
- Update duckdb imports for future ([#155](https://github.com/theislab/ehrdata/pull/155)) @eroell

## [0.0.6]

### Fixed
- Cleaned up and updated tutorial notebooks ([#140](https://github.com/theislab/ehrdata/pull/140)) @agerardy

### Added
- {func}`~ehrdata.io.read_csv` Reads a csv file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.read_h5ad` Reads an h5ad file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.read_zarr` Reads a zarr file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.write_h5ad` Writes an h5ad file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.write_zarr` Writes a zarr file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.from_pandas` Transform a given {class}`~pandas.DataFrame` into an {class}`~ehrdata.EHRData` object ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.to_pandas` Transform an {class}`~ehrdata.EHRData` object into a {class}`~pandas.DataFrame` ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.dt.mimic_2` Loads the MIMIC-II dataset ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.dt.mimic_2_preprocessed` Loads the preprocessed MIMIC-II dataset ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.dt.diabetes_130_raw` Loads the raw diabetes-130 dataset ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.dt.diabetes_130_fairlearn` Loads the preprocessed diabetes-130 dataset by fairlearn ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.infer_feature_types` Infer feature types in an {class}`~ehrdata.EHRData` object ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.feature_type_overview` Overview of inferred feature types ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.replace_feature_types` Replacing inferred feature types ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.harmonize_missing_values` Harmonize missing values in an {class}`~ehrdata.EHRData` object ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell

## [0.0.5]

### Fixed

- Initialize EHRData with X and layers ([#132](https://github.com/theislab/ehrdata/pull/132)) @eroell

### Added

### Modified

- Rename `.t` attribute to `.tem`

## [0.0.4]

### Fixed

- Zarr version to less than 3

## [0.0.3]

### Fixed

- Added missing zarr dependency

## [0.0.2]

### Added

- Expanded documentation
- Improved OMOP Extraction
- Support for [COO sparse matrices](https://github.com/pydata/sparse) for R
- A `ed.dt.ehrdata_blobs` test data generator function
- Replace -1 encoded missing values with nans in physionet2012 challenge data

### Breaking changes

- Renamed `r` to `R`

## [0.0.1] - 2024-11-04

### Added

- Initial release

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions

### Fixed

- tutorial notebooks updated to align with breaking changes
