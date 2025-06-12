# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.6] Not yet released

### Fixed

### Added
- `ehrdata.io.read_csv` Reads a csv file.
- `ehrdata.io.read_h5ad` Reads an h5ad file.
- `ehrdata.io.read_zarr` Reads a zarr file.
- `ehrdata.io.write_h5ad` Writes an h5ad file
- `ehrdata.io.write_zarr` Writes a zarr file.
- `ehrdata.dt.mimic_2` Loads the MIMIC-II dataset.
- `ehrdata.dt.mimic_2_preprocessed` Loads the preprocessed MIMIC-II dataset.
- `ehrdata.dt.diabetes_130_raw` Loads the raw diabetes-130 dataset.
- `ehrdata.dt.diabetes_130_fairlearn` Loads the preprocessed diabetes-130 dataset by fairlearn.
- `ehrdata.tl.infer_feature_types` Infer feature types in an `EHRData` object.
- `ehrdata.tl.feature_type_overview` Overview of inferred feature types.
- `ehrdata.tl.replace_feature_types` Replacing inferred feature types.
- `ehrdata.tl.harmonize_missing_values` Harmonize missing values in an `EHRData` object.
- `ehrdata.tl.from_pandas` Transform a given Pandas `DataFrame` into an `EHRData` object.
- `ehrdata.tl.to_pandas` Transform an `EHRData` object into a Pandas `DataFrame`.


### Modified

## [0.0.5]

### Fixed

- Initialize EHRData with X and layers

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
