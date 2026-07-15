# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed
 - {func}`~ehrdata.integrations.vitessce.gen_default_config` no longer fails with `X must be 2-dimensional` when the time series lives in a 3D `.X` (e.g. from {func}`~ehrdata.dt.physionet2019`); the selected source is reduced to the chosen `timestep` before writing, and `layer` now defaults to `None` (use `.X`). ([#271](https://github.com/theislab/ehrdata/pull/271)) @eroell
 - {func}`~ehrdata.infer_feature_types` binary detection contained a latent bug: the integrality guard used `np.all(<generator>)`, which is always truthy and so never actually ran. The check is now the equivalent, correct `set(col.unique()) == {0, 1}`. No user-visible behaviour changes, as the disabled guard was redundant with the `{0, 1}` set check. ([#268](https://github.com/theislab/ehrdata/pull/268)) @Zethson
 - {func}`~ehrdata.infer_feature_types` no longer emits a warning with a blank feature name (`Feature  was detected as categorical features stored numerically.`) when no feature is uncertain. The warning is now only shown when at least one feature was detected as categorical stored numerically, and lists the affected feature names. ([#267](https://github.com/theislab/ehrdata/pull/267)) @Zethson
 - Slicing a 3D {attr}`~ehrdata.EHRData.X` along the time axis now also slices `.X`. Previously, subsetting the third axis updated `.shape` and `.tem` but returned the parent's full-length `.X`, so `edata[:, :, idx].X.shape` disagreed with `edata[:, :, idx].shape`. `.X` now applies the time-axis index exactly like a 3D layer. ([#259](https://github.com/theislab/ehrdata/issues/259)) @eroell

### Documentation
 - The {class}`~ehrdata.EHRData` attribute reference now describes `.obsm`, `.varm`, `.obsp` and `.varp` in anndata's terms instead of showing a generic placeholder. ([#263](https://github.com/theislab/ehrdata/pull/263)) @eroell
 - Updated logo with 3D `.X`. Tutorials updated to use `.X` canonically. Docstring examples no longer pass `layer="tem_data"`: the time series is kept in the default 3D `.X`. ([#273](https://github.com/theislab/ehrdata/pull/273)) @eroell

## [0.3.0]

### Added
 - {func}`~ehrdata.io.read_h5ed` and {func}`~ehrdata.io.write_h5ed` are the new primary HDF5 I/O functions: `.h5ed` is now ehrdata's on-disk format, marking it distinct from anndata's `.h5ad`. `read_h5ad` and `write_h5ad` remain as deprecated aliases. ([#252](https://github.com/theislab/ehrdata/pull/252)) @eroell
 - {func}`~ehrdata.io.write_h5ed` and {func}`~ehrdata.io.write_zarr` now write the ehrdata on-disk encoding 0.2.0. Because AnnData only guarantees 2D but not 3D arrays in `X`/`layers` (see [scverse/anndata#2430](https://github.com/scverse/anndata/issues/2430)), 3D arrays are relocated into `.obsm` (under reserved `_ed_ondisk_X` / `_ed_ondisk_layers_<name>` keys) and dropped from `X`/`layers` on write, then restored on read. Backwards compatibility for reading is maintained. The recommended file extensions for `hdf5` are now `.h5ed`. ([#252](https://github.com/theislab/ehrdata/pull/252)) @eroell

### Fixed
 - {func}`~ehrdata.infer_feature_types` can handle `EHRData` objects with `X` as `None`. ([#246](https://github.com/theislab/ehrdata/pull/246)) @sueoglu
 - {func}`~ehrdata.dt.physionet2019` no longer raises a shape mismatch on the full dataset: persons whose dynamic measurements all fall outside the observation window are now padded with missing values instead of being dropped from the time series tensor. ([#251](https://github.com/theislab/ehrdata/issues/251)) @eroell
 - {func}`~ehrdata.harmonize_missing_values` no longer logs a warning for each numeric layer when it has nothing to harmonize; reading a fully numeric dataset is now quiet.([#252](https://github.com/theislab/ehrdata/pull/252))  @eroell

### Maintenance
 - ehrdata is now compatible with anndata 0.13 (unified `X`/`layers` storage, `IndexManager` view indices, index type aliases moved to `anndata.typing`, and the stricter 2D-only `X`/`layers` on-disk rule) while remaining compatible with anndata `<0.13`. CI gains an integration-free `core-anndata-min` lane (pinned to `anndata<0.13`) and a pre-release lane to catch upstream breakage early. ([#252](https://github.com/theislab/ehrdata/pull/252)) @eroell
 - `DaskArray` import fixed to follow anndata 0.12.16, and `dask` added to the intersphinx mapping. ([#248](https://github.com/theislab/ehrdata/pull/248)) @sueoglu
 - CI now caches downloaded datasets used by `ehrdata.dt` to reduce flaky upstream hosts (e.g. physionet.org) breaking the test and notebook workflows. ([#250](https://github.com/theislab/ehrdata/pull/250)) @eroell
 - Dataset downloads now use [pooch](https://www.fatiando.org/pooch/) instead of a custom `requests`-based downloader, aligning with the scverse ecosystem and providing caching out of the box. ([#251](https://github.com/theislab/ehrdata/issues/251)) @eroell
 - The `tqdm` dependency has been removed. ([#254](https://github.com/theislab/ehrdata/pull/254)) @eroell

### Documentation
 - Improve {class}`~ehrdata.EHRData` API documentation. ([#258](https://github.com/theislab/ehrdata/issues/258)) @eroell

## [0.2.1]

### Fixed
 - Compatibility with `anndata>=0.12.13` ([#240](https://github.com/theislab/ehrdata/pull/240)) @eroell

## [0.2.0]

### Fixed
 - Assigning `.X` to a view of an X-less {class}`~ehrdata.EHRData` (e.g. one created with `layers=` only) no longer raises `TypeError: 'NoneType' object does not support item assignment`. The view is now materialised before the assignment, consistent with how AnnData handles other field modifications on views. ([#233](https://github.com/theislab/ehrdata/pull/233)) @eroell

### Modified
 - {func}`~ehrdata.infer_feature_types` considers integers from 0, ..., n as numeric. It further provides a new argument `binary_as`, to steer if columns 0/1 should be considered numeric or categorical. ([#231](https://github.com/theislab/ehrdata/pull/231)) @eroell

## [0.1.2]

### Added
 - {func}`~ehrdata.io.from_pandas` with `format='long'` provides a new keyword argument `fill_time_gaps` that fills missing timegaps in the common case of integer time steps from 0 to n_timesteps ([#229](https://github.com/theislab/ehrdata/pull/229)) @eroell

### Modified
 - {func}`~ehrdata.dt.mimic_2` column `censor_flg` switched to lifeline's convention with 1=event, 0=censored, before this dataset loader function had them vice versa since the dataset provides them as such originally. ([#227](https://github.com/theislab/ehrdata/pull/227)) @sueoglu

### Fixed
 - {func}`~ehrdata.io.from_pandas` with `format='long'` misordered entries in `.X`/`.layers` with `.obs` if the input df was not sorted for the obs id keys, which is now fixed. ([#228](https://github.com/theislab/ehrdata/pull/228)) @eroell

### Documentation
 - Documentation style polishing ([#223](https://github.com/theislab/ehrdata/pull/223)) @zethson

## [0.1.1]

### Added
 - {func}`~ehrdata.io.omop.setup_connection` can read `.parquet` files. ([#217](https://github.com/theislab/ehrdata/pull/217)) @eroell

### Fixed
 - Sliceing of `EHRData` objects fixed when the backing object is an `AnnData`. ([#218](https://github.com/theislab/ehrdata/pull/218)) @eroell

### Maintenance
 - More concise messages in {func}`~ehrdata.infer_feature_types`. ([#215](https://github.com/theislab/ehrdata/pull/215)) @zethson


## [0.1.0]

### Added
 - {func}`~ehrdata.move_to_obs` and {func}`~ehrdata.move_to_x` are new helpers for conveniently moving variables from central 2D arrays to the `.obs` field, and vice versa. ([#199](https://github.com/theislab/ehrdata/pull/201)) @eroell
  - {func}`~ehrdata.dt.physionet2019` as another out-of-the-box, conveniently available dataset with 40'000 ICU stays from the Physionet 2019 challenge. ([#204](https://github.com/theislab/ehrdata/pull/204)) @eroell
 - `time_precision` parameter (`"date"` or `"datetime"`) to {func}`~ehrdata.io.omop.setup_variables` and {func}`~ehrdata.io.omop.setup_interval_variables` for finer temporal granularity control. ([#210](https://github.com/theislab/ehrdata/pull/210)) @eroell

### Fixed
- `read_h5ad` fixed issues when `backed=True`. ([#199](https://github.com/theislab/ehrdata/pull/199)) @eroell
- `read_h5ad` fixed bug when `.X` is `None` and `harmonize_missing_features` is `True`. ([#206](https://github.com/theislab/ehrdata/pull/206)) @eroell
- {func}`~ehrdata.io.omop.setup_obs` with `observation_table="person_visit_occurrence"` now supports multiple visits per patient, creating one row per visit with unique observation IDs, instead of failing with xarray conversion errors with non-unique indices. ([#210](https://github.com/theislab/ehrdata/pull/210)) @eroell
- OMOP time interval boundaries now use half-open intervals `[start, end)` to prevent duplicate measurements at interval boundaries. ([#210](https://github.com/theislab/ehrdata/pull/210)) @eroell

### Maintenance
- Support Python3.14 ([#194](https://github.com/theislab/ehrdata/pull/194)) @Zethson
- Address `FutureWarning`s across multiple places ([#200](https://github.com/theislab/ehrdata/pull/200)) @eroell
- Enhanced tutorial structure ([#208](https://github.com/theislab/ehrdata/pull/208)) @eroell

### Modified
- Dataset generator function `ed.dt.ehrdata_blobs` now takes `n_cat_var` and `n_categories` arguments to generate categorical (integer encoded) time series data ([#207](https://github.com/theislab/ehrdata/pull/207)) @sueoglu
- If `enrich_var_with_feature_info=True` in {func}`~ehrdata.io.omop.setup_variables` and {func}`~ehrdata.io.omop.setup_interval_variables`, `data_table_concept_ids` not included within the concept table are now mapped from their respective alternate `concept_id` included in the concept_relationship table to retrieve the available feature information. ([#205](https://github.com/theislab/ehrdata/pull/205)) @KilianDahm
- {func}`~ehrdata.io.omop.setup_variables` and {func}`~ehrdata.io.omop.setup_interval_variables` with use of `"person"` now checks `birth_datetime` for meaningful behaviour and error messages. ([#210](https://github.com/theislab/ehrdata/pull/210)) @eroell
- {func}`~ehrdata.integrations.vitessce.gen_default_config` provides convenience to generate a config directly from an `EHRData` object, and should be used instead of the previous `ehrdata.integrations.vitessce.gen_config`. ([#211](https://github.com/theislab/ehrdata/pull/211)) @eroell

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
- `read_h5ad` Reads an h5ad file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- {func}`~ehrdata.io.read_zarr` Reads a zarr file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
- `write_h5ad` Writes an h5ad file ([#136](https://github.com/theislab/ehrdata/pull/136)) @eroell
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
