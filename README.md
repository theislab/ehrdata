# ehrdata

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/ehrdata/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/ehrdata

<p align="center">
<img src="https://raw.githubusercontent.com/theislab/ehrdata/main/docs/_static/tutorial_images/ehrdata_logo.png" width="320" alt="EHRData overview: an X data array with obs, var, and tem annotations, alongside layers, obsm, varm, obsp, varp, and uns.">
</p>

`EHRData` is a data framework that comprises a FAIR storage format and a collection of Python libraries for performant access, alignment, and processing of uni- and multi-modal electronic health record datasets.
This repository contains the core `ehrdata` library, which has the `EHRData` class at its heart.
See the [ehrapy][] package for an analysis package that uses ehrdata to enable the analysis of electronic health record datasetes.

## Getting started

`EHRData` extends [AnnData][] to represent data of **n** observations × **d** variables × **t** time points — a natural fit for the time-resolved measurements found in electronic health records.

```python
import ehrdata as ed

# Load the PhysioNet 2019 sepsis-prediction challenge dataset
# (downloaded and cached on first use)
edata = ed.dt.physionet2019()
edata
```

```
EHRData object with n_obs × n_vars × n_t = 40336 × 35 × 48
    obs: 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'training_Set'
    var: 'Parameter'
    tem: '0', '1', '2', ..., '45', '46', '47'
    shape of .X: (40336, 35, 48)
```

The 35 clinical parameters are stored over 48 hourly time steps in a single three-dimensional array, with patient- and variable-level metadata aligned alongside.
You can slice across all three axes at once — for example, the 48-hour trajectory of the sepsis label for one patient:

```python
edata[edata.obs.index == "p020378", edata.var_names == "SepsisLabel"].X
```

For more, please refer to the [documentation][], in particular the [API documentation][].

## Disclaimer

`ehrdata` is under heavy construction, and its API not stable.
If you find it potentially interesting for your work, reach out to us via the [scverse zulip platform](https://scverse.zulipchat.com/)!
We can help you using it and will be able to stabilize things you need.

If you have inputs on features, please do not hesitate to open an issue on our [issue tracker][]!

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install ehrdata:

1) Install the latest release of `ehrdata` from [PyPI][]:

```bash
pip install ehrdata
```

2. Install the latest development version:

```bash
pip install git+https://github.com/theislab/ehrdata.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org
[anndata]: https://anndata.readthedocs.io/en/stable
[ehrapy]: https://ehrapy.readthedocs.io/en/stable
[issue tracker]: https://github.com/theislab/ehrdata/issues
[tests]: https://github.com/theislab/ehrdata/actions/workflows/test.yml
[documentation]: https://ehrdata.readthedocs.io
[changelog]: https://ehrdata.readthedocs.io/en/latest/changelog.html
[api documentation]: https://ehrdata.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/ehrdata
