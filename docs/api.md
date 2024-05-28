# API

## IOs

```{eval-rst}
.. module:: ehrdata.io
.. currentmodule:: ehrdata

.. autosummary::
    :toctree: generated

    io.init_omop
    io.to_dataframe
    io.extract_features
    io.extract_note
    io.from_dataframe
```

## Preprocessing

```{eval-rst}
.. module:: ehrdata.pp
.. currentmodule:: ehrdata

.. autosummary::
    :toctree: generated

    pp.get_feature_statistics
    pp.qc_lab_measurements
    pp.drop_nan

```

## Tools

```{eval-rst}
.. module:: ehrdata.tl
.. currentmodule:: ehrdata

.. autosummary::
    :toctree: generated

    tl.get_concept_name
    tl.aggregate_timeseries_in_bins
```

## Plotting

```{eval-rst}
.. module:: ehrdata.pl
.. currentmodule:: ehrdata

.. autosummary::
    :toctree: generated

    pl.feature_counts
    pl.plot_timeseries
    pl.violin
```
