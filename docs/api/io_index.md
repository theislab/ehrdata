# Reading and writing

```{eval-rst}
.. module:: ehrdata
    :no-index:
```

## General I/O

```{eval-rst}
.. autosummary::
    :toctree: io
    :nosignatures:

    io.read_csv
    io.read_h5ad
    io.read_zarr
    io.write_h5ad
    io.write_zarr
    io.from_pandas
    io.to_pandas

```

## OMOP CDM

```{eval-rst}
.. autosummary::
    :toctree: io
    :nosignatures:

    io.omop.setup_connection
    io.omop.setup_obs
    io.omop.setup_variables
    io.omop.setup_interval_variables

```
