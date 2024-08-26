from __future__ import annotations

import datetime
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from ehrapy.anndata import df_to_anndata
from matplotlib.axes import Axes

from ehrdata.io._omop import to_dataframe
from ehrdata.tl import get_concept_name
from ehrdata.utils._omop_utils import get_column_types, map_concept_id, read_table


def feature_counts(
    adata: AnnData,
    source: Literal[
        "observation",
        "measurement",
        "procedure_occurrence",
        "specimen",
        "device_exposure",
        "drug_exposure",
        "condition_occurrence",
    ],
    number: int = 20,
    use_dask: bool = None,
) -> pd.DataFrame:
    """Plot feature counts for a given source table and return a dataframe with feature names and counts.

    Args:
        adata: Anndata object.
        source: Source table name.
        number: Number of top features to plot. Defaults to 20.
        use_dask: If True, dask will be used to read the tables. For large tables, it is highly recommended to use dask. If None, it will be set to adata.uns["use_dask"]. Defaults to None.

    Returns
    -------
        Dataframe with feature names and counts.
    """
    path = adata.uns["filepath_dict"][source]
    if isinstance(path, list):
        if not use_dask or use_dask is None:
            use_dask = True
            warnings.warn(f"Multiple files detected for table [{source}]. Using dask to read the table.", stacklevel=2)
    if not use_dask:
        use_dask = adata.uns["use_dask"]

    column_types = get_column_types(adata.uns, table_name=source)
    if source in ["observation", "measurement", "specimen"]:
        id_key = f"{source}_concept_id"
    else:
        id_key = source.split("_")[0] + "_concept_id"
    df_source = read_table(adata.uns, table_name=source, dtype=column_types, usecols=[id_key, "visit_occurrence_id"])
    df_source = df_source[df_source["visit_occurrence_id"].isin(set(adata.obs.index))]
    feature_counts = df_source[id_key].value_counts()
    if use_dask:
        feature_counts = feature_counts.compute()
    feature_counts = feature_counts.to_frame().reset_index(drop=False)[0:number]

    feature_counts[f"{id_key}_1"], feature_counts[f"{id_key}_2"] = map_concept_id(
        adata.uns, concept_id=feature_counts[id_key], verbose=False
    )
    feature_counts["feature_name"] = get_concept_name(adata, concept_id=feature_counts[f"{id_key}_1"])
    if feature_counts[f"{id_key}_1"].equals(feature_counts[f"{id_key}_2"]):
        feature_counts.drop(f"{id_key}_2", axis=1, inplace=True)
        feature_counts.rename(columns={f"{id_key}_1": id_key})
        feature_counts = feature_counts.reindex(columns=["feature_name", id_key, "count"])
    else:
        feature_counts = feature_counts.reindex(columns=["feature_name", f"{id_key}_1", f"{id_key}_2", "count"])
    # sns.color_palette("Paired")
    ax = sns.barplot(feature_counts, x="feature_name", y="count", palette=sns.color_palette("Paired"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return feature_counts


def plot_timeseries(
    adata: AnnData,
    visit_occurrence_id: int,
    key: str | list[str],
    slot: str | None = "obsm",
    value_key: str = "value_as_number",
    time_key: str = "measurement_datetime",
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    from_time: str | datetime.datetime | None = None,
    to_time: str | datetime.datetime | None = None,
    show: bool | None = None,
):
    """Plot timeseries data using data from adata.obsm.

    Args:
        adata: Anndata object.
        visit_occurrence_id: Visit_occurrence_id to plot.
        key: Feature key or list of keys in adata.obsm to plot.
        slot: Slot to use. Defaults to "obsm".
        value_key: Key in awkward array in adata.obsm to be used as value. Defaults to "value_as_number".
        time_key: Key in awkward array in adata.obsm to be used as time. Defaults to "measurement_datetime".
        from_time: Start time for the plot. Defaults to None.
        to_time: End time for the plot. Defaults to None.
        x_label: x label name. Defaults to None.
        y_label: y label name. Defaults to None.
        title: Title of the plot. Defaults to None.
        show: Show the plot, do not return axis.
    """
    if isinstance(key, str):
        key_list = [key]
    else:
        key_list = key

    # Initialize min_x and max_x
    min_x = None
    max_x = None

    if slot == "obsm":
        _, ax = plt.subplots(figsize=(20, 6))
        # Scatter plot
        for key in key_list:
            df = to_dataframe(adata, features=key, visit_occurrence_id=visit_occurrence_id)
            if from_time:
                if isinstance(from_time, str):
                    from_time = pd.to_datetime(from_time)
                df = df[df[time_key] >= from_time]
            if to_time:
                if isinstance(to_time, str):
                    to_time = pd.to_datetime(to_time)
                df = df[df[time_key] <= to_time]
            df.sort_values(by=time_key, inplace=True)
            x = df[time_key]
            y = df[value_key]

            # Check if x is empty
            if not x.empty:
                ax.scatter(x=x, y=y, label=key)
                ax.legend(bbox_to_anchor=(0.5, -0.1), ncol=4)

                ax.plot(x, y)

                if min_x is None or min_x > x.min():
                    min_x = x.min()
                if max_x is None or max_x < x.max():
                    max_x = x.max()

            else:
                # Skip this iteration if x is empty
                continue

        if min_x is not None and max_x is not None:
            # Adapt this to input data
            # TODO step
            # plt.xticks(np.arange(min_x, max_x, step=1))
            # Adapt this to input data
            plt.xlabel(x_label if x_label else "Datetime")
        plt.ylabel(y_label if y_label else "Value")
        plt.title(title if title else f"Timeseries plot for visit_occurrence_id: {visit_occurrence_id}")
        plt.tight_layout()
        if not show:
            return ax
        else:
            plt.show()


def violin(
    adata: AnnData,
    obsm_key: str = None,
    keys: str | Sequence[str] = None,
    groupby: str | None = None,
    log: bool | None = False,
    use_raw: bool | None = None,
    stripplot: bool = True,
    jitter: float | bool = True,
    size: int = 1,
    layer: str | None = None,
    scale: Literal["area", "count", "width"] = "width",
    order: Sequence[str] | None = None,
    multi_panel: bool | None = None,
    xlabel: str = "",
    ylabel: str | Sequence[str] = None,
    rotation: float | None = None,
    show: bool | None = None,
    save: bool | str | None = None,
    ax: Axes | None = None,
    **kwds,
):  # pragma: no cover
    """Violin plot.

    Args:
        adata: AnnData object containing all observations.
        obsm_key: Feature key or list of keys in adata.obsm to plot.
        keys: Keys for accessing variables of `.var_names` or fields of `.obs`.
        groupby: The key of the observation grouping to consider.
        log: Plot on logarithmic axis.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        stripplot: Add a stripplot on top of the violin plot.
        jitter: Add jitter to the stripplot (only when stripplot is True).
        size: Size of the jitter points.
        layer: Name of the AnnData object layer that wants to be plotted.
        scale: The method used to scale the width of each violin.
        order: Order in which to show the categories.
        multi_panel: Display keys in multiple panels also when `groupby` is not None.
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        rotation: Rotation of xtick labels.
        show: Show the plot, do not return axis.
        save: Save the plot to file.
        ax: Matplotlib Axes object to use. If not passed, uses the current Axes instance.
        **kwds: Additional keyword arguments passed to seaborn.violinplot.

    Returns
    -------
        A matplotlib Axes object if ax is None, else None.
    """
    if obsm_key:
        df = to_dataframe(adata, features=obsm_key)
        df = df[["visit_occurrence_id", "value_as_number"]]
        df = df.rename(columns={"value_as_number": obsm_key})

        if groupby:
            df = df.set_index("visit_occurrence_id").join(adata.obs[groupby].to_frame()).reset_index(drop=False)
            adata = df_to_anndata(df, columns_obs_only=["visit_occurrence_id", groupby])
        else:
            adata = df_to_anndata(df, columns_obs_only=["visit_occurrence_id"])
        keys = obsm_key

    violin_partial = partial(
        sc.pl.violin,
        keys=keys,
        log=log,
        use_raw=use_raw,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        layer=layer,
        scale=scale,
        order=order,
        multi_panel=multi_panel,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=rotation,
        show=show,
        save=save,
        ax=ax,
        **kwds,
    )

    return violin_partial(adata=adata, groupby=groupby)
