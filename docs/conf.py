from __future__ import annotations

# -- Path setup --------------------------------------------------------------
import sys
import warnings
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.registry import intersphinx


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
warnings.filterwarnings("ignore", message="Extra installs are necessary to use", category=UserWarning)


# -- Project information -----------------------------------------------------

info = metadata("ehrdata")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = release = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
needs_sphinx = "4.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "scverse_doc",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autodoc_type_aliases = {
    "Index": "Index",
    "Index1D": "Index1D",
    "XDataType": "XDataType",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"

intersphinx_mapping = intersphinx("anndata", "scanpy", core=False) | {
    "dask": ("https://docs.dask.org/en/stable/", None),
    "fsspec": ("https://filesystem-spec.readthedocs.io/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/latest", None),
    "lamin": ("https://docs.lamin.ai", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/main", None),
    "vitessce": ("https://python-docs.vitessce.io", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
    "ehrapy": ("https://ehrapy.readthedocs.io/en/latest", None),
}

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/tutorial_images/ehrdata_logo.png"

# If building the documentation fails because of a missing link that is outside your control,
# you can add an exception to this list:
nitpick_ignore = [
    ("py:class", "pathlib._local.Path"),
    ("py:class", "types.EllipsisType"),
    ("py:data", "types.EllipsisType"),
    # TODO: remove once https://github.com/sphinx-doc/sphinx/pull/13508 is released
    ("py:class", "ehrdata._types.TypeAliasType"),
    ("py:class", "zarr.storage._common.TypeAliasType"),
    # typing.Union fails in tutorials/tutorial_time_series_with_pypots
    ("py:data", "typing.Union"),
    # https://github.com/duckdb/duckdb-web/issues/3806
    ("py:class", "duckdb.duckdb.DuckDBPyConnection"),
    ("py:class", "_duckdb.DuckDBPyConnection"),
    ("py:class", "duckdb.duckdb.DuckDBPyRelation"),
    ("py:class", "_duckdb.DuckDBPyRelation"),
    ("py:class", "duckdb._relation.DuckDBPyRelation"),  # duckdb>=2.0
    # Is documented as a py:attribute instead
    ("py:class", "numpy.int64"),
    ("py:class", "numpy._typing._array_like.GenericAlias"),
    # For now not in public facing API
    ("py:class", "awkward.highlevel.Array"),
    ("py:class", "h5py._hl.dataset.Dataset"),
    ("py:class", "zarr.core.Array"),
    ("py:class", "zarr.core.buffer.core.Buffer"),
    ("py:class", "ehrdata._compat.ZappyArray"),
    ("py:class", "anndata.compat.CupyArray"),
    ("py:class", "anndata.compat.CupySparseMatrix"),
    ("py:class", "sparse.numba_backend._coo.core.COO"),
    ("py:class", "sparse._coo.core.COO"),
    ("py:class", "sparse.COO"),
    ("py:class", "pandas.core.arrays.base.ExtensionArray"),
    ("py:class", "ehrdata.core.ehrdata.AlignedMappingProperty3D"),
]

# Redirect broken parameter annotation classes
xref_overrides = {
    "zarr.storage._common.StorePath": "zarr.storage.StorePath",
    "zarr.core.group.StoreLike": "zarr.storage.StoreLike",
    "zarr.core.group.Group": "zarr.Group",
    "lnschema_core.models.Artifact": "lamindb.Artifact",
    "pandas.core.series.Series": "pandas.Series",
    "pyarrow.lib.Table": "pyarrow.Table",
}
