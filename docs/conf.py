from __future__ import annotations

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

info = metadata("ehrdata")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "scanpydoc.elegant_typehints",
    "scanpydoc.definition_list_typed_field",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/latest", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
    "vitessce": ("https://python-docs.vitessce.io", None),
    "lamin": ("https://docs.lamin.ai", None),
    "sparse": ("https://sparse.pydata.org/en/stable", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

html_theme = "scanpydoc"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = project_name
html_logo = "_static/tutorial_images/logo.png"
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "theislab",
    "github_repo": project_name,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

pygments_style = "default"

# If building the documentation fails because of a missing link that is outside your control,
# you can add an exception to this list:
nitpick_ignore = [
    ("py:class", "pathlib._local.Path"),
    ("py:class", "types.EllipsisType"),
    # https://github.com/duckdb/duckdb-web/issues/3806
    ("py:class", "duckdb.duckdb.DuckDBPyConnection"),
    # Is documented as a py:attribute instead
    ("py:class", "numpy.int64"),
    # For now not in public facing API
    ("py:class", "awkward.highlevel.Array"),
    ("py:class", "h5py._hl.dataset.Dataset"),
    ("py:class", "zarr.core.Array"),
    ("py:class", "ehrdata._compat.ZappyArray"),
    ("py:class", "dask.array.core.Array"),
    ("py:class", "anndata.compat.CupyArray"),
    ("py:class", "anndata.compat.CupySparseMatrix"),
    ("py:class", "sparse.numba_backend._coo.core.COO"),
    ("py:class", "sparse._coo.core.COO"),
]

# Redirect broken parameter annotation classes
qualname_overrides = {
    "zarr._storage.store.Store": "zarr.storage.MemoryStore",
    "zarr.core.group.Group": "zarr.group.Group",
    "lnschema_core.models.Artifact": "lamindb.Artifact",
}
