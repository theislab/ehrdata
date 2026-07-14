from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def skip_member_handler(
    app: Sphinx,
    what: Literal["module", "class", "exception", "function", "method", "attribute", "property"],
    name: str,
    obj: object,
    skip: bool,  # noqa: FBT001
    options,
) -> bool | None:
    """Keep ehrdata members and inherited AnnData attributes; drop inherited methods."""
    if what not in {"method", "attribute", "property"}:
        return None
    if isinstance(obj, property):
        obj = obj.fget
    if name.startswith("_"):
        return True
    if name in {"T", "raw"}:
        return True
    module = getattr(obj, "__module__", None)
    if module is None:
        return None
    if module.startswith("ehrdata"):
        return None
    if what == "method":
        return True
    if module.startswith("anndata"):
        return None
    return True


def setup(app: Sphinx) -> None:
    """Setup lamindb for CI."""
    import lamindb as ln

    with suppress(RuntimeError):
        ln.setup.init(storage="/tmp/lamindb")

    app.connect("autodoc-skip-member", skip_member_handler)
