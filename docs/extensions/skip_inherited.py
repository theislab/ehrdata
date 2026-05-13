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
    """Skip inherited members."""
    if what not in {"method", "attribute", "property"}:
        return None
    if isinstance(obj, property):
        obj = obj.fget
    if name == "__getitem__":
        return False
    if name.startswith("_"):
        return True
    if not hasattr(obj, "__module__"):
        return None
    if not obj.__module__.startswith("ehrdata"):
        return True
    return None


def setup(app: Sphinx) -> None:
    """Setup lamindb for CI."""
    import lamindb as ln

    with suppress(RuntimeError):
        ln.setup.init(storage="/tmp/lamindb")

    app.connect("autodoc-skip-member", skip_member_handler)
