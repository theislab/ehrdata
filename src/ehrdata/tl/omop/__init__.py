"""Tools for working with OMOP data."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._dataset import EHRDataset

__all__ = ["EHRDataset"]


def __getattr__(name: str):
    if name == "EHRDataset":
        from ._dataset import EHRDataset

        return EHRDataset
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
