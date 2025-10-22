"""PyTorch integrations for ehrdata."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._dataset import OMOPEHRDataset

__all__ = ["OMOPEHRDataset"]


def __getattr__(name: str):
    if name == "OMOPEHRDataset":
        from ._dataset import OMOPEHRDataset

        return OMOPEHRDataset

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
