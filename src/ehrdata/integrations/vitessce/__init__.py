"""Vitessce integrations for ehrdata."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._config import gen_config

__all__ = ["gen_config"]


def __getattr__(name: str):
    if name == "gen_config":
        from ._config import gen_config

        return gen_config

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
