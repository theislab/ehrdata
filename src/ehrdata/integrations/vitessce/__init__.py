"""Vitessce integrations for ehrdata."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._config import gen_config, gen_default_config

__all__ = ["gen_config", "gen_default_config"]


def __getattr__(name: str):
    if name == "gen_config":
        from ._config import gen_config

        return gen_config

    elif name == "gen_default_config":
        from ._config import gen_default_config

        return gen_default_config

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
