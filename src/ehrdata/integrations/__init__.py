from importlib import import_module
from importlib.util import find_spec

__all__ = ["torch", "vitessce"]


def __getattr__(name: str):
    if name == "torch":
        if find_spec("torch"):
            torch_module = import_module(f"{__name__}.torch")
            return torch_module
        msg = "torch is not installed. Install with `pip install torch` or `pip install ehrdata[torch]`"
        raise ImportError(msg)

    elif name == "vitessce":
        if find_spec("vitessce"):
            vitessce_module = import_module(f"{__name__}.vitessce")
            return vitessce_module
        msg = "vitessce is not installed. Install with `pip install vitessce` or `pip install ehrdata[vitessce]`"
        raise ImportError(msg)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
