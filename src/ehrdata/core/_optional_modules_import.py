def lazy_import_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(
            "The optional module 'torch' is not installed. Please install it using 'pip install ehrdata[torch]'."
        ) from None
