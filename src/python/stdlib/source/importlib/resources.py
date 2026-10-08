"""Package data paths confined to shellsim's virtual filesystem."""

from pathlib import Path


def files(anchor):
    """Return the virtual directory containing a VFS-backed package or module."""
    if isinstance(anchor, str):
        from importlib import import_module

        anchor = import_module(anchor)
    path = getattr(anchor, "__file__", None)
    if path is None or path.startswith("<frozen "):
        raise TypeError("resources require a filesystem-backed module")
    return Path(path).parent
