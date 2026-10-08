"""Package resource reads through shellsim's virtual filesystem."""

from importlib import import_module
from pathlib import Path


def get_data(package, resource):
    """Read a package-relative file as bytes from the simulated filesystem.

    Frozen modules have no resource directory, so they return ``None``. Resource
    names cannot use absolute paths or parent traversal; reads stay in the VFS.
    """
    if not isinstance(package, str) or not isinstance(resource, str):
        raise TypeError("package and resource must be strings")
    module = import_module(package)
    filename = getattr(module, "__file__", None)
    if filename is None or filename.startswith("<frozen "):
        return None
    if resource.startswith("/") or ".." in resource.split("/"):
        raise ValueError("resource path must stay within the package")
    return (Path(filename).parent / resource).read_bytes()
