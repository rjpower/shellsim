"""Admit prior records only for the equivalent driver and packaging revision.

These fixed source boundaries prevent a missing new key from re-adopting old
bytes after a producer changes. SDK receipts have their own source policies.
"""

from pathlib import Path

from ports._support.graph import Port
from ports._support.store import file_hash

_DRIVER = "f1b711cd6605e8f21ca3b413e4ece5e34ec9f7e2017eff0699b8f0c8bcabba60"
_BINDING = "015e61fb5c52f214c0e00edd6cfcb6cf9b8eb77b22ffaba920c532965001963c"
_PACKAGING = "146c3e3957f75fd39cec24cba100a8a31ada8c39c4baa5a368b74b94db9ffd77"


def compatible_driver(port: Port) -> bool:
    root = Path(__file__).resolve().parents[1]
    return file_hash(root / "_support/node_build.py") == _DRIVER and (
        port.recipe["build_system"] in {"pure-wheel", "host-wheel", "source-tree"}
        or file_hash(root / "_support/sdk_products.py") == _BINDING
    )


def compatible_packaging() -> bool:
    return file_hash(Path(__file__).with_name("delivery_products.py")) == _PACKAGING
