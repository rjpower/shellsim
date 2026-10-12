"""Recognize the original delivery format when admitting prior records."""

from pathlib import Path

from ports._support.store import file_hash

_PACKAGING = "146c3e3957f75fd39cec24cba100a8a31ada8c39c4baa5a368b74b94db9ffd77"


def compatible_packaging() -> bool:
    return file_hash(Path(__file__).with_name("delivery_products.py")) == _PACKAGING
