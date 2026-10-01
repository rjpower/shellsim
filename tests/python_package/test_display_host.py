"""Exercise browser key parsing without opening a host socket."""

from __future__ import annotations

import pytest

from shellsim.display_host import _key_event


def test_key_event_decodes_one_bounded_transition() -> None:
    assert _key_event((27).to_bytes(4, "little") + (1).to_bytes(4, "little")) == (27, True)


@pytest.mark.parametrize(
    "body",
    [b"", b"\x00" * 8, (0x10000).to_bytes(4, "little") + b"\x00" * 4, b"\x01\x00\x00\x00\x02\x00\x00\x00"],
)
def test_key_event_rejects_invalid_input(body: bytes) -> None:
    with pytest.raises(ValueError):
        _key_event(body)
