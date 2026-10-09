"""Read the declared shared-library dependencies of a Wasm side module."""

from pathlib import Path


def number(data, offset):
    value = 0
    for shift in range(0, 35, 7):
        byte = data[offset]
        offset += 1
        if shift == 28 and byte > 15:
            raise ValueError("overflowing Wasm metadata integer")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("overflowing Wasm metadata integer")


def string(data, offset):
    size, offset = number(data, offset)
    end = offset + size
    if end > len(data):
        raise ValueError("truncated Wasm metadata string")
    return data[offset:end].decode(), end


def needed_libraries(path: Path):
    """Read LLVM's declared dependency list from the verified target artifact."""
    data = path.read_bytes()
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("expected a Wasm core module")
    offset = 8
    needed = []
    while offset < len(data):
        kind = data[offset]
        size, payload = number(data, offset + 1)
        offset = payload + size
        if offset > len(data):
            raise ValueError("truncated Wasm section")
        if kind != 0:
            continue
        name, payload = string(data, payload)
        if name != "dylink.0":
            continue
        section = data[payload:offset]
        cursor = 0
        while cursor < len(section):
            kind = section[cursor]
            size, payload = number(section, cursor + 1)
            cursor = payload + size
            if cursor > len(section):
                raise ValueError("truncated dylink subsection")
            if kind != 2:
                continue
            count, payload = number(section, payload)
            for _ in range(count):
                name, payload = string(section, payload)
                needed.append(name)
            if payload != cursor:
                raise ValueError("invalid dylink dependency length")
    return needed
