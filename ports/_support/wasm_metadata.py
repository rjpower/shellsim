"""Read the declared shared-library dependencies of a Wasm side module."""

from pathlib import Path


def number(data, offset):
    value = 0
    for shift in range(0, 35, 7):
        if offset >= len(data):
            raise ValueError("truncated Wasm metadata integer")
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


def function_signatures(path: Path):
    """Read core function types for side-module provider ABI validation.

    Only direct function imports have callable signatures. GOT imports are
    address globals and remain governed by the dynamic loader's relocation
    checks. Non-function import descriptors are skipped without resolving them.
    """
    data = path.read_bytes()
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("expected a Wasm core module")
    types, functions, imports, exports = [], [], {}, {}
    offset = 8
    while offset < len(data):
        kind = data[offset]
        size, cursor = number(data, offset + 1)
        end = cursor + size
        offset = end
        if end > len(data):
            raise ValueError("truncated Wasm section")
        section = data[cursor:end]
        if kind not in {1, 2, 3, 7}:
            continue
        count, cursor = number(section, 0)
        for _ in range(count):
            if kind == 1:
                if section[cursor] != 0x60:
                    raise ValueError("unsupported Wasm function type")
                cursor += 1
                params, cursor = number(section, cursor)
                arguments = tuple(section[cursor : cursor + params])
                cursor += params
                results, cursor = number(section, cursor)
                returns = tuple(section[cursor : cursor + results])
                cursor += results
                if len(arguments) != params or len(returns) != results:
                    raise ValueError("truncated Wasm function type")
                types.append((arguments, returns))
            elif kind == 3:
                index, cursor = number(section, cursor)
                if index >= len(types):
                    raise ValueError("Wasm function type index outside type section")
                functions.append(types[index])
            elif kind == 7:
                name, cursor = string(section, cursor)
                category = section[cursor]
                index, cursor = number(section, cursor + 1)
                if category == 0:
                    if index >= len(functions):
                        raise ValueError("Wasm exported function index outside function section")
                    exports[name] = functions[index]
            else:
                module, cursor = string(section, cursor)
                name, cursor = string(section, cursor)
                category = section[cursor]
                cursor += 1
                if category == 0:
                    index, cursor = number(section, cursor)
                    if index >= len(types):
                        raise ValueError("Wasm import type index outside type section")
                    signature = types[index]
                    functions.append(signature)
                    imports[module, name] = signature
                elif category in {1, 2}:
                    if category == 1:
                        cursor += 1
                    flags, cursor = number(section, cursor)
                    _, cursor = number(section, cursor)
                    if flags & 1:
                        _, cursor = number(section, cursor)
                elif category == 3:
                    cursor += 2
                elif category == 4:
                    _, cursor = number(section, cursor + 1)
                else:
                    raise ValueError("unsupported Wasm import type")
            if cursor > len(section):
                raise ValueError("truncated Wasm metadata")
        if cursor != len(section):
            raise ValueError("invalid Wasm metadata section length")
    return imports, exports


def validate_provider_signatures(module: Path, providers: dict[str, Path]) -> list[str]:
    """Check direct calls against the declared shared providers' actual exports."""
    needed = needed_libraries(module)
    if not set(needed) <= providers.keys():
        raise ValueError("Python extension imports an undeclared native provider")
    imports, _ = function_signatures(module)
    exports = {name: function_signatures(providers[name])[1] for name in needed}
    for (namespace, symbol), signature in imports.items():
        candidates = [table[symbol] for table in exports.values() if symbol in table]
        if namespace in exports and symbol not in exports[namespace]:
            raise ValueError(f"native provider omits imported function: {namespace}:{symbol}")
        if any(candidate != signature for candidate in candidates):
            raise ValueError(f"native provider function signature differs: {symbol}")
    return needed
