"""Classify compiler options through bounded LLVM GNU response files.

Expansion is for wrapper policy only. The compiler receives its original argv
and performs its own response expansion. Nested paths follow Clang's default
working-directory semantics, rather than the containing response directory.
"""

from pathlib import Path

_MAX_RESPONSE_BYTES = 4 * 1024**2
_MAX_RESPONSE_FILES = 256
_MAX_RESPONSE_DEPTH = 16
_MAX_RESPONSE_ARGUMENTS = 65536


def _response_tokens(text: str) -> list[str]:
    """Match LLVM TokenizeGNUCommandLine quotes and next-character escapes."""
    tokens, token = [], []
    position, started = 0, False
    while position < len(text):
        character = text[position]
        if character in " \t\r\n":
            if started:
                tokens.append("".join(token))
                token, started = [], False
            position += 1
            continue
        started = True
        if character == "\\" and position + 1 < len(text):
            position += 1
            token.append(text[position])
        elif character in "\"'":
            quote = character
            position += 1
            while position < len(text) and text[position] != quote:
                if text[position] == "\\" and position + 1 < len(text):
                    position += 1
                token.append(text[position])
                position += 1
        else:
            token.append(character)
        position += 1
    if started:
        tokens.append("".join(token))
    return tokens


def response_arguments(arguments: list[str], directory: Path) -> list[str]:
    """Read option tokens without changing the original compiler command.

    Fail before invoking the compiler if a response cannot be read, recursively
    includes itself, or exceeds bounds. Repeated noncyclic inclusions are legal
    and count toward the aggregate work limits.
    """
    result, pending = [], [(arguments, frozenset())]
    total_bytes, files = 0, 0
    while pending:
        values, active = pending.pop()
        for index, argument in enumerate(values):
            if not argument.startswith("@"):
                if len(result) >= _MAX_RESPONSE_ARGUMENTS:
                    raise ValueError("compiler response arguments exceed bound")
                result.append(argument)
                continue
            path = (directory / argument[1:]).resolve()
            if path in active:
                raise ValueError(f"cyclic compiler response file: {path}")
            if len(active) >= _MAX_RESPONSE_DEPTH or files >= _MAX_RESPONSE_FILES:
                raise ValueError("compiler response nesting or file count exceeds bound")
            files += 1
            try:
                with path.open("rb") as stream:
                    data = stream.read(_MAX_RESPONSE_BYTES - total_bytes + 1)
            except OSError as error:
                raise ValueError(f"cannot read compiler response file: {path}") from error
            total_bytes += len(data)
            if total_bytes > _MAX_RESPONSE_BYTES:
                raise ValueError("compiler response bytes exceed bound")
            try:
                text = (
                    data.decode("utf-16")
                    if data.startswith((b"\xff\xfe", b"\xfe\xff"))
                    else data.decode("utf-8-sig", errors="surrogateescape")
                )
            except UnicodeError as error:
                raise ValueError(f"invalid compiler response encoding: {path}") from error
            tokens = _response_tokens(text)
            if len(tokens) > _MAX_RESPONSE_ARGUMENTS:
                raise ValueError("compiler response arguments exceed bound")
            pending.append((values[index + 1 :], active))
            pending.append((tokens, active | {path}))
            break
    return result
