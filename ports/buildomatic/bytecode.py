"""Generate portable, checked-hash bytecode with the declared guest CPython producer.

The host's Python version never determines shipped bytecode. Source files remain in
place so CPython validates their hashes and repairs code filenames when importing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ports._support.store import relative_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BytecodeCompiler:
    """A declared runtime artifact and its graph identity."""

    bundle: Path
    key: str


def compile_sources(bundle: Path, sources: dict[str, bytes]) -> dict[str, bytes]:
    """Compile without importing package code; syntax-error fixtures retain source only."""
    from shellsim import Context, Limits
    from shellsim.cpython import CPythonRuntime

    if not sources:
        return {}
    environment = Context().Environment(limits=Limits(cpu=10**12, memory=2 * 1024**3, disk=1024**3, output=8 * 1024**2))
    CPythonRuntime(bundle).mount(environment)
    for name, data in sources.items():
        environment.mkdir("/work/bytecode/" + str(Path(relative_path(name)).parent), parents=True)
        environment.write_file("/work/bytecode/" + relative_path(name), data)
    environment.write_file("/work/sources.json", json.dumps(sorted(sources)))
    result = environment.run_python("""
import json
import importlib.util
import os
from importlib._bootstrap_external import _code_to_hash_pyc

with open('/work/sources.json') as stream:
    sources = json.load(stream)
compiled, skipped = [], []
for name in sources:
    path = '/work/bytecode/' + name
    with open(path, 'rb') as stream:
        source = stream.read()
    try:
        code = compile(source, '/' + name, 'exec', dont_inherit=True)
    except SyntaxError:
        skipped.append(name)
    else:
        output = importlib.util.cache_from_source(path)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        # py_compile uses one os.write, but WASI permits short writes. Buffered
        # output completes large files; this isolated build needs no atomic rename.
        bytecode = _code_to_hash_pyc(code, importlib.util.source_hash(source), checked=True)
        with open(output, 'wb') as stream:
            stream.write(bytecode)
        compiled.append(output.removeprefix('/work/bytecode/'))
print(json.dumps({'compiled': compiled, 'skipped': skipped}))
""")
    if result.returncode:
        raise RuntimeError("guest bytecode compilation failed: " + result.stderr.decode(errors="replace"))
    output = json.loads(result.stdout)
    for name in output["skipped"]:
        logger.warning("Keeping syntax-error source without bytecode: %s", name)
    return {name: environment.read_file("/work/bytecode/" + name) for name in output["compiled"]}
