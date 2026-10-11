# Pydantic-core

The locked upstream Rust project uses the canonical PIC stdlib and scheduler
translator with the cached SDK linker. Its native module lives inside the
unchanged Python package tree, including public schema helpers and type stubs.
Generated linker/PyO3 configuration paths are stable across private workspaces,
allowing Cargo to reuse compiled dependencies during source and packaging retries.

The guest probe covers typed dictionaries, integer conversion, JSON parsing and
serialization, and native validation errors. Pydantic and FastAPI exercise the
same provider through their public model and request-validation APIs.
