# rpds-py

The upstream locked Cargo project builds with ordinary Rust and the canonical
PIC standard-library dependency. The SDK linker imports the process's libc
and Python symbols, uses deferred shared-memory initialization, and exports
only required entry points. The build-only scheduler tool translates atomic
waits before the extension is placed in its CPython 3.13 WASI wheel.

The guest probe covers persistent maps, sets and lists, preserves the original
containers after updates, and checks missing-key exceptions. Referencing and
JSON Schema probes exercise this provider through real upstream consumers.
