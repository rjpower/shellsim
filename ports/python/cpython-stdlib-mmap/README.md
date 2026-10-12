The upstream CPython extension uses the SDK-pinned wasi-libc memory-map emulation,
rebuilt as a PIC archive. Cross-process shared mappings remain unsupported.
File maps copy virtual file contents; `flush()` cannot persist changes.
