# Native catalogs

The common ports driver publishes native catalogs from selected verified artifacts.
Request native packages or guest tools with `python -m ports`, `--store`, and a
fresh `--output`. The catalog records exact toolchain and compiled dependency
relationships. The public installer resolves supported package constraints,
verifies inventory and ABI, then mounts files atomically.

Previously sealed TinyCC and static SDK catalogs remain installable with their
original metadata. The obsolete standalone catalog assembly CLI is retired.
See [native ports](../README.md) and [port publication](../../README.md).
