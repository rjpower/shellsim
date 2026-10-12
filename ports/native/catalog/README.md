# Native catalogs

Run `ports build <native-package>`, then `ports export --url-prefix <store-http-url>`.
Use `--store` for products and `--state` for client metadata. The catalog records exact toolchain and compiled dependency
relationships. The public installer resolves supported package constraints,
verifies inventory and ABI, then mounts files atomically.

Previously sealed TinyCC and static SDK catalogs remain installable with their
original metadata. The obsolete standalone catalog assembly CLI is retired.
See [native ports](../README.md) and [port publication](../../README.md).
