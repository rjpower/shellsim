# Threaded WASI platform

The canonical definition has `platform` (default) and `tooling` outputs.
SDK tooling admits the pinned SDK 34 archive inventory. The platform producer
builds patched wasi-libc using the separately materialized host compiler.
Its target is `wasm32-wasip1-threads`; ABI and scheduler protocols are declared
in the static producer policy.

The common ports command selects these products through the SDK graph.
`build(ctx)` receives admitted source archives, native CMake/Ninja and exact
compiler dependency receipts. The driver publishes the resulting verified
product. SDK cache migration retains original inventories and receipt hashes.
See [SDK materialization](../../_support/SDK.md).

The [standalone pthread fixtures](tests/README.md) audit raw wait/notify and the
virtual scheduler. Fixture reproduction uses its own pinned metadata and can
consume existing verified historical toolchains. It is separate from production
package authoring.
