# Native ports

Each native port has one static recipe and typed Python build entrypoint.
The common driver admits SDK products and exact dependency artifacts, merges
headers/libraries/pkg-config files into a private sysroot, and publishes verified
exports. CMake, Meson and make helpers consume only admitted target tools and
host build bindings. Guest programs cannot acquire host capabilities.

Artifacts bind source, ABI, compiler/platform and actual dependency receipt identities.
Native catalogs install declared guest tools and development data through the public
API. Dependency closure, hashes and destination conflicts are checked before mounting.
Package installation constraints remain separate from exact build graph selections.

See [port authoring](../README.md), [native helpers](../_support/native_adapters.md)
and [SDK products](../_support/SDK.md). Historical sealed catalogs retain their
original wire format and provenance.
