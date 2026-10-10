# LLVM products

One `recipe.json` declares pinned LLVM source and finite `host`, `guest` and
`development` outputs. The default `host` variant produces the patched native
compiler, linker and TableGen tools used by SDK and package builds. The `guest`
variant produces executable Clang, Wasm LLD and archive tools. `development`
installs the guest SDK headers, libraries and compiler resources.

```sh
uv run --no-project --python /path/to/installed-shellsim/bin/python \
  python -m ports toolchain/llvm:guest --store /path/to/ports-store \
  --output /path/to/release --check
```

The canonical `build(ctx)` selects the producer using the resolved static variant.
The host producer retains compatible Ninja state and seals immutable products.
Source, patches, runtime protocols, compiled auxiliary inputs and actual dependency
receipts participate in admission. Shared producer implementation bytes are hashed
automatically. Existing verified LLVM products can be admitted by the explicit
reviewed migration described in [SDK materialization](../../_support/SDK.md).
Changing orchestration does not establish a full cold LLVM bootstrap.

See [guest LLVM](GUEST.md) for compiler behavior and resource limits.
