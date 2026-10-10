# Native graph publication

Guest native artifacts publish through the existing `native.zip` release asset.
`Environment.from_release(descriptor, tools=["package>=1"])` resolves and mounts
verified exports atomically. Host tools and target-platform products stay outside
both guest catalogs and acceptance.

Native artifacts and installed dependency closures allow 1 GiB of exports;
release archives and unpacked catalogs allow 2 GiB. These transport bounds leave
room for the compiler, SDK and scientific libraries together. Installation still
obeys the environment's disk and memory budgets.

Native recipes may declare `install` with `name` (a package alias), `kind`
(`build-tool`, `devel`, or `runtime`), and `destinations` mapping exported relative
paths to canonical absolute guest paths. Unspecified destinations use
`/usr/local/<export>`. The environment's normal PATH includes `/usr/local/bin`.
Guest-tool recipes use `build-tool`; libraries default to `devel`. Every verified
export is installed, including licenses. `exports.tools` defines executable mode
0755; other exports use 0644, matching the existing native installer contract.

Catalog dependency edges preserve exact artifact identities. Linked edges match
sealed `target_dependencies` and `dependency_artifacts`. Install-only edges have
`linked:false` and match `runtime_dependencies` and `runtime_artifacts`. Both
participate in resolution and installation. Build and platform edges never become
guest dependencies. The catalog retains the actual target triple and toolchain
receipt, rather than changing threaded artifact identities. A record's `target`
overrides the catalog default, allowing independently executable serial tools
and threaded runtime files to coexist. Linked edges must retain the same target.

A native port can declare a `shell` probe with a port-local `script` and bounded
`args`. Native and Python probes remain available. Each native-port probe installs
its selected package through the public release before execution, allowing shell
or Python scripts to inspect headers and run installed guest commands. Host C
compilation probes still use admitted exact dependency archives, then execute the
result only in the guest. Native `link_flags` accepts bounded
`-Wl,--wrap=<symbol>` switches for canonical libc facade ownership; it does not
enable host library search or arbitrary compiler options.

Recipes may declare `empty_directories` as payload-relative paths mapped to mode
`0755`. These directories must exist and be empty when sealed. Their paths and
modes are part of the artifact identity and survive native release transport.
`install.directories` may override their guest destinations; the default is
`/usr/local/<payload-relative-path>`. Repeated installation restores declared
empty directories along with verified files. Undeclared directory records are
not accepted by the release verifier.

Exported core Wasm tools receive the admitted cohort ABI marker during sealing.
An existing conflicting marker rejects the artifact; shell scripts retain their
original bytes.
