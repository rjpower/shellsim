# Native package catalog

`NativePackageUniverse(catalog_path).install(environment, specs)` resolves bare
names, exact pins and PEP440 version ranges from a bounded local catalog. It checks
all transitive version constraints together, including exact compiled-provider
artifact identities, before mounting one staged rootfs transaction. Unavailable
releases are errors; installation never builds packages or invokes host programs.

Format 1 records contain `name`, `version`, `kind` (`build-tool`, `devel`, `runtime`),
artifact directory and digest, source-to-VFS `destinations`, and typed `dependencies`
with requirement and kind. `recipe_name` permits an explicit catalog alias such as
zlib-devel for zlib; `recipe_port` names dependency keys when different. Profile,
ABI and toolchain hashes must match each artifact. Linked dependency edges must
match recipe versions and exact dependency-artifact hashes. Independent executables
may use different admitted profiles. Exports cannot write project directories.

Each Environment owns its native installation ledger and lock. Installed artifact
identities and destinations are immutable across catalogs and universe instances.
Repeated exact installation preserves existing files; distinct compatible packages
can be added. Provider replacement, upgrades and uninstall are not supported yet.
A failed install preserves both the prior ledger and the virtual filesystem.

The producer assembles pinned make 4.4.1, shellsim-c-toolchain 0.1.30 and zlib-devel 1.3.1.
The compiler distribution contains TinyCC 0.9.28rc at commit 22a2e10, its corresponding
source, notices and verified sysroot. Zlib wrapping preserves the original SDK 34
artifact provenance and unchanged archive/header bytes, with the measured two-unit
TinyCC C/zlib proof. It is not a rebuild or a general compiler-ABI certification.

Run the producer with `PYTHONPATH=.:toolchain/src uv run --no-project --python 3.13
python -m ports.native.catalog.build MAKE_ARTIFACT ZLIB_ARTIFACT C_ZLIB_PROOF OUTPUT`. The output catalog
installs make at `/usr/bin/make`, cc at `/usr/bin/cc`, compiler/sysroot files at `/tcc`
and `/wasi-sysroot`, and zlib development files at `/opt/zlib`.

The CPython release producer can seal this catalog as an optional `native.zip`
asset using `--native-catalog /path/to/catalog.json`. Call
`NativePackageUniverse.from_release(trusted_descriptor, offline=True)` after
the asset is cached, then use the same `install` method and version specs.
Native delivery validates all listed artifacts and each version's dependency
closure; it does not select a CPython interpreter or download the host uv
resolver. The native archive is separate so a task that only needs make or cc
can load it without the Python cohort.
