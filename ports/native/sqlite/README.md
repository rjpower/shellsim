SQLite builds the pinned upstream amalgamation as a shared library. Select
`native/sqlite:cli` for the `sqlite3` guest command, or
`python/cpython:stdlib-sqlite3` for CPython's DB-API extension. The upstream WASI
VFS uses dotfile locking. Loadable SQLite extensions and CLI commands that launch
external programs are unavailable on WASI.
