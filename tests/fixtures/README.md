# zlib source fixture

`zlib-1.3.2.tar.gz` is the unmodified release archive for upstream
[`madler/zlib` tag `v1.3.2`](https://github.com/madler/zlib/releases/tag/v1.3.2).
SHA-256: `b99a0b86c0ba9360ec7e78c4f1e43b1cbdf1e6936c8fa0f6835c0cd694a495a1`.
The archive is loaded into shellsim's VFS by `tests/zlib_build.rs`; the test does not execute
upstream scripts or compiler tools on the host.
