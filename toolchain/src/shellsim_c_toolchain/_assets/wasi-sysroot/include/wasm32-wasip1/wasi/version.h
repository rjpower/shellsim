#ifndef __WASI_VERSION_H
#define __WASI_VERSION_H

#ifndef __wasi__
#error <wasi/version.h> is only supported on WASI platforms.
#endif

// After llvm/llvm-project#165345 these can be conditionally defined by a clang
// version test here to use the upstream definitions.
#define __wasip1__
/* #undef __wasip2__ */
/* #undef __wasip3__ */

// The `__wasi_cooperative_threads__` define is used to test whether cooperative
// threads are enabled for this build of wasi-libc. This is detected at
// build-time and only works on supported versions of clang.
#ifdef __wasm_libcall_thread_context__
/* #undef __wasi_cooperative_threads__ */
#endif

#define __wasi_sdk_major__ 34
#define __wasi_sdk_version__ "34.0"

#endif /* __WASI_VERSION_H */
