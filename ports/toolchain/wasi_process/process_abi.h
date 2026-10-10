/* Versioned process ABI for shellsim's WASI Preview 1 runtime.
 * All pointers refer to bounded guest memory. The kernel validates every vector,
 * string, action and output address before it creates a virtual child. */
#ifndef SHELLSIM_WASI_PROCESS_ABI_H
#define SHELLSIM_WASI_PROCESS_ABI_H

#include <fcntl.h>
#include <stdint.h>

/* SDK 34 defines O_CLOEXEC as zero, which cannot express pipe2/open intent. */
#if O_CLOEXEC != 0
#error "shellsim's O_CLOEXEC port must be reviewed for this SDK"
#endif
#undef O_CLOEXEC
#define O_CLOEXEC 0x00080000
#define SHELLSIM_PROCESS_O_CLOEXEC O_CLOEXEC

#define SHELLSIM_PROCESS_MAX_ARGUMENTS 256
#define SHELLSIM_PROCESS_MAX_ENVIRONMENT 256
#define SHELLSIM_PROCESS_MAX_ACTIONS 64
#define SHELLSIM_PROCESS_MAX_STRING_BYTES 131072

#define SHELLSIM_PROCESS_SEARCH_PATH 1u
#define SHELLSIM_PROCESS_SETSIGDEF 2u
#define SHELLSIM_PROCESS_SIGNAL_DEFAULT 0
#define SHELLSIM_PROCESS_SIGNAL_IGNORE 1
#define SHELLSIM_PROCESS_ID_SELF 0
#define SHELLSIM_PROCESS_ID_PARENT 1

enum shellsim_process_action_kind {
    SHELLSIM_PROCESS_CLOSE = 1,
    SHELLSIM_PROCESS_DUP2 = 2,
    SHELLSIM_PROCESS_CLOSEFROM = 3,
    SHELLSIM_PROCESS_OPEN = 4,
    SHELLSIM_PROCESS_CHDIR = 5,
};

/* Fixed Wasm32 layout; path is a guest offset to a NUL-terminated string. */
struct shellsim_process_action {
    uint32_t kind;
    int32_t fd;
    int32_t target;
    int32_t flags;
    uint32_t mode;
    uint32_t path;
};

#ifdef __cplusplus
static_assert(sizeof(struct shellsim_process_action) == 24, "process action ABI size");
#else
_Static_assert(sizeof(struct shellsim_process_action) == 24, "process action ABI size");
#endif

#endif
