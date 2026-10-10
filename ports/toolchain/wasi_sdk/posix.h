/* Functions absent from WASI libc, supplied by the canonical virtual runtime. */
#ifndef SHELLSIM_POSIX_H
#define SHELLSIM_POSIX_H
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif
mode_t umask(mode_t mask);
#ifdef __cplusplus
}
#endif
#endif
