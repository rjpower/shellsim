/* Functions absent from WASI libc, supplied by the canonical virtual runtime. */
#ifndef SHELLSIM_POSIX_H
#define SHELLSIM_POSIX_H
#include <sys/types.h>
mode_t umask(mode_t mask);
#endif
