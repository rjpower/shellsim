/* Forced into CPython's WASI posixmodule object so it sees the virtual wait API. */
#ifndef SHELLSIM_WASI_PROCESS_PORT_H
#define SHELLSIM_WASI_PROCESS_PORT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/wait.h>

int kill(pid_t pid, int signal_number);
pid_t getppid(void);

#ifdef __cplusplus
}
#endif
#endif
