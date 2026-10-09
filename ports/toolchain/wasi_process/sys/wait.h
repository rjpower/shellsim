/* Wait status layout for shellsim virtual children; no host process IDs exist. */
#ifndef SHELLSIM_WASI_SYS_WAIT_H
#define SHELLSIM_WASI_SYS_WAIT_H

#include <sys/types.h>

#define WNOHANG 1
#define WIFEXITED(status) (((status) & 0x7f) == 0)
#define WEXITSTATUS(status) (((status) >> 8) & 0xff)
#define WIFSIGNALED(status) (((status) & 0x7f) != 0 && ((status) & 0x7f) != 0x7f)
#define WTERMSIG(status) ((status) & 0x7f)
#define WIFSTOPPED(status) (((status) & 0xff) == 0x7f)
#define WSTOPSIG(status) (((status) >> 8) & 0xff)

#ifdef __cplusplus
extern "C" {
#endif
pid_t waitpid(pid_t pid, int *status, int options);
#ifdef __cplusplus
}
#endif
#endif
