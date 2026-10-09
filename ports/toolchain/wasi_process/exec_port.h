/* Canonical exec/wait declarations for SDK 34 consumers. */
#ifndef SHELLSIM_WASI_EXEC_PORT_H
#define SHELLSIM_WASI_EXEC_PORT_H
#include <sys/types.h>
int execve(const char *path, char *const argv[], char *const environment[]);
int execv(const char *path, char *const argv[]);
int execvp(const char *path, char *const argv[]);
pid_t wait(int *status);
#endif
