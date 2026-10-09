/* Exec transfers control to a new virtual image; no host process is involved. */
#include "exec_port.h"
#include "process_abi.h"
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

__attribute__((import_module("shellsim_posix_v1"), import_name("process_exec")))
extern int shellsim_process_exec(const char *path, char *const argv[], uint32_t argc,
                                char *const environment[], uint32_t environment_count,
                                uint32_t flags);

static int vector_count(char *const values[], uint32_t *count) {
    if (values == NULL) {
        *count = 0;
        return 0;
    }
    for (uint32_t index = 0; index <= SHELLSIM_PROCESS_MAX_ARGUMENTS; ++index) {
        if (values[index] == NULL) {
            *count = index;
            return 0;
        }
    }
    errno = E2BIG;
    return -1;
}

static int replace(const char *path, char *const argv[], char *const environment[], uint32_t flags) {
    uint32_t argc, environment_count;
    if (path == NULL || argv == NULL) {
        errno = EFAULT;
        return -1;
    }
    if (vector_count(argv, &argc) || vector_count(environment, &environment_count)) return -1;
    int error = shellsim_process_exec(path, argv, argc, environment, environment_count, flags);
    /* Success unwinds the old image inside the host import and cannot return. */
    errno = error != 0 ? error : EIO;
    return -1;
}

int execve(const char *path, char *const argv[], char *const environment[]) {
    return replace(path, argv, environment, 0);
}

int execv(const char *path, char *const argv[]) {
    return execve(path, argv, environ);
}

int execvp(const char *path, char *const argv[]) {
    return replace(path, argv, environ, SHELLSIM_PROCESS_SEARCH_PATH);
}

pid_t wait(int *status) { return waitpid(-1, status, 0); }
