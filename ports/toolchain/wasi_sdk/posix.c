/* Canonical libc descriptor operations backed by the simulated process FD table.
 * WASI Preview1 has no duplication or exec-inheritance operation. The explicit
 * import extends that boundary without granting a host descriptor capability. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wasi/api.h>
#include <wasi/libc-find-relpath.h>
#include "posix.h"

__attribute__((import_module("shellsim_posix_v1"), import_name("umask")))
extern unsigned virtual_umask(unsigned mask);

mode_t umask(mode_t mask) { return virtual_umask(mask); }

__attribute__((import_module("shellsim_posix_v1"), import_name("cwd_get")))
extern int virtual_cwd_get(char *, unsigned);
__attribute__((import_module("shellsim_posix_v1"), import_name("cwd_set")))
extern int virtual_cwd_set(const char *, unsigned, char *, unsigned);

/* SDK relative-path resolution reads this canonical pointer. Keeping its storage
 * bounded avoids a second independently allocated cwd or a stale pointer. */
static char cwd_storage[4096] = "/";
char *__wasilibc_cwd = cwd_storage;

__attribute__((constructor)) static void initialize_cwd(void) {
    int error = virtual_cwd_get(cwd_storage, sizeof(cwd_storage));
    if (error) _Exit(126);
}

char *getcwd(char *buffer, size_t size) {
    size_t required = strlen(cwd_storage) + 1;
    if (buffer && !size) { errno = EINVAL; return NULL; }
    if (size && size < required) { errno = ERANGE; return NULL; }
    if (!buffer) {
        buffer = malloc(size ? size : required);
        if (!buffer) { errno = ENOMEM; return NULL; }
    }
    memcpy(buffer, cwd_storage, required);
    return buffer;
}

int chdir(const char *path) {
    int error = virtual_cwd_set(path, strlen(path), cwd_storage, sizeof(cwd_storage));
    if (error) { errno = error; return -1; }
    return 0;
}

/* This public SDK hook normally shares chdir.c.obj with chdir. Define the
 * complete symbol group so archive selection cannot pull the old cwd owner.
 * Its documented caller-owned buffer and preopen lookup contract are preserved. */
int __wasilibc_find_relpath_alloc(const char *path, const char **prefix,
                                char **relative, size_t *capacity, int can_realloc) {
    char absolute[8192];
    const char *lookup = path;
    if (path[0] != '/') {
        size_t cwd_length = strlen(cwd_storage);
        size_t path_length = strlen(path);
        if (path_length >= sizeof(absolute) - cwd_length - 1) {
            errno = ENAMETOOLONG;
            return -1;
        }
        memcpy(absolute, cwd_storage, cwd_length);
        absolute[cwd_length] = '/';
        memcpy(absolute + cwd_length + 1, path, path_length + 1);
        lookup = absolute;
    }
    const char *suffix;
    int fd = __wasilibc_find_abspath(lookup, prefix, &suffix);
    if (fd < 0) return -1;
    size_t required = strlen(suffix) + 1;
    if (*capacity < required) {
        if (!can_realloc) { errno = ERANGE; return -1; }
        char *replacement = realloc(*relative, required);
        if (!replacement) { errno = ENOMEM; return -1; }
        *relative = replacement;
        *capacity = required;
    }
    memcpy(*relative, suffix, required);
    return fd;
}

__attribute__((import_module("shellsim_posix_v1"), import_name("descriptor_control")))
extern int descriptor_control(int, unsigned, int, int *);

static int control(int fd, unsigned operation, int argument) {
    int result;
    int error = descriptor_control(fd, operation, argument, &result);
    if (error) {
        errno = error;
        return -1;
    }
    return result;
}

int dup(int fd) { return control(fd, 3, 0); }
int dup2(int fd, int destination) { return control(fd, 5, destination); }
int __dup3(int fd, int destination, int flags) {
    if (fd == destination || flags) {
        errno = EINVAL;
        return -1;
    }
    return dup2(fd, destination);
}
int dup3(int fd, int destination, int flags) { return __dup3(fd, destination, flags); }

int fcntl(int fd, int command, ...) {
    int argument = 0;
    if (command == F_SETFD || command == F_DUPFD || command == F_DUPFD_CLOEXEC || command == F_SETFL) {
        va_list arguments;
        va_start(arguments, command);
        argument = va_arg(arguments, int);
        va_end(arguments);
    }
    switch (command) {
        case F_GETFD: return control(fd, 1, 0);
        case F_SETFD: return control(fd, 2, argument);
        case F_DUPFD: return control(fd, 3, argument);
        case F_DUPFD_CLOEXEC: return control(fd, 4, argument);
        case F_GETFL: {
            __wasi_fdstat_t state;
            int error = __wasi_fd_fdstat_get(fd, &state);
            if (error) { errno = error; return -1; }
            return ((state.fs_rights_base & __WASI_RIGHTS_FD_READ) ? O_RDONLY : 0)
                 | ((state.fs_rights_base & __WASI_RIGHTS_FD_WRITE) ? O_WRONLY : 0)
                 | state.fs_flags;
        }
        case F_SETFL: {
            if (argument & ~(O_APPEND | O_ACCMODE)) { errno = EINVAL; return -1; }
            int error = __wasi_fd_fdstat_set_flags(fd, argument & O_APPEND);
            if (error) { errno = error; return -1; }
            return 0;
        }
        default: errno = EINVAL; return -1;
    }
}
