/* SPDX-License-Identifier: MIT */
/* Bridge POSIX calls absent from WASI Preview 1 to shellsim's virtual syscalls. */

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern int shellsim_path_chmod(const char *, unsigned, unsigned)
    __asm__("shellsim.path_chmod");

int chmod(const char *path, mode_t mode)
{
    int error;
    char absolute[PATH_MAX];
    const char *target = path;
    if (!path) {
        errno = EFAULT;
        return -1;
    }
    if (path[0] != '/') {
        size_t base_len, path_len;
        if (!getcwd(absolute, sizeof absolute))
            return -1;
        base_len = strlen(absolute);
        path_len = strlen(path);
        if (base_len + 1 + path_len >= sizeof absolute) {
            errno = ENAMETOOLONG;
            return -1;
        }
        absolute[base_len++] = '/';
        memcpy(absolute + base_len, path, path_len + 1);
        target = absolute;
    }
    error = shellsim_path_chmod(target, strlen(target), (unsigned)mode);
    if (error) {
        errno = error;
        return -1;
    }
    return 0;
}
