"""Build upstream fcntl over virtual whole-file locks and descriptor controls."""

from ports.api import BuildContext, python_extension


def build(ctx: BuildContext):
    source = ctx.require_native().source / "Modules/shellsim_fcntl.c"
    source.write_text(
        """
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/file.h>
#include <sys/ioctl.h>

__attribute__((import_module("shellsim_posix_v1"), import_name("descriptor_flock")))
extern int virtual_flock(int fd, int operation);
__attribute__((import_module("shellsim_posix_v1"), import_name("descriptor_control")))
extern int virtual_control(int fd, uint32_t operation, int argument, int *result);

int shellsim_flock(int fd, int operation) {
    int error = virtual_flock(fd, operation);
    if (!error) return 0;
    errno = error;
    return -1;
}

int shellsim_fcntl(int fd, int command, ...) {
    uint32_t operation;
    int argument = 0;
    switch (command) {
        case F_GETFD: operation = 1; break;
        case F_SETFD: operation = 2; break;
        case F_DUPFD: operation = 3; break;
#ifdef F_DUPFD_CLOEXEC
        case F_DUPFD_CLOEXEC: operation = 4; break;
#endif
        default: errno = ENOTSUP; return -1;
    }
    if (command != F_GETFD) {
        va_list arguments;
        va_start(arguments, command);
        argument = va_arg(arguments, int);
        va_end(arguments);
    }
    int result;
    int error = virtual_control(fd, operation, argument, &result);
    if (!error) return result;
    errno = error;
    return -1;
}

int shellsim_ioctl(int fd, int command, ...) {
    errno = ENOTSUP;
    return -1;
}
"""
    )
    return python_extension(
        ctx,
        module="fcntl",
        sources=("Modules/fcntlmodule.c", "Modules/shellsim_fcntl.c"),
        defines=(
            "HAVE_FLOCK=1",
            "HAVE_SYS_FILE_H=1",
            "flock=shellsim_flock",
            "fcntl=shellsim_fcntl",
            "ioctl=shellsim_ioctl",
        ),
        licenses=("LICENSE",),
    )
