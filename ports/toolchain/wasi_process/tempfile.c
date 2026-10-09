/* Bounded exclusive temporary files in the virtual filesystem. SDK 34 declares
 * mkstemp but omits its implementation from the Preview 1 libc archive. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wasi/api.h>

int mkstemp(char *template) {
    if (template == NULL) {
        errno = EINVAL;
        return -1;
    }
    size_t length = strnlen(template, 4097);
    if (length > 4096 || length < 6 || memcmp(template + length - 6, "XXXXXX", 6)) {
        errno = EINVAL;
        return -1;
    }
    char *suffix = template + length - 6;
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        unsigned char bytes[6];
        int error = __wasi_random_get(bytes, sizeof(bytes));
        if (error != 0) {
            errno = error;
            break;
        }
        for (unsigned index = 0; index < 6; ++index) suffix[index] = alphabet[bytes[index] % 62];
        int fd = open(template, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) return fd;
        if (errno != EEXIST) break;
    }
    memcpy(suffix, "XXXXXX", 6);
    return -1;
}

FILE *tmpfile(void) {
    char template[] = "/tmp/shellsim-XXXXXX";
    int fd = mkstemp(template);
    if (fd < 0) return NULL;
    if (unlink(template) != 0) {
        int error = errno;
        close(fd);
        errno = error;
        return NULL;
    }
    FILE *file = fdopen(fd, "w+");
    if (file == NULL) {
        int error = errno;
        close(fd);
        errno = error;
    }
    return file;
}
