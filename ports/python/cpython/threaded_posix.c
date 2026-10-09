/* Keep the process CWD stable across fuel yields in threaded libc callers.
 * The shared SDK owner group includes this lock alongside getcwd and cwd.
 * Reusing its scheduler-backed lock also prevents archive selection from
 * bringing in a second CWD owner. No guest allocator runs during replay. */
#define getcwd shellsim_unlocked_getcwd
#define chdir shellsim_unlocked_chdir
#define __wasilibc_find_relpath_alloc shellsim_unlocked_find_relpath_alloc
#include "../../toolchain/wasi_sdk/posix.c"
#undef getcwd
#undef chdir
#undef __wasilibc_find_relpath_alloc

volatile int __wasilibc_cwd_lock;
extern void __lock(volatile int *);
extern void __unlock(volatile int *);

char *getcwd(char *buffer, size_t size) {
    int allocated = !buffer;
    if (allocated) {
        if (!size) size = sizeof(cwd_storage);
        buffer = malloc(size);
        if (!buffer) { errno = ENOMEM; return NULL; }
    }
    __lock(&__wasilibc_cwd_lock);
    char *result = shellsim_unlocked_getcwd(buffer, size);
    __unlock(&__wasilibc_cwd_lock);
    if (!result && allocated) free(buffer);
    return result;
}

int chdir(const char *path) {
    __lock(&__wasilibc_cwd_lock);
    int result = shellsim_unlocked_chdir(path);
    __unlock(&__wasilibc_cwd_lock);
    return result;
}

int __wasilibc_find_relpath_alloc(const char *path, const char **prefix,
                                char **relative, size_t *capacity, int can_realloc) {
    char absolute[8192];
    if (path[0] != '/') {
        size_t path_length = strlen(path);
        __lock(&__wasilibc_cwd_lock);
        size_t cwd_length = strlen(cwd_storage);
        if (path_length >= sizeof(absolute) - cwd_length - 1) {
            __unlock(&__wasilibc_cwd_lock);
            errno = ENAMETOOLONG;
            return -1;
        }
        memcpy(absolute, cwd_storage, cwd_length);
        __unlock(&__wasilibc_cwd_lock);
        absolute[cwd_length] = '/';
        memcpy(absolute + cwd_length + 1, path, path_length + 1);
        path = absolute;
    }
    /* Allocation and preopen lookup happen after taking a stable CWD snapshot. */
    return shellsim_unlocked_find_relpath_alloc(path, prefix, relative, capacity, can_realloc);
}
