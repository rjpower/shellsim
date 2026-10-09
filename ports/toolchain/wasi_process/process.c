/* Canonical libc process entry points backed by virtual PIDs and descriptors.
 * The Rust kernel repeats every bound check before touching process state. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

_Static_assert(sizeof(void *) == 4, "shellsim process ABI requires Wasm32 pointers");

__attribute__((import_module("shellsim_posix_v1"), import_name("process_pipe")))
extern int shellsim_process_pipe(int flags, int *read_fd, int *write_fd);

__attribute__((import_module("shellsim_posix_v1"), import_name("process_spawn")))
extern int shellsim_process_spawn(
    const char *path, char *const argv[], uint32_t argc, char *const environment[],
    uint32_t environment_count, const struct shellsim_process_action *actions,
    uint32_t action_count, uint32_t options, uint32_t signal_defaults, pid_t *pid
);

__attribute__((import_module("shellsim_posix_v1"), import_name("process_wait")))
extern int shellsim_process_wait(pid_t pid, int options, int *status, pid_t *observed_pid);

__attribute__((import_module("shellsim_posix_v1"), import_name("process_kill")))
extern int shellsim_process_kill(pid_t pid, int signal_number);

__attribute__((import_module("shellsim_posix_v1"), import_name("process_signal_disposition")))
extern int shellsim_process_signal_disposition(int signal_number, int mode);

__attribute__((import_module("shellsim_posix_v1"), import_name("process_identity")))
extern int shellsim_process_identity(int operation);

pid_t getpid(void) { return shellsim_process_identity(SHELLSIM_PROCESS_ID_SELF); }
pid_t getppid(void) { return shellsim_process_identity(SHELLSIM_PROCESS_ID_PARENT); }

extern void (*__real_signal(int signal_number, void (*handler)(int)))(int);
extern int __real_open(const char *path, int flags, ...);
extern int __real_openat(int directory, const char *path, int flags, ...);

static int finish_open(int fd, int flags) {
    if (fd < 0 || !(flags & O_CLOEXEC)) return fd;
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) == 0) return fd;
    int error = errno;
    close(fd);
    errno = error;
    return -1;
}

int __wrap_open(const char *path, int flags, ...) {
    int sdk_flags = flags & ~O_CLOEXEC;
    int fd;
    if (flags & O_CREAT) {
        va_list arguments;
        va_start(arguments, flags);
        mode_t mode = va_arg(arguments, mode_t);
        va_end(arguments);
        fd = __real_open(path, sdk_flags, mode);
    } else {
        fd = __real_open(path, sdk_flags);
    }
    return finish_open(fd, flags);
}

int __wrap_openat(int directory, const char *path, int flags, ...) {
    int sdk_flags = flags & ~O_CLOEXEC;
    int fd;
    if (flags & O_CREAT) {
        va_list arguments;
        va_start(arguments, flags);
        mode_t mode = va_arg(arguments, mode_t);
        va_end(arguments);
        fd = __real_openat(directory, path, sdk_flags, mode);
    } else {
        fd = __real_openat(directory, path, sdk_flags);
    }
    return finish_open(fd, flags);
}

void (*__wrap_signal(int signal_number, void (*handler)(int)))(int) {
    switch (signal_number) {
        case SIGHUP: case SIGINT: case SIGUSR1: case SIGUSR2:
        case SIGPIPE: case SIGTERM: case SIGCHLD: case SIGCONT: case SIGXFSZ:
            break;
        case SIGBUS: case SIGILL: case SIGFPE: case SIGABRT: case SIGSEGV:
            // WASI libc delivers raise() synchronously to its own signal table.
            // The virtual kernel does not accept these numbers from process_kill.
            return __real_signal(signal_number, handler);
        default:
            errno = ENOTSUP;
            return SIG_ERR;
    }
    if (handler != SIG_DFL && handler != SIG_IGN) {
        errno = ENOTSUP;
        return SIG_ERR;
    }
    void (*previous)(int) = __real_signal(signal_number, handler);
    if (previous == SIG_ERR) return SIG_ERR;
    int mode = handler == SIG_IGN ? SHELLSIM_PROCESS_SIGNAL_IGNORE : SHELLSIM_PROCESS_SIGNAL_DEFAULT;
    int error = shellsim_process_signal_disposition(signal_number, mode);
    if (error) {
        __real_signal(signal_number, previous);
        errno = error;
        return SIG_ERR;
    }
    return previous;
}

static int checked_string(const char *value, size_t *remaining) {
    if (!value || !*remaining) return E2BIG;
    size_t length = strnlen(value, *remaining);
    if (length == *remaining) return E2BIG;
    *remaining -= length + 1;
    return 0;
}

static int checked_vector(char *const values[], uint32_t maximum, uint32_t *count, size_t *remaining) {
    *count = 0;
    if (!values) return 0;
    while (*count < maximum && values[*count]) {
        int error = checked_string(values[*count], remaining);
        if (error) return error;
        ++*count;
    }
    return *count == maximum && values[*count] ? E2BIG : 0;
}

static int add_action(posix_spawn_file_actions_t *actions, struct shellsim_process_action action) {
    if (!actions) return EINVAL;
    if (actions->count >= SHELLSIM_PROCESS_MAX_ACTIONS) return E2BIG;
    actions->actions[actions->count++] = action;
    return 0;
}

static int add_path_action(
    posix_spawn_file_actions_t *actions, struct shellsim_process_action action, const char *path
) {
    if (!actions || !path) return EINVAL;
    if (actions->count >= SHELLSIM_PROCESS_MAX_ACTIONS) return E2BIG;
    if (actions->owned_path_bytes >= SHELLSIM_PROCESS_MAX_STRING_BYTES) return E2BIG;
    size_t remaining = SHELLSIM_PROCESS_MAX_STRING_BYTES - actions->owned_path_bytes;
    size_t length = strnlen(path, remaining);
    if (length == remaining) return E2BIG;
    char *copy = malloc(length + 1);
    if (!copy) return ENOMEM;
    memcpy(copy, path, length + 1);
    uint32_t index = actions->count;
    action.path = (uint32_t)(uintptr_t)copy;
    actions->actions[index] = action;
    actions->owned_paths[index] = copy;
    actions->owned_path_bytes += length + 1;
    actions->count = index + 1;
    return 0;
}

int posix_spawn_file_actions_init(posix_spawn_file_actions_t *actions) {
    if (!actions) return EINVAL;
    memset(actions, 0, sizeof(*actions));
    return 0;
}

int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions) {
    if (!actions) return EINVAL;
    for (uint32_t index = 0; index < actions->count; ++index) free(actions->owned_paths[index]);
    memset(actions, 0, sizeof(*actions));
    return 0;
}

int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t *actions, int fd) {
    if (fd < 0) return EBADF;
    return add_action(actions, (struct shellsim_process_action){.kind=SHELLSIM_PROCESS_CLOSE, .fd=fd});
}

int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions, int source, int destination) {
    if (source < 0 || destination < 0) return EBADF;
    return add_action(actions, (struct shellsim_process_action){
        .kind=SHELLSIM_PROCESS_DUP2, .fd=source, .target=destination
    });
}

int posix_spawn_file_actions_addclosefrom_np(posix_spawn_file_actions_t *actions, int minimum) {
    if (minimum < 0) return EBADF;
    return add_action(actions, (struct shellsim_process_action){
        .kind=SHELLSIM_PROCESS_CLOSEFROM, .fd=minimum
    });
}

int posix_spawn_file_actions_addopen(
    posix_spawn_file_actions_t *actions, int fd, const char *path, int flags, mode_t mode
) {
    if (fd < 0 || !path) return EINVAL;
    return add_path_action(actions, (struct shellsim_process_action){
        .kind=SHELLSIM_PROCESS_OPEN, .fd=fd, .flags=flags, .mode=mode
    }, path);
}

int posix_spawn_file_actions_addchdir_np(posix_spawn_file_actions_t *actions, const char *path) {
    if (!path) return EINVAL;
    return add_path_action(actions, (struct shellsim_process_action){
        .kind=SHELLSIM_PROCESS_CHDIR
    }, path);
}

int posix_spawnattr_init(posix_spawnattr_t *attributes) {
    if (!attributes) return EINVAL;
    memset(attributes, 0, sizeof(*attributes));
    return 0;
}

int posix_spawnattr_destroy(posix_spawnattr_t *attributes) {
    return attributes ? 0 : EINVAL;
}

int posix_spawnattr_setflags(posix_spawnattr_t *attributes, short flags) {
    if (!attributes) return EINVAL;
    if (flags & ~POSIX_SPAWN_SETSIGDEF) return ENOTSUP;
    attributes->flags = (unsigned short)flags;
    return 0;
}

int posix_spawnattr_setpgroup(posix_spawnattr_t *attributes, pid_t group) {
    (void)attributes;
    (void)group;
    return ENOTSUP;
}

int posix_spawnattr_setsigdefault(posix_spawnattr_t *attributes, const void *signals) {
    (void)attributes;
    (void)signals;
    return ENOTSUP;
}

int posix_spawnattr_setsigdefault_mask_np(posix_spawnattr_t *attributes, uint32_t signals) {
    if (!attributes || (signals & 1u)) return EINVAL;
    attributes->signal_defaults = signals;
    return 0;
}

int posix_spawnattr_setsigmask(posix_spawnattr_t *attributes, const void *signals) {
    (void)attributes;
    (void)signals;
    return ENOTSUP;
}

static int spawn_impl(
    pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const argv[], char *const environment[],
    uint32_t options
) {
    if (!pid || !path || !argv || !argv[0]) return EINVAL;
    size_t remaining = SHELLSIM_PROCESS_MAX_STRING_BYTES;
    int error = checked_string(path, &remaining);
    if (error) return error;
    uint32_t argc, environment_count;
    error = checked_vector(argv, SHELLSIM_PROCESS_MAX_ARGUMENTS, &argc, &remaining);
    if (error) return error;
    error = checked_vector(environment, SHELLSIM_PROCESS_MAX_ENVIRONMENT, &environment_count, &remaining);
    if (error) return error;
    uint32_t action_count = actions ? actions->count : 0;
    if (action_count > SHELLSIM_PROCESS_MAX_ACTIONS) return E2BIG;
    for (uint32_t index = 0; index < action_count; ++index) {
        const struct shellsim_process_action *action = &actions->actions[index];
        if (action->kind == SHELLSIM_PROCESS_OPEN || action->kind == SHELLSIM_PROCESS_CHDIR) {
            error = checked_string((const char *)(uintptr_t)action->path, &remaining);
            if (error) return error;
        }
    }
    uint32_t defaults = 0;
    if (attributes) {
        if (attributes->flags & ~POSIX_SPAWN_SETSIGDEF) return ENOTSUP;
        if (attributes->flags & POSIX_SPAWN_SETSIGDEF) {
            options |= SHELLSIM_PROCESS_SETSIGDEF;
            defaults = attributes->signal_defaults;
        }
    }
    return shellsim_process_spawn(
        path, argv, argc, environment, environment_count,
        actions ? actions->actions : 0, action_count, options, defaults, pid
    );
}

int posix_spawn(
    pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const argv[], char *const environment[]
) {
    return spawn_impl(pid, path, actions, attributes, argv, environment, 0);
}

int posix_spawnp(
    pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const argv[], char *const environment[]
) {
    return spawn_impl(pid, path, actions, attributes, argv, environment, SHELLSIM_PROCESS_SEARCH_PATH);
}

int pipe2(int fds[2], int flags) {
    if (!fds) { errno = EFAULT; return -1; }
    if (flags & ~(O_CLOEXEC | O_NONBLOCK)) { errno = EINVAL; return -1; }
    int error = shellsim_process_pipe(flags, &fds[0], &fds[1]);
    if (error) { errno = error; return -1; }
    return 0;
}

int pipe(int fds[2]) { return pipe2(fds, 0); }

pid_t waitpid(pid_t pid, int *status, int options) {
    if (options & ~WNOHANG) { errno = EINVAL; return -1; }
    int ignored_status = 0;
    if (!status) status = &ignored_status;
    pid_t observed = 0;
    int error = shellsim_process_wait(pid, options, status, &observed);
    if (error) { errno = error; return -1; }
    return observed;
}

int kill(pid_t pid, int signal_number) {
    int error = shellsim_process_kill(pid, signal_number);
    if (error) { errno = error; return -1; }
    return 0;
}
