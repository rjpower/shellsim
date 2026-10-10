/* Exercise linked libc behavior through the virtual process and descriptor APIs. */
#define _WASI_EMULATED_SIGNAL 1
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "exec_port.h"
#include "process_port.h"
#include "tempfile_port.h"
extern char **environ;
static void *worker(void *parent_errno) {
    assert(&errno != parent_errno);
    char *cwd = getcwd(NULL, 0);
    assert(cwd && strcmp(cwd, "/tmp") == 0);
    free(cwd);
    errno = EINVAL;
    return NULL;
}
int main(void) {
    int descriptors[2];
    assert(pipe(descriptors) == 0);
    assert(write(descriptors[1], "guest", 5) == 5);
    char buffer[5];
    assert(read(descriptors[0], buffer, sizeof buffer) == 5);
    assert(memcmp(buffer, "guest", sizeof buffer) == 0);
    assert(close(descriptors[0]) == 0);
    assert(close(descriptors[1]) == 0);
    errno = 0;
    assert(read(descriptors[0], buffer, sizeof buffer) == -1);
    assert(errno == EBADF);
    char *original_cwd = getcwd(NULL, 0);
    assert(original_cwd);
    assert(chdir("/tmp") == 0);
    char cwd[32];
    assert(getcwd(cwd, sizeof cwd) == cwd);
    assert(strcmp(cwd, "/tmp") == 0);
    int relative = open("shellsim-posix-cwd", O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(relative >= 0);
    assert(close(relative) == 0);
    assert(unlink("/tmp/shellsim-posix-cwd") == 0);
    errno = EDOM;
    int *const parent_errno = &errno;
    pthread_t thread;
    assert(pthread_create(&thread, NULL, worker, parent_errno) == 0);
    assert(&errno == parent_errno);
    assert(errno == EDOM);
    assert(pthread_join(thread, NULL) == 0);
    assert(&errno == parent_errno);
    assert(errno == EDOM);
    assert(chdir(original_cwd) == 0);
    free(original_cwd);
    char name[] = "/tmp/shellsim-posix-XXXXXX";
    int temporary = mkstemp(name);
    assert(temporary >= 0);
    assert(close(temporary) == 0);
    assert(unlink(name) == 0);
    char *arguments[] = {"sh", "-c", "exit 7", NULL};
    pid_t child;
    assert(posix_spawn(&child, "/bin/sh", NULL, NULL, arguments, environ) == 0);
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 7);
    /* Ordinary compiler links carry more than 1,400 runtime-retention roots. */
    char **large = calloc(4098, sizeof *large);
    assert(large);
    large[0] = "sh";
    large[1] = "-c";
    large[2] = "i=0; for a; do test \"$a\" = \"arg-$i\" || exit 9; i=$((i+1)); done; test $i = 1500";
    large[3] = "argv-probe";
    for (int i = 0; i < 1500; ++i) {
        large[i + 4] = malloc(32);
        assert(large[i + 4]);
        snprintf(large[i + 4], 32, "arg-%d", i);
    }
    assert(posix_spawn(&child, "/bin/sh", NULL, NULL, large, environ) == 0);
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char *oversized_arguments[4098];
    for (int i = 0; i < 4097; ++i) oversized_arguments[i] = "arg";
    oversized_arguments[4097] = NULL;
    assert(posix_spawn(&child, "/bin/sh", NULL, NULL, oversized_arguments, environ) == E2BIG);
    errno = 0;
    assert(execve("/bin/sh", oversized_arguments, environ) == -1);
    assert(errno == E2BIG);
    char *oversized_environment[258];
    for (int i = 0; i < 257; ++i) oversized_environment[i] = "KEY=value";
    oversized_environment[257] = NULL;
    assert(posix_spawn(&child, "/bin/sh", NULL, NULL, arguments, oversized_environment) == E2BIG);
    errno = 0;
    assert(execve("/bin/sh", arguments, oversized_environment) == -1);
    assert(errno == E2BIG);
    assert(getpid() > 0);
    /* Successful exec replaces this image and the same ordered-argv child exits 0. */
    execve("/bin/sh", large, environ);
    assert(0 && "execve unexpectedly returned");
    return 1;
}
