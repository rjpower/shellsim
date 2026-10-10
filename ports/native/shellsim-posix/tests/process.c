/* Exercise linked libc behavior through the virtual process and descriptor APIs. */
#define _WASI_EMULATED_SIGNAL 1
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "process_port.h"
#include "tempfile_port.h"
extern char **environ;
static void *worker(void *parent_errno) {
    assert(&errno != parent_errno);
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
    errno = EDOM;
    int *const parent_errno = &errno;
    pthread_t thread;
    assert(pthread_create(&thread, NULL, worker, parent_errno) == 0);
    assert(&errno == parent_errno);
    assert(errno == EDOM);
    assert(pthread_join(thread, NULL) == 0);
    assert(&errno == parent_errno);
    assert(errno == EDOM);
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
    assert(getpid() > 0);
    return 0;
}
