/* Check inherited SIGPIPE disposition after a virtual posix_spawn. */
#include <errno.h>
#include <unistd.h>

int main(void) {
    char token;
    if (read(0, &token, 1) != 1) return 2;
    if (write(1, &token, 1) == -1 && errno == EPIPE) return 3;
    return 4;
}
