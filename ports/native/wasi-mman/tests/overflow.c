/* A map header must not wrap a guest-controlled allocation length. */
#define _WASI_EMULATED_MMAN
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>

int main(void) {
    void *map = mmap(0, SIZE_MAX, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(map == MAP_FAILED && errno == ENOMEM);
    return 0;
}
