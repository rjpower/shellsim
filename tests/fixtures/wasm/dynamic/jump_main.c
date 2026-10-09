/* Recover a target longjmp that crosses a separately compiled library. */
#include <dlfcn.h>
#include <setjmp.h>
#include <stdio.h>

int main(void) {
    void *library = dlopen("/lib/jump.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) { printf("load: %s\n", dlerror()); return 1; }
    void (*jump)(jmp_buf) = dlsym(library, "side_jump");
    if (!jump) return 2;
    jmp_buf buffer;
    int value = setjmp(buffer);
    if (!value) jump(buffer);
    if (value != 37) return 3;
    printf("cross-module longjmp: %d\n", value);
    return 0;
}
