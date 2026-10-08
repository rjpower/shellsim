/* The process-lifetime bridge exposes no host capabilities. */
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

__attribute__((import_module("shellsim_dylink_v1"), import_name("open")))
extern uint32_t shellsim_open(const char *, uint32_t, uint32_t);
__attribute__((import_module("shellsim_dylink_v1"), import_name("symbol")))
extern uint32_t shellsim_symbol(uint32_t, const char *, uint32_t);
__attribute__((import_module("shellsim_dylink_v1"), import_name("error")))
extern uint32_t shellsim_error(char *, uint32_t);

void *__wrap_dlopen(const char *path, int flags) {
    if (!path) return NULL;
    return (void *)(uintptr_t)shellsim_open(path, strlen(path), flags);
}
void *__wrap_dlsym(void *handle, const char *symbol) {
    return (void *)(uintptr_t)shellsim_symbol((uintptr_t)handle, symbol, strlen(symbol));
}
char *__wrap_dlerror(void) {
    static char error[2048];
    return shellsim_error(error, sizeof(error)) ? error : NULL;
}
int __wrap_dlclose(void *handle) {
    (void)handle;
    /* No code or storage is unloaded; the initial profile retains every library. */
    return 0;
}
