#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

__attribute__((import_module("shellsim_ffi_v1"), import_name("invoke")))
extern int ffi_invoke(uint32_t index, const uint8_t *tags, const uint64_t *values,
                      uint32_t count, uint32_t result_tag, uint64_t *result);
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_alloc")))
extern int ffi_closure_alloc(uint32_t dispatch, uint32_t userdata, uint32_t *index);
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_release")))
extern int ffi_closure_release(uint32_t index);

static int callback_dispatch(uint32_t userdata, int argument) {
    return (int)userdata + argument;
}

int main(void) {
    void *provider = dlopen("/lib/libffi_proof.so", RTLD_NOW | RTLD_LOCAL);
    if (!provider) {
        puts(dlerror());
        return 1;
    }
    void *add = dlsym(provider, "ffi_add");
    void *mix = dlsym(provider, "ffi_mix");
    void *identity = dlsym(provider, "ffi_identity");
    void *apply = dlsym(provider, "ffi_apply");
    int (*via_host)(int (*)(int, int), int, int) =
        (int (*)(int (*)(int, int), int, int))dlsym(provider, "ffi_via_host");
    if (!add || !mix || !identity || !apply || !via_host) return 2;
    if (via_host((int (*)(int, int))add, 19, 23) != 42) return 10;

    uint8_t tags[2] = {1, 1};
    uint64_t values[2] = {17, 25};
    uint64_t result = 0;
    if (ffi_invoke((uint32_t)(uintptr_t)add, tags, values, 2, 1, &result) || result != 42)
        return 3;

    double input = 1.5;
    tags[0] = 4;
    memcpy(&values[0], &input, sizeof(input));
    values[1] = 2;
    if (ffi_invoke((uint32_t)(uintptr_t)mix, tags, values, 2, 4, &result)) return 4;
    double mixed;
    memcpy(&mixed, &result, sizeof(mixed));
    if (mixed != 3.5) return 5;

    int object = 7;
    tags[0] = 1;
    values[0] = (uint32_t)(uintptr_t)&object;
    if (ffi_invoke((uint32_t)(uintptr_t)identity, tags, values, 1, 1, &result)
        || (uint32_t)result != (uint32_t)(uintptr_t)&object)
        return 6;

    uint32_t callback = 0;
    if (ffi_closure_alloc((uint32_t)(uintptr_t)callback_dispatch, 10, &callback)) return 7;
    tags[1] = 1;
    values[0] = callback;
    values[1] = 7;
    if (ffi_invoke((uint32_t)(uintptr_t)apply, tags, values, 2, 1, &result)
        || result != 18)
        return 8;
    if (ffi_closure_release(callback) || ffi_closure_release(callback) != 28) return 9;
    puts("separate SDK provider, side FFI import and nested callback: ok");
    return 0;
}
