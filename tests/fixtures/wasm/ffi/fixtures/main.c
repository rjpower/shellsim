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
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_alloc_typed")))
extern int ffi_closure_alloc_typed(uint32_t dispatch, uint32_t userdata, const uint8_t *tags,
                                   uint32_t count, uint32_t result_tag, uint32_t *index);

static int callback_dispatch(uint32_t userdata, int argument) {
    return (int)userdata + argument;
}

static int typed_double_dispatch(uint32_t userdata, const uint64_t *arguments,
                                 uint64_t *result) {
    double value;
    memcpy(&value, arguments, sizeof(value));
    value += (double)userdata;
    memcpy(result, &value, sizeof(value));
    return 0;
}

__attribute__((visibility("default"))) int ffi_main_export(int value) {
    return value + 2;
}

int main(void) {
    void *main_image = dlopen(NULL, RTLD_NOW);
    if (!main_image) return 11;
    int (*from_main)(int) = (int (*)(int))dlsym(main_image, "ffi_main_export");
    if (!from_main || from_main(40) != 42) return 12;
    if (dlopen(NULL, 0) || !dlerror()) return 13;
    void *provider = dlopen("/lib/libffi_fixture.so", RTLD_NOW | RTLD_LOCAL);
    if (!provider) {
        puts(dlerror());
        return 1;
    }
    void *add = dlsym(provider, "ffi_add");
    void *mix = dlsym(provider, "ffi_mix");
    void *identity = dlsym(provider, "ffi_identity");
    void *apply = dlsym(provider, "ffi_apply");
    double (*apply_double)(double (*)(double), double) = dlsym(provider, "ffi_apply_double");
    int (*via_host)(int, int) = (int (*)(int, int))dlsym(provider, "ffi_via_host");
    if (!add || !mix || !identity || !apply || !apply_double || !via_host) return 2;
    if (via_host(19, 23) != 42) return 10;

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
    uint32_t typed_callback = 0;
    const uint8_t double_tag = 4;
    if (ffi_closure_alloc_typed((uint32_t)(uintptr_t)typed_double_dispatch, 2, &double_tag, 1, 4,
                                &typed_callback)) return 14;
    if (apply_double((double (*)(double))(uintptr_t)typed_callback, 1.5) != 4.5) return 15;
    if (ffi_closure_release(typed_callback)) return 16;
    puts("separate SDK provider, side FFI import and nested callback: ok");
    return 0;
}
