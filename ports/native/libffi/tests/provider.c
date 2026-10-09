#include <stdint.h>

__attribute__((import_module("shellsim_ffi_v1"), import_name("invoke")))
extern int ffi_invoke(uint32_t index, const uint8_t *tags, const uint64_t *values,
                      uint32_t count, uint32_t result_tag, uint64_t *result);

__attribute__((visibility("default"))) int ffi_add(int left, int right) {
    return left + right;
}

__attribute__((visibility("default"))) double ffi_mix(double left, int right) {
    return left + (double)right;
}

__attribute__((visibility("default"))) void *ffi_identity(void *pointer) {
    return pointer;
}

__attribute__((visibility("default"))) int ffi_apply(int (*callback)(int), int value) {
    return callback(value) + 1;
}

__attribute__((visibility("default"))) double ffi_apply_double(double (*callback)(double),
                                                                 double value) {
    return callback(value) + 1.0;
}

__attribute__((visibility("default"))) int ffi_apply_sint8(int8_t (*callback)(int8_t), int8_t value) {
    return callback(value);
}

__attribute__((visibility("default"))) int ffi_apply_sint16(int16_t (*callback)(int16_t), int16_t value) {
    return callback(value);
}

__attribute__((visibility("default"))) int ffi_expect_promoted(int left, int right) {
    return left + right;
}

__attribute__((visibility("default"))) int8_t ffi_return_sint8(void) {
    return -1;
}

__attribute__((visibility("default"))) uint8_t ffi_return_uint8(void) {
    return 255;
}

__attribute__((visibility("default"))) int16_t ffi_return_sint16(void) {
    return -2;
}

__attribute__((visibility("default"))) uint16_t ffi_return_uint16(void) {
    return 65534;
}

__attribute__((visibility("default"))) int ffi_via_host(int left, int right) {
    const uint8_t tags[2] = {1, 1};
    const uint64_t values[2] = {(uint32_t)left, (uint32_t)right};
    uint64_t result = 0;
    if (ffi_invoke((uint32_t)(uintptr_t)ffi_add, tags, values, 2, 1, &result)) return -1;
    return (int)result;
}
