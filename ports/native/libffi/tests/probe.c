#include <dlfcn.h>
#include <ffi.h>
#include <stdint.h>
#include <stdio.h>

static void callback(ffi_cif *cif, void *result, void **arguments, void *userdata) {
    (void)cif;
    *(double *)result = *(double *)arguments[0] + *(double *)userdata;
}

static void callback_sint8(ffi_cif *cif, void *result, void **arguments, void *userdata) {
    (void)cif;
    (void)arguments;
    (void)userdata;
    *(int8_t *)result = -7;
}

static void callback_sint16(ffi_cif *cif, void *result, void **arguments, void *userdata) {
    (void)cif;
    (void)arguments;
    (void)userdata;
    *(int16_t *)result = -300;
}

int main(void) {
    void *provider = dlopen("/lib/libffi_probe.so", RTLD_NOW | RTLD_LOCAL);
    if (provider == NULL) return 1;
    void *add = dlsym(provider, "ffi_add");
    void *apply = dlsym(provider, "ffi_apply_double");
    if (add == NULL || apply == NULL) return 2;

    ffi_cif cif;
    ffi_type *integer_types[] = {&ffi_type_sint32, &ffi_type_sint32};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &ffi_type_sint32,
                     integer_types) != FFI_OK) return 3;
    int left = 17, right = 25, sum = 0;
    void *integer_values[] = {&left, &right};
    ffi_call(&cif, (void (*)(void))add, &sum, integer_values);
    if (sum != 42) return 4;

    ffi_type *double_type[] = {&ffi_type_double};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, &ffi_type_double,
                     double_type) != FFI_OK) return 5;
    void *code = NULL;
    ffi_closure *closure = ffi_closure_alloc(sizeof(*closure), &code);
    if (closure == NULL || code == NULL) return 6;
    double userdata = 2.0;
    if (ffi_prep_closure_loc(closure, &cif, callback, &userdata, code) != FFI_OK)
        return 7;
    double rejected_userdata = 100.0;
    if (ffi_prep_closure_loc(closure, &cif, callback, &rejected_userdata, code) != FFI_BAD_TYPEDEF)
        return 8;

    ffi_cif apply_cif;
    ffi_type *apply_types[] = {&ffi_type_pointer, &ffi_type_double};
    if (ffi_prep_cif(&apply_cif, FFI_DEFAULT_ABI, 2, &ffi_type_double,
                     apply_types) != FFI_OK) return 9;
    double input = 1.5, result = 0;
    void *apply_values[] = {&code, &input};
    ffi_call(&apply_cif, (void (*)(void))apply, &result, apply_values);
    if (result != 4.5) return 10;
    ffi_closure_free(closure);

    void *promoted = dlsym(provider, "ffi_expect_promoted");
    void *return_s8 = dlsym(provider, "ffi_return_sint8");
    void *return_u8 = dlsym(provider, "ffi_return_uint8");
    void *return_s16 = dlsym(provider, "ffi_return_sint16");
    void *return_u16 = dlsym(provider, "ffi_return_uint16");
    if (!promoted || !return_s8 || !return_u8 || !return_s16 || !return_u16)
        return 13;
    int8_t negative = -1;
    uint8_t positive = 255;
    ffi_type *narrow_types[] = {&ffi_type_sint8, &ffi_type_uint8};
    void *narrow_values[] = {&negative, &positive};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &ffi_type_sint32,
                     narrow_types) != FFI_OK) return 14;
    ffi_call(&cif, (void (*)(void))promoted, &sum, narrow_values);
    if (sum != 254) return 15;
    int16_t negative16 = -2;
    uint16_t positive16 = 65534;
    ffi_type *narrow16_types[] = {&ffi_type_sint16, &ffi_type_uint16};
    void *narrow16_values[] = {&negative16, &positive16};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &ffi_type_sint32,
                     narrow16_types) != FFI_OK) return 16;
    ffi_call(&cif, (void (*)(void))promoted, &sum, narrow16_values);
    if (sum != 65532) return 17;
    struct {
        void *function;
        ffi_type *type;
        ffi_sarg expected;
    } narrow_returns[] = {
        {return_s8, &ffi_type_sint8, -1},
        {return_u8, &ffi_type_uint8, 255},
        {return_s16, &ffi_type_sint16, -2},
        {return_u16, &ffi_type_uint16, 65534},
    };
    for (unsigned i = 0; i < 4; ++i) {
        if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 0,
                         narrow_returns[i].type, NULL) != FFI_OK) return 18;
        ffi_sarg narrow_result = 0x12345678;
        ffi_call(&cif, (void (*)(void))narrow_returns[i].function,
                 &narrow_result, NULL);
        if (narrow_result != narrow_returns[i].expected) return 19;
    }

    int (*apply8)(int8_t (*)(int8_t), int8_t) = dlsym(provider, "ffi_apply_sint8");
    int (*apply16)(int16_t (*)(int16_t), int16_t) = dlsym(provider, "ffi_apply_sint16");
    if (apply8 == NULL || apply16 == NULL) return 20;
    ffi_type *s8_arguments[] = {&ffi_type_sint8};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, &ffi_type_sint8, s8_arguments) != FFI_OK) return 21;
    closure = ffi_closure_alloc(sizeof(*closure), &code);
    if (closure == NULL || ffi_prep_closure_loc(closure, &cif, callback_sint8, NULL, code) != FFI_OK)
        return 22;
    if (apply8((int8_t (*)(int8_t))code, 1) != -7) return 23;
    ffi_closure_free(closure);
    ffi_type *s16_arguments[] = {&ffi_type_sint16};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, &ffi_type_sint16, s16_arguments) != FFI_OK) return 24;
    closure = ffi_closure_alloc(sizeof(*closure), &code);
    if (closure == NULL || ffi_prep_closure_loc(closure, &cif, callback_sint16, NULL, code) != FFI_OK)
        return 25;
    if (apply16((int16_t (*)(int16_t))code, 1) != -300) return 26;
    ffi_closure_free(closure);

    ffi_type *aggregate_elements[] = {&ffi_type_sint32, NULL};
    ffi_type aggregate = {0, 0, FFI_TYPE_STRUCT, aggregate_elements};
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 0, &aggregate, NULL) != FFI_BAD_TYPEDEF)
        return 11;
    if (ffi_prep_cif_var(&cif, FFI_DEFAULT_ABI, 1, 2, &ffi_type_sint32,
                         integer_types) != FFI_BAD_TYPEDEF)
        return 12;
    puts("upstream libffi common code and SDK34 scalar backend: ok");
    return 0;
}
