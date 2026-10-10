#include <ffi.h>
#include <stdint.h>
#include <stdio.h>

static int add(int left, int right) { return left + right; }

static void increment(ffi_cif *cif, void *result, void **arguments, void *userdata) {
    (void)cif;
    *(int *)result = *(int *)arguments[0] + *(int *)userdata;
}

int main(void) {
    ffi_cif call;
    ffi_type *types[] = {&ffi_type_sint32, &ffi_type_sint32};
    if (ffi_prep_cif(&call, FFI_DEFAULT_ABI, 2, &ffi_type_sint32, types) != FFI_OK) return 1;
    int left = 17, right = 25, result = 0;
    void *arguments[] = {&left, &right};
    ffi_call(&call, FFI_FN(add), &result, arguments);
    if (result != 42) return 2;

    ffi_type *callback_types[] = {&ffi_type_sint32};
    if (ffi_prep_cif(&call, FFI_DEFAULT_ABI, 1, &ffi_type_sint32, callback_types) != FFI_OK) return 3;
    void *code = NULL;
    ffi_closure *closure = ffi_closure_alloc(sizeof(*closure), &code);
    int offset = 3;
    if (closure == NULL || ffi_prep_closure_loc(closure, &call, increment, &offset, code) != FFI_OK) return 4;
    int (*callback)(int) = code;
    if (callback(39) != 42) return 5;
    ffi_closure_free(closure);

    if (ffi_prep_cif_var(&call, FFI_DEFAULT_ABI, 1, 2, &ffi_type_sint32, types) == FFI_OK) return 6;
    puts("libffi scalar calls, callbacks and variadic rejection passed");
    return 0;
}
