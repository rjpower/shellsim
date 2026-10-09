/* Compile against the real threaded SDK and upstream libffi scalar backend.
 * A process callback pointer is shared; TLS, errno and native Funcs are local.
 */
#include <errno.h>
#include <ffi.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static _Thread_local int bias;
static ffi_cif signature;
static void *callback_code;
static int factor = 2;

static double target(int integer, double real) {
    return integer + real * factor + bias;
}

static void callback(ffi_cif *cif, void *result, void **arguments, void *userdata) {
    if (userdata != &factor || cif != &signature) abort();
    ffi_call(cif, FFI_FN(target), result, arguments);
    errno = 77;
}

static void *worker(void *argument) {
    bias = (int)(intptr_t)argument;
    int integer = 3;
    double real = 2.5, result = 0;
    void *arguments[] = {&integer, &real};
    errno = 11;
    ffi_call(&signature, FFI_FN(callback_code), &result, arguments);
    if (result != bias + 8 || errno != 77) abort();
    return (void *)(intptr_t)bias;
}

int main(void) {
    ffi_type *types[] = {&ffi_type_sint32, &ffi_type_double};
    if (ffi_prep_cif(&signature, FFI_DEFAULT_ABI, 2, &ffi_type_double, types) != FFI_OK)
        abort();
    ffi_closure *closure = ffi_closure_alloc(sizeof(*closure), &callback_code);
    if (!closure || ffi_prep_closure_loc(closure, &signature, callback, &factor, callback_code) != FFI_OK)
        abort();
    bias = 10;
    if (worker((void *)(intptr_t)10) != (void *)(intptr_t)10 || errno != 77) abort();
    pthread_t threads[2];
    if (pthread_create(&threads[0], NULL, worker, (void *)(intptr_t)100) ||
        pthread_create(&threads[1], NULL, worker, (void *)(intptr_t)200)) abort();
    for (int index = 0; index < 2; index++) {
        void *result = NULL;
        if (pthread_join(threads[index], &result) || result != (void *)(intptr_t)(100 * (index + 1)))
            abort();
    }
    if (bias != 10) abort();
    void *old_code = callback_code;
    ffi_closure_free(closure);
    closure = ffi_closure_alloc(sizeof(*closure), &callback_code);
    if (!closure || callback_code == old_code) abort();
    ffi_closure_free(closure);
    puts("threaded libffi: scalar callback, nested call, TLS, errno, join passed");
    return 0;
}
