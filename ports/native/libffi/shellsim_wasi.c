/* SDK 34 libffi primitive lowering through shellsim_ffi_v1.
 *
 * Upstream libffi's wasm backend uses Emscripten JavaScript. This backend
 * keeps its common cif/type code and supplies the WASI table boundary. Only
 * scalar values with a direct Wasm type are admitted; unsupported signatures
 * fail while preparing the cif, before any foreign call or callback exists.
 */

#include <ffi.h>
#include <ffi_common.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SHELLSIM_MAX_ARGS 16
#define SHELLSIM_CLOSURE_MAGIC 0x53464649u

__attribute__((import_module("shellsim_ffi_v1"), import_name("invoke")))
extern int shellsim_invoke(uint32_t index, const uint8_t *tags,
                           const uint64_t *arguments, uint32_t count,
                           uint32_t result_tag, uint64_t *result);
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_reserve")))
extern int shellsim_closure_reserve(uint32_t *slot);
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_define_typed")))
extern int shellsim_closure_define(uint32_t slot, uint32_t dispatcher,
                                   uint32_t userdata, const uint8_t *tags,
                                   uint32_t count, uint32_t result_tag);
__attribute__((import_module("shellsim_ffi_v1"), import_name("closure_release")))
extern int shellsim_closure_release(uint32_t slot);

typedef struct {
    uint32_t magic;
    uint32_t slot;
    uint32_t prepared;
    uint32_t reserved;
} closure_header;

static uint8_t scalar_tag(const ffi_type *type) {
    switch (type->type) {
        case FFI_TYPE_VOID: return 0;
        case FFI_TYPE_FLOAT: return 3;
        case FFI_TYPE_DOUBLE: return 4;
        case FFI_TYPE_SINT64:
        case FFI_TYPE_UINT64: return 2;
        case FFI_TYPE_INT:
        case FFI_TYPE_SINT8:
        case FFI_TYPE_UINT8:
        case FFI_TYPE_SINT16:
        case FFI_TYPE_UINT16:
        case FFI_TYPE_SINT32:
        case FFI_TYPE_UINT32:
        case FFI_TYPE_POINTER: return 1;
        default: return 255;
    }
}

ffi_status FFI_HIDDEN ffi_prep_cif_machdep(ffi_cif *cif) {
    if (cif->abi != FFI_WASM32) return FFI_BAD_ABI;
    if (cif->nargs > SHELLSIM_MAX_ARGS || scalar_tag(cif->rtype) == 255)
        return FFI_BAD_TYPEDEF;
    for (unsigned i = 0; i < cif->nargs; ++i) {
        if (scalar_tag(cif->arg_types[i]) == 0 ||
            scalar_tag(cif->arg_types[i]) == 255)
            return FFI_BAD_TYPEDEF;
    }
    cif->nfixedargs = cif->nargs;
    return FFI_OK;
}

ffi_status FFI_HIDDEN ffi_prep_cif_machdep_var(ffi_cif *cif,
                                                unsigned nfixedargs,
                                                unsigned ntotalargs) {
    (void)cif;
    (void)nfixedargs;
    (void)ntotalargs;
    return FFI_BAD_TYPEDEF;
}

void ffi_call(ffi_cif *cif, void (*fn)(void), void *rvalue, void **avalue) {
    uint8_t tags[SHELLSIM_MAX_ARGS] = {0};
    uint64_t arguments[SHELLSIM_MAX_ARGS] = {0};
    uint64_t result = 0;
    for (unsigned i = 0; i < cif->nargs; ++i) {
        tags[i] = scalar_tag(cif->arg_types[i]);
        memcpy(&arguments[i], avalue[i], cif->arg_types[i]->size);
        if (cif->arg_types[i]->type == FFI_TYPE_SINT8)
            arguments[i] = (uint32_t)(int32_t)*(int8_t *)avalue[i];
        else if (cif->arg_types[i]->type == FFI_TYPE_SINT16)
            arguments[i] = (uint32_t)(int32_t)*(int16_t *)avalue[i];
    }
    int status = shellsim_invoke((uint32_t)(uintptr_t)fn, tags, arguments,
                                 cif->nargs, scalar_tag(cif->rtype), &result);
    if (status != 0) __builtin_trap();
    if (rvalue != NULL && cif->rtype->type != FFI_TYPE_VOID) {
        switch (cif->rtype->type) {
            case FFI_TYPE_SINT8: *(ffi_sarg *)rvalue = (int8_t)result; break;
            case FFI_TYPE_SINT16: *(ffi_sarg *)rvalue = (int16_t)result; break;
            case FFI_TYPE_UINT8: *(ffi_arg *)rvalue = (uint8_t)result; break;
            case FFI_TYPE_UINT16: *(ffi_arg *)rvalue = (uint16_t)result; break;
            default: memcpy(rvalue, &result, cif->rtype->size); break;
        }
    }
}

void *ffi_closure_alloc(size_t size, void **code) {
    if (code == NULL || size < sizeof(ffi_closure) || size > 4096)
        return NULL;
    closure_header *header = calloc(1, sizeof(*header) + size);
    if (header == NULL) return NULL;
    uint32_t slot = 0;
    if (shellsim_closure_reserve(&slot) != 0) {
        free(header);
        return NULL;
    }
    header->magic = SHELLSIM_CLOSURE_MAGIC;
    header->slot = slot;
    *code = (void *)(uintptr_t)slot;
    return header + 1;
}

void ffi_closure_free(void *pointer) {
    if (pointer == NULL) return;
    closure_header *header = (closure_header *)pointer - 1;
    if (header->magic != SHELLSIM_CLOSURE_MAGIC) __builtin_trap();
    if (shellsim_closure_release(header->slot) != 0) __builtin_trap();
    header->magic = 0;
    free(header);
}

static int closure_dispatch(uint32_t userdata, const uint64_t *arguments,
                            uint64_t *result) {
    ffi_closure *closure = (ffi_closure *)(uintptr_t)userdata;
    void *values[SHELLSIM_MAX_ARGS];
    for (unsigned i = 0; i < closure->cif->nargs; ++i)
        values[i] = (void *)&arguments[i];
    *result = 0;
    closure->fun(closure->cif, result, values, closure->user_data);
    if (closure->cif->rtype->type == FFI_TYPE_SINT8)
        *result = (uint32_t)(int32_t)*(int8_t *)result;
    else if (closure->cif->rtype->type == FFI_TYPE_SINT16)
        *result = (uint32_t)(int32_t)*(int16_t *)result;
    return 0;
}

ffi_status ffi_prep_closure_loc(ffi_closure *closure, ffi_cif *cif,
                                void (*fun)(ffi_cif *, void *, void **, void *),
                                void *user_data, void *codeloc) {
    if (closure == NULL || cif == NULL || fun == NULL ||
        cif->nargs > SHELLSIM_MAX_ARGS)
        return FFI_BAD_TYPEDEF;
    closure_header *header = (closure_header *)closure - 1;
    if (header->magic != SHELLSIM_CLOSURE_MAGIC ||
        header->prepared || codeloc != (void *)(uintptr_t)header->slot)
        return FFI_BAD_TYPEDEF;
    uint8_t tags[SHELLSIM_MAX_ARGS] = {0};
    for (unsigned i = 0; i < cif->nargs; ++i)
        tags[i] = scalar_tag(cif->arg_types[i]);
    ffi_cif *previous_cif = closure->cif;
    void (*previous_fun)(ffi_cif *, void *, void **, void *) = closure->fun;
    void *previous_userdata = closure->user_data;
    closure->cif = cif;
    closure->fun = fun;
    closure->user_data = user_data;
    int status = shellsim_closure_define(
        header->slot, (uint32_t)(uintptr_t)closure_dispatch,
        (uint32_t)(uintptr_t)closure, tags, cif->nargs, scalar_tag(cif->rtype));
    if (status == 0) {
        header->prepared = 1;
        return FFI_OK;
    }
    closure->cif = previous_cif;
    closure->fun = previous_fun;
    closure->user_data = previous_userdata;
    return FFI_BAD_TYPEDEF;
}
