/* Measure the numerical and Fortran ABI through actual guest calls. */
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"

_Static_assert(sizeof(blasint) == 4, "LP64 BLAS integer width");
_Static_assert(sizeof(BLASLONG) == 4, "wasm32 long width");
_Static_assert(sizeof(uintptr_t) == 4, "wasm32 address width");
extern int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern int dgesv_(int *, int *, double *, int *, int *, double *, int *, int *);
extern void zdotc_(double complex *, int *, double complex *, int *, double complex *, int *);
extern void cdotu_(float complex *, int *, float complex *, int *, float complex *, int *);
extern float sdot_(int *, float *, int *, float *, int *);
extern float slange_(char *, int *, int *, float *, int *, float *);
extern int ilaver_(int *, int *, int *);

int main(void) {
    assert(setenv("SHELLSIM_OPENBLAS_ABI", "123", 1) == 0);
    const char *abi = getenv("SHELLSIM_OPENBLAS_ABI");
    assert(abi != NULL && strcmp(abi, "123") == 0 && atoi(abi) == 123);
    char no = 'N';
    int n = 2, one = 1, info = 0, pivots[2];
    double alpha = 1, beta = 0;
    double a[] = {1, 3, 2, 4}, b[] = {5, 7, 6, 8}, c[4] = {0};
    assert(dgemm_(&no, &no, &n, &n, &n, &alpha, a, &n, b, &n, &beta, c, &n) == 0);
    assert(c[0] == 19 && c[1] == 43 && c[2] == 22 && c[3] == 50);
    double system[] = {3, 1, 1, 2}, rhs[] = {9, 8};
    assert(dgesv_(&n, &one, system, &n, pivots, rhs, &n, &info) == 0);
    assert(info == 0 && fabs(rhs[0] - 2) < 1e-12 && fabs(rhs[1] - 3) < 1e-12);
    int invalid = -1;
    dgesv_(&invalid, &one, system, &n, pivots, rhs, &n, &info);
    assert(info == -1);
    double complex x[] = {1 + 2*I, 3 - I}, y[] = {2 - I, -1 + 4*I}, result;
    zdotc_(&result, &n, x, &one, y, &one);
    assert(creal(result) == -7 && cimag(result) == 6);
    float complex sx[] = {1 + 2*I, 3 - I}, sy[] = {2 - I, -1 + 4*I}, sr;
    cdotu_(&sr, &n, sx, &one, sy, &one);
    assert(crealf(sr) == 5 && cimagf(sr) == 16);
    float realx[] = {1, 2}, realy[] = {3, 4};
    assert(sdot_(&n, realx, &one, realy, &one) == 11);
    char norm = 'M'; float workspace[2];
    assert(slange_(&norm, &n, &one, realy, &n, workspace) == 4);
    int major, minor, patch;
    ilaver_(&major, &minor, &patch);
    assert(major >= 3);
    puts("OpenBLAS: dgemm, dgesv, invalid input, complex and REAL ABI passed");
    return 0;
}
