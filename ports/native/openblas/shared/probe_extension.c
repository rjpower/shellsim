/* A separately linked consumer exercises the original numerical probe through DT_NEEDED. */
#include <Python.h>
#define main run_numerical_probe
#include "../probe.c"
#undef main

static PyObject *probe(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    if (run_numerical_probe() != 0) {
        PyErr_SetString(PyExc_AssertionError, "numerical probe failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"probe", probe, METH_NOARGS, "Check BLAS, LAPACK and the numerical ABI."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "openblas_probe", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit_openblas_probe(void) { return PyModule_Create(&module); }
