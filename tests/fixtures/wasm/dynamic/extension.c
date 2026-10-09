#include <Python.h>
#ifndef EXTENSION_NAME
#define EXTENSION_NAME tiny_one
#endif
#ifndef EXTENSION_VALUE
#define EXTENSION_VALUE 17
#endif
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

static long calls;
static PyObject *value(PyObject *self, PyObject *args) {
    (void)self;
    long increment;
    if (!PyArg_ParseTuple(args, "l", &increment)) return NULL;
    calls += increment;
    return PyLong_FromLong(EXTENSION_VALUE + calls);
}
static PyMethodDef methods[] = {
    {"value", value, METH_VARARGS, "Increment independent C state and return its value."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, STRINGIFY(EXTENSION_NAME), NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC CONCAT(PyInit_, EXTENSION_NAME)(void) {
    return PyModule_Create(&definition);
}
