#include <Python.h>
#include <dlfcn.h>

extern void *__wrap_dlopen(const char *path, int flags);
extern void *__wrap_dlsym(void *handle, const char *name);
extern char *__wrap_dlerror(void);

static PyObject *answer(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    void *handle = __wrap_dlopen(NULL, RTLD_NOW);
    if (!handle) {
        PyErr_SetString(PyExc_RuntimeError, __wrap_dlerror());
        return NULL;
    }
    PyObject *(*make_long)(long) = __wrap_dlsym(handle, "PyLong_FromLong");
    if (!make_long) {
        PyErr_SetString(PyExc_RuntimeError, __wrap_dlerror());
        return NULL;
    }
    return make_long(42);
}

static PyMethodDef methods[] = {
    {"answer", answer, METH_NOARGS, "Call a Python C API export through the main-image handle."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "python_main_handle", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_python_main_handle(void) {
    return PyModule_Create(&definition);
}
