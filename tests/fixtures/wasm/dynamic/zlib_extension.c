#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <zlib.h>

static PyObject *roundtrip(PyObject *self, PyObject *input) {
    char *bytes;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(input, &bytes, &length) < 0) return NULL;
    uLongf capacity = compressBound((uLong)length);
    unsigned char *compressed = PyMem_Malloc(capacity);
    if (!compressed) return PyErr_NoMemory();
    int status = compress2(compressed, &capacity, (const Bytef *)bytes, (uLong)length, 6);
    PyObject *result = NULL;
    if (status == Z_OK) {
        result = PyBytes_FromStringAndSize(NULL, length);
        if (result) {
            uLongf restored = (uLongf)length;
            status = uncompress((Bytef *)PyBytes_AS_STRING(result), &restored, compressed, capacity);
            if (status != Z_OK || restored != (uLongf)length) Py_CLEAR(result);
        }
    }
    PyMem_Free(compressed);
    if (!result && !PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "zlib roundtrip failed");
    return result;
}

static PyMethodDef methods[] = {{"roundtrip", roundtrip, METH_O, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "zlib_consumer", NULL, -1, methods};
PyMODINIT_FUNC PyInit_zlib_consumer(void) { return PyModule_Create(&module); }
