// Each separately linked extension imports the executable's C++ runtime.
#include <Python.h>
#include <stdexcept>

#define JOIN_INNER(a, b) a##b
#define JOIN(a, b) JOIN_INNER(a, b)
#define STRING_INNER(value) #value
#define STRING(value) STRING_INNER(value)

static int destroyed;
namespace {
struct Guard { ~Guard() { ++destroyed; } };
}
static void throw_error() {
    Guard guard;
    throw std::runtime_error("independent C++ extension");
}
static PyObject *thrower(PyObject *, PyObject *) {
    return PyCapsule_New(reinterpret_cast<void *>(throw_error), "shellsim.thrower", nullptr);
}
static PyObject *catch_call(PyObject *, PyObject *capsule) {
    void *pointer = PyCapsule_GetPointer(capsule, "shellsim.thrower");
    if (!pointer) return nullptr;
    auto callback = reinterpret_cast<void (*)()>(pointer);
    Guard guard;
    try { try { callback(); } catch (...) { throw; } }
    catch (const std::runtime_error &error) { return PyUnicode_FromString(error.what()); }
    PyErr_SetString(PyExc_RuntimeError, "callback did not throw");
    return nullptr;
}
static PyObject *count(PyObject *, PyObject *) { return PyLong_FromLong(destroyed); }
static PyMethodDef methods[] = {
    {"thrower", thrower, METH_NOARGS, nullptr},
    {"catch_call", catch_call, METH_O, nullptr},
    {"destroyed", count, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, STRING(EXTENSION_NAME), nullptr, -1, methods,
};
PyMODINIT_FUNC JOIN(PyInit_, EXTENSION_NAME)(void) { return PyModule_Create(&module); }
