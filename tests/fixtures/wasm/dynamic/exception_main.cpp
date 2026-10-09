// Verify one canonical C++ runtime across separately linked WASM modules.
#include "exception_types.hpp"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <typeinfo>

FixtureError::~FixtureError() = default;

static int destroyed;
extern "C" void mark_destroyed(int value) { destroyed += value; }
extern "C" void main_throw() { throw FixtureError("from main"); }

int main(int argc, char **argv) {
    void *library = dlopen(argc > 1 ? argv[1] : "/lib/exception.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::printf("load: %s\n", dlerror()); return 1; }
    auto side_throw = reinterpret_cast<void (*)()>(dlsym(library, "side_throw"));
    auto side_catch = reinterpret_cast<int (*)(void (*)())>(dlsym(library, "side_catch"));
    auto side_type = reinterpret_cast<const std::type_info *(*)()>(dlsym(library, "side_type"));
    if (!side_throw || !side_catch || !side_type) return 2;
    if (side_type() != &typeid(FixtureError)) return 3;
    try { side_throw(); } catch (const FixtureError &error) {
        if (std::strcmp(error.what(), "from side") != 0) return 4;
    }
    if (side_catch(main_throw) != 42 || destroyed != 3) return 5;
    std::printf("cross-module typed catch/rethrow/destructors: %d\n", destroyed);
    return 0;
}
