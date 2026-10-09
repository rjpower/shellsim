// Runtime functions and the exception tag are imports from the executable.
#include "exception_types.hpp"
#include <cstring>
#include <typeinfo>

extern "C" void mark_destroyed(int);
struct Guard { int value; ~Guard() { mark_destroyed(value); } };

extern "C" const std::type_info *side_type() { return &typeid(FixtureError); }
extern "C" void side_throw() {
    Guard guard{1};
    throw FixtureError("from side");
}
extern "C" int side_catch(void (*callback)()) {
    Guard guard{2};
    try { try { callback(); } catch (...) { throw; } }
    catch (const FixtureError &error) {
        return std::strcmp(error.what(), "from main") == 0 ? 42 : -1;
    }
    return 0;
}
