#include <stdio.h>
#include <string.h>
#include <stdexcept>
extern "C" void library_throw();
static int destructed;
struct Guard { ~Guard() { ++destructed; } };
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "uncaught") == 0) library_throw();
    try {
        Guard guard;
        try { library_throw(); } catch (...) { throw; }
        return 1;
    } catch (const std::runtime_error &) {
        if (destructed != 1) return 2;
    }
    puts("C++: cross-library catch, rethrow and destruction passed");
    return 0;
}
