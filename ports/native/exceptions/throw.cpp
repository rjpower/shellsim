/* Use the pinned C++ runtime's typed exception across a separately built archive. */
#include <stdexcept>
extern "C" void library_throw() { throw std::runtime_error("library error"); }
