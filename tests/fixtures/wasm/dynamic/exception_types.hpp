// The same interface is compiled independently in the executable and library.
#include <stdexcept>

struct FixtureError : std::runtime_error {
    explicit FixtureError(const char *message) : std::runtime_error(message) {}
    ~FixtureError() override;
};
