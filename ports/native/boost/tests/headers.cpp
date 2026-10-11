#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

int main() {
    static_assert(BOOST_VERSION == 108500);
    std::string value = "CHEMISTRY";
    boost::algorithm::to_lower(value);
    return value == "chemistry" ? 0 : 1;
}
