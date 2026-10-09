/* Check the published zlib interface, a real roundtrip and corrupt-input error. */
#include <zlib.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    const unsigned char input[] = "zlib graph adapter: independent target library";
    unsigned char compressed[256], restored[256];
    uLongf size = sizeof(compressed), output = sizeof(restored);
    if (compress2(compressed, &size, input, sizeof(input), 6) != Z_OK) return 1;
    if (uncompress(restored, &output, compressed, size) != Z_OK) return 2;
    if (output != sizeof(input) || memcmp(input, restored, output)) return 3;
    compressed[0] = 0;
    output = sizeof(restored);
    if (uncompress(restored, &output, compressed, size) != Z_DATA_ERROR) return 4;
    puts("zlib: roundtrip and corrupt-input rejection passed");
    return 0;
}
