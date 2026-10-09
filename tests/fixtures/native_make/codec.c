
#include <zlib.h>
#include <string.h>
int check_codec(void) {
    unsigned char input[2048], compressed[4096], output[2048];
    uLongf compressed_size = sizeof compressed, output_size = sizeof output;
    for (unsigned i = 0; i < sizeof input; ++i) input[i] = i % 7;
    if (compress2(compressed, &compressed_size, input, sizeof input, 6) != Z_OK) return 1;
    if (compressed_size >= sizeof input) return 2;
    if (uncompress(output, &output_size, compressed, compressed_size) != Z_OK) return 3;
    if (output_size != sizeof input || memcmp(input, output, sizeof input)) return 4;
    if (crc32(0, (const Bytef *)"123456789", 9) != 0xcbf43926UL) return 5;
    compressed[0] = 0;
    output_size = sizeof output;
    if (uncompress(output, &output_size, compressed, compressed_size) != Z_DATA_ERROR) return 6;
    return strcmp(zlibVersion(), ZLIB_VERSION) != 0;
}
