/* Exercise the upstream libjpeg API. Invalid data uses its fatal error handler;
 * recovering in the same process requires setjmp, absent from WASI SDK 24. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

int main(int argc, char **argv) {
    struct jpeg_compress_struct encoder;
    struct jpeg_decompress_struct decoder;
    struct jpeg_error_mgr encode_error, decode_error;
    unsigned char *encoded = NULL;
    unsigned long length = 0;
    unsigned char row[6] = {240, 20, 30, 240, 20, 30};
    JSAMPROW scanline = row;
    decoder.err = jpeg_std_error(&decode_error);
    jpeg_create_decompress(&decoder);
    if (argc > 1 && strcmp(argv[1], "invalid") == 0) {
        const unsigned char invalid[] = {0, 1, 2, 3};
        jpeg_mem_src(&decoder, invalid, sizeof(invalid));
        jpeg_read_header(&decoder, TRUE);
        return 99;
    }
    encoder.err = jpeg_std_error(&encode_error);
    jpeg_create_compress(&encoder);
    jpeg_mem_dest(&encoder, &encoded, &length);
    encoder.image_width = 2;
    encoder.image_height = 1;
    encoder.input_components = 3;
    encoder.in_color_space = JCS_RGB;
    jpeg_set_defaults(&encoder);
    jpeg_set_quality(&encoder, 95, TRUE);
    jpeg_start_compress(&encoder, TRUE);
    jpeg_write_scanlines(&encoder, &scanline, 1);
    jpeg_finish_compress(&encoder);
    jpeg_destroy_compress(&encoder);
    jpeg_mem_src(&decoder, encoded, length);
    jpeg_read_header(&decoder, TRUE);
    jpeg_start_decompress(&decoder);
    if (decoder.output_width != 2 || decoder.output_height != 1 || decoder.output_components != 3) return 2;
    jpeg_read_scanlines(&decoder, &scanline, 1);
    for (int i = 0; i < 6; i++) {
        int expected = (i % 3 == 0) ? 240 : (i % 3 == 1 ? 20 : 30);
        if (abs((int)row[i] - expected) > 3) return 3;
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    free(encoded);
    puts("JPEG round trip: 2x1 RGB within tolerance 3");
    return 0;
}
