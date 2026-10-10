/* Validate the shared JPEG codec and recover from malformed input in-process. */
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <jpeglib.h>

struct recoverable_error { struct jpeg_error_mgr base; jmp_buf jump; };
static void fail(j_common_ptr codec) {
    struct recoverable_error *error = (struct recoverable_error *)codec->err;
    longjmp(error->jump, 1);
}
static int malformed(void) {
    struct jpeg_decompress_struct decoder;
    struct recoverable_error error;
    const unsigned char invalid[] = {0, 1, 2, 3};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = fail;
    jpeg_create_decompress(&decoder);
    if (setjmp(error.jump)) {
        jpeg_destroy_decompress(&decoder);
        return 0;
    }
    jpeg_mem_src(&decoder, invalid, sizeof invalid);
    jpeg_read_header(&decoder, TRUE);
    jpeg_destroy_decompress(&decoder);
    return 1;
}
static int roundtrip(void) {
    struct jpeg_compress_struct encoder;
    struct jpeg_decompress_struct decoder;
    struct jpeg_error_mgr encode_error, decode_error;
    unsigned char *encoded = NULL;
    unsigned long length = 0;
    unsigned char row[6] = {240, 20, 30, 240, 20, 30};
    JSAMPROW scanline = row;
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
    if (jpeg_write_scanlines(&encoder, &scanline, 1) != 1) return 1;
    jpeg_finish_compress(&encoder);
    jpeg_destroy_compress(&encoder);
    decoder.err = jpeg_std_error(&decode_error);
    jpeg_create_decompress(&decoder);
    jpeg_mem_src(&decoder, encoded, length);
    jpeg_read_header(&decoder, TRUE);
    jpeg_start_decompress(&decoder);
    if (decoder.output_width != 2 || decoder.output_height != 1 || decoder.output_components != 3) return 2;
    if (jpeg_read_scanlines(&decoder, &scanline, 1) != 1) return 3;
    for (int i = 0; i < 6; i++) {
        int expected = (i % 3 == 0) ? 240 : (i % 3 == 1 ? 20 : 30);
        if (abs((int)row[i] - expected) > 3) return 4;
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    free(encoded);
    return 0;
}
int main(void) {
    if (malformed() || roundtrip() || malformed() || roundtrip()) return 1;
    puts("Shared JPEG: RGB roundtrip and malformed-input recovery passed");
    return 0;
}
