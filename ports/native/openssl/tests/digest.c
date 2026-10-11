/* Exercise upstream crypto operations through the admitted static archive. */
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <string.h>

int main(void) {
    static const unsigned char expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (!SHA256((const unsigned char *)"abc", 3, digest) ||
        memcmp(digest, expected, 32)) return 1;
    if (!EVP_Digest("abc", 3, digest, &size, EVP_sha256(), NULL) ||
        size != 32 || memcmp(digest, expected, 32)) return 2;
    return 0;
}
