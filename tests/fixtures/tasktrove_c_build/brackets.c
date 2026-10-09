/* Candidate for Codeelo brcktsrm; process each bracket string as a stream. */
#include <stdio.h>

int main(void) {
    int cases;
    if (scanf("%d", &cases) != 1 || cases < 1) return 2;
    for (int test = 0; test < cases; ++test) {
        int character;
        do {
            character = getchar();
        } while (character == ' ' || character == '\n' || character == '\r' || character == '\t');
        if (character == EOF) return 2;
        long long depth = 0;
        int balanced = 1;
        while (character != EOF && character != ' ' && character != '\n' && character != '\r' && character != '\t') {
            if (character == '(') ++depth;
            else --depth;
            if (depth < 0) balanced = 0;
            character = getchar();
        }
        puts(balanced && depth == 0 ? "YES" : "NO");
    }
    return 0;
}
