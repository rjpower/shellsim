#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare(const void *left, const void *right) {
    int a = *(const int *)left, b = *(const int *)right;
    return (a > b) - (a < b);
}

static void *worker(void *unused) {
    (void)unused;
    errno = 29;
    return NULL;
}

int main(void) {
    int values[] = {42, 7, 20};
    qsort(values, 3, sizeof(int), compare);
    if (values[0] != 7 || values[1] != 20 || values[2] != 42) return 1;
    char *text = malloc(32);
    if (text == NULL) return 2;
    strcpy(text, "installed guest SDK");
    if (strlen(text) != 19) return 3;
    free(text);
    pthread_t thread;
    errno = 17;
    if (pthread_create(&thread, NULL, worker, NULL)) return 4;
    if (pthread_join(thread, NULL)) return 5;
    if (errno != 17) return 6;
    puts("installed guest SDK libc passed");
    return 0;
}
