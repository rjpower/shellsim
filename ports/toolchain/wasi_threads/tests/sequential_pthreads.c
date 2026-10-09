/* Hold fifteen real workers alive, then reuse their slots for forty joins. */
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int ready;
static int released;
static _Thread_local int local_value = 5;

static void *held_worker(void *argument) {
    assert(local_value == 5);
    local_value = (int)(intptr_t)argument;
    assert(pthread_mutex_lock(&mutex) == 0);
    ready++;
    assert(pthread_cond_broadcast(&condition) == 0);
    while (!released)
        assert(pthread_cond_wait(&condition, &mutex) == 0);
    assert(pthread_mutex_unlock(&mutex) == 0);
    return argument;
}

static void *sequential_worker(void *argument) {
    assert(local_value == 5);
    local_value = (int)(intptr_t)argument;
    return (void *)(intptr_t)local_value;
}

int main(void) {
    pthread_attr_t attributes;
    assert(pthread_attr_init(&attributes) == 0);
    assert(pthread_attr_setstacksize(&attributes, 65536) == 0);
    pthread_t workers[15], excess;
    local_value = 99;
    for (int index = 0; index < 15; index++)
        assert(pthread_create(&workers[index], &attributes, held_worker, (void *)(intptr_t)(index + 1)) == 0);
    assert(pthread_mutex_lock(&mutex) == 0);
    while (ready != 15)
        assert(pthread_cond_wait(&condition, &mutex) == 0);
    assert(pthread_create(&excess, &attributes, held_worker, NULL) == EAGAIN);
    released = 1;
    assert(pthread_cond_broadcast(&condition) == 0);
    assert(pthread_mutex_unlock(&mutex) == 0);
    for (int index = 0; index < 15; index++) {
        void *result;
        assert(pthread_join(workers[index], &result) == 0);
        assert((intptr_t)result == index + 1);
    }
    for (int index = 0; index < 40; index++) {
        pthread_t thread;
        void *result;
        assert(pthread_create(&thread, &attributes, sequential_worker, (void *)(intptr_t)(index + 17)) == 0);
        assert(pthread_join(thread, &result) == 0);
        assert((intptr_t)result == index + 17);
    }
    assert(local_value == 99);
    assert(pthread_attr_destroy(&attributes) == 0);
    puts("pthread slots: live cap and 40 sequential joins passed");
    return 0;
}
