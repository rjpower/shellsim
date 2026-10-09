/* Real pthread/TLS synchronization probe. Every waiter resumes independently. */
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int ready;
static int counter;
static _Thread_local int local_value = 5;
static int initialized_data = 11;
static int constructor_runs;

__attribute__((constructor)) static void initialize_process(void) {
    constructor_runs++;
}

static void *consumer(void *argument) {
    assert(local_value == 5 && initialized_data == 37 && constructor_runs == 1);
    local_value = (int)(intptr_t)argument;
    assert(pthread_mutex_lock(&mutex) == 0);
    while (!ready)
        assert(pthread_cond_wait(&condition, &mutex) == 0);
    counter += local_value;
    assert(pthread_mutex_unlock(&mutex) == 0);
    return (void *)(intptr_t)local_value;
}

static void *producer(void *argument) {
    assert(local_value == 5 && initialized_data == 37 && constructor_runs == 1);
    local_value = (int)(intptr_t)argument;
    assert(pthread_mutex_lock(&mutex) == 0);
    counter += local_value;
    ready = 1;
    assert(pthread_cond_signal(&condition) == 0);
    assert(pthread_mutex_unlock(&mutex) == 0);
    return (void *)(intptr_t)local_value;
}

int main(void) {
    pthread_t first, second;
    void *first_result, *second_result;
    assert(constructor_runs == 1 && initialized_data == 11);
    initialized_data = 37;
    local_value = 99;
    assert(pthread_create(&first, NULL, consumer, (void *)(intptr_t)17) == 0);
    assert(pthread_create(&second, NULL, producer, (void *)(intptr_t)23) == 0);
    assert(pthread_join(first, &first_result) == 0);
    assert(pthread_join(second, &second_result) == 0);
    assert((intptr_t)first_result == 17 && (intptr_t)second_result == 23);
    assert(counter == 40 && local_value == 99 && initialized_data == 37 && constructor_runs == 1);

    struct timespec deadline;
    assert(clock_gettime(CLOCK_REALTIME, &deadline) == 0);
    deadline.tv_nsec += 5000000;
    if (deadline.tv_nsec >= 1000000000) {
        deadline.tv_nsec -= 1000000000;
        deadline.tv_sec++;
    }
    assert(pthread_mutex_lock(&mutex) == 0);
    assert(pthread_cond_timedwait(&condition, &mutex, &deadline) == ETIMEDOUT);
    assert(pthread_mutex_unlock(&mutex) == 0);
    puts("two pthreads: join, mutex, condition, TLS, virtual timeout passed");
    return 0;
}
