/* A normal C dylink provider; libffi remains the existing upstream backend. */
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>

static _Thread_local int bias;

typedef double (*callback_type)(int, double);

void set_bias(int value) { bias = value; }
int get_bias(void) { return bias; }

double add(int integer, double real) { return integer + real + bias; }

double call_callback(callback_type callback, int integer, double real) {
    errno = 37;
    double result = callback(integer, real);
    return errno == 73 ? result : -999.0;
}

typedef struct {
    callback_type callback;
    int integer;
    double real;
    double result;
} callback_job;

static void *foreign_worker(void *argument) {
    callback_job *job = argument;
    bias = 300;
    job->result = call_callback(job->callback, job->integer, job->real);
    return NULL;
}

double call_from_pthread(callback_type callback, int integer, double real) {
    callback_job job = {callback, integer, real, 0};
    pthread_t thread;
    if (pthread_create(&thread, NULL, foreign_worker, &job) || pthread_join(thread, NULL))
        abort();
    return job.result;
}
