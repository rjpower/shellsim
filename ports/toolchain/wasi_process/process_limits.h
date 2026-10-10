/* Implementation limits shared by virtual spawn and exec. These do not change
 * public action layouts or the headers consumed by native toolchain clients. */
#ifndef SHELLSIM_WASI_PROCESS_LIMITS_H
#define SHELLSIM_WASI_PROCESS_LIMITS_H
#define SHELLSIM_PROCESS_MAX_ARGUMENTS 4096
#define SHELLSIM_PROCESS_MAX_ENVIRONMENT 256
#define SHELLSIM_PROCESS_MAX_STRING_BYTES 131072
#endif
