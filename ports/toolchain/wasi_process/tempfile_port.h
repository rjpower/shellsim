/* SDK 34 hides this declaration on Preview 1 because its archive omits it. */
#ifndef SHELLSIM_WASI_TEMPFILE_PORT_H
#define SHELLSIM_WASI_TEMPFILE_PORT_H

#ifdef __cplusplus
extern "C" {
#endif
int mkstemp(char *template);
#ifdef __cplusplus
}
#endif
#endif
