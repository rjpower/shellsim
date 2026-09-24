/* Shellsim's own opt-in Doomgeneric platform adapter. The engine and WAD are external inputs. */
#include "doomgeneric.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int shellsim_display_open(unsigned, unsigned, unsigned)
    __asm__("shellsim.display_open");
extern int shellsim_display_present(unsigned, const void *, unsigned, unsigned)
    __asm__("shellsim.display_present");
extern int shellsim_input_poll_key(unsigned, void *)
    __asm__("shellsim.input_poll_key");
extern int shellsim_display_close(unsigned)
    __asm__("shellsim.display_close");

struct key_event {
    unsigned code;
    unsigned pressed;
};

static int display_handle;
static unsigned char frame_rgba[DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4];
/* The opt-in headless probe advances one deterministic 35 Hz tic per clock read. Real
   interactive scheduling belongs in shellsim's virtual clock/process interface. */
static uint32_t probe_ms;

void DG_Init(void) {
    display_handle = shellsim_display_open(DOOMGENERIC_RESX, DOOMGENERIC_RESY, 1);
    if (display_handle <= 0) {
        fprintf(stderr, "virtual display open failed: %d\n", display_handle);
        exit(1);
    }
}

void DG_DrawFrame(void) {
    unsigned bytes = DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4;
    for (unsigned pixel = 0; pixel < DOOMGENERIC_RESX * DOOMGENERIC_RESY; ++pixel) {
        uint32_t color = DG_ScreenBuffer[pixel];
        frame_rgba[pixel * 4] = (unsigned char)(color >> 16);
        frame_rgba[pixel * 4 + 1] = (unsigned char)(color >> 8);
        frame_rgba[pixel * 4 + 2] = (unsigned char)color;
        frame_rgba[pixel * 4 + 3] = 255;
    }
    int result = shellsim_display_present(
        display_handle, frame_rgba, bytes, DOOMGENERIC_RESX * 4);
    if (result != 0) {
        fprintf(stderr, "virtual display present failed: %d\n", result);
        exit(1);
    }
}

void DG_SleepMs(uint32_t ms) {
    probe_ms += ms;
}

uint32_t DG_GetTicksMs(void) {
    probe_ms += 29;
    return probe_ms;
}

int DG_GetKey(int *pressed, unsigned char *key) {
    struct key_event event;
    int result = shellsim_input_poll_key(display_handle, &event);
    if (result == 6) return 0; /* WASI EAGAIN */
    if (result != 0 || event.code > 255) {
        fprintf(stderr, "virtual input failed: %d\n", result);
        exit(1);
    }
    *pressed = event.pressed != 0;
    *key = (unsigned char)event.code;
    return 1;
}

void DG_SetWindowTitle(const char *title) {
    (void)title;
}

int main(int argc, char **argv) {
    doomgeneric_Create(argc, argv);
    for (int frame = 0; frame < 16; ++frame) doomgeneric_Tick();
    return shellsim_display_close(display_handle);
}
