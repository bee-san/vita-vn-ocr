/* screen.h - text output on the PS Vita framebuffer for vocr_test.
 *
 * One 960x544 A8B8G8R8 framebuffer in CDRAM. Text is drawn with font.bin, a 16-px
 * bitmap font that vita/bundle.py rasterises from the OFL font M+ 1p (ASCII half width,
 * everything else full width), so decoded Japanese can be read on the console. Without
 * a font only the coloured status bar is drawn. Not part of the runtime.
 */
#ifndef VOCR_SCREEN_H
#define VOCR_SCREEN_H

#include <stddef.h>
#include <stdint.h>

#define SCR_W 960
#define SCR_H 544
#define SCR_ROW 16 /* text row height in pixels */

/* A8B8G8R8 colours */
#define SCR_BLACK 0xFF000000u
#define SCR_WHITE 0xFFFFFFFFu
#define SCR_GREY 0xFFA0A0A0u
#define SCR_GREEN 0xFF60E060u
#define SCR_RED 0xFF5050FFu
#define SCR_YELLOW 0xFF40E0FFu
#define SCR_CYAN 0xFFFFE060u

int scr_init(void);                           /* 0, or the memblock / display error */
int scr_load_font(const void *data, size_t n); /* 0 if font.bin is valid; data must stay alive */
int scr_has_font(void);
void scr_fill(int x, int y, int w, int h, uint32_t color);
/* Draws UTF-8 text at pixel (x, y) clipped to max_px pixels; returns the width drawn. */
int scr_text(int x, int y, uint32_t color, const char *utf8, int max_px);
/* Fixed rows 0 .. SCR_HEAD_ROWS-1 at the top and the status bar at the bottom. */
#define SCR_HEAD_ROWS 4
void scr_head(int row, uint32_t color, const char *utf8);
void scr_status(uint32_t color, const char *utf8, unsigned done, unsigned total);
/* Scrolling log between the head and the status bar. */
void scr_log(uint32_t color, const char *utf8);

#endif
