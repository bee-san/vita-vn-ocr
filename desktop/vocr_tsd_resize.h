/* vocr_tsd_resize.h - textspot-desktop's frame luma and integer resize (issue #173): desktop/vocr_desk.c's rules
 * (vocr_desk_grey, vocr_desk_resize) and tools/tsdesk.resize's, bit for bit, with SIMD paths and row bands for
 * threads (vocr_tsd_resize.c). vocr_tsd.c reads every frame through these; vocr_desk.* is vocr-desktop's and is not
 * changed. vocr_tsd_input_size (the input policy) is declared in vocr_tsd.h and defined in vocr_tsd_resize.c. */
#ifndef VOCR_TSD_RESIZE_H
#define VOCR_TSD_RESIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Integer luma of an RGB (channels 3) or RGBA (4) frame: (77 R + 150 G + 29 B) >> 8 (vocr_desk_grey's, the
 * recognizer's), w x h pixels, rows `stride` bytes apart, into out (w x h, rows w apart). */
void vocr_tsd_grey(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                   unsigned char *out);

/* vocr_desk_resize's rule (Q14 area weights when a side shrinks, bilinear on pixel centres when it grows, separable,
 * integer): a w x h grey image (rows `stride` bytes apart) to dw x dh (rows dw apart); scratch: an int-aligned buffer
 * of vocr_tsd_resize_scratch(w, h, dw, dh) bytes. */
void vocr_tsd_resize(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned char *dst,
                     unsigned dw, unsigned dh, int *scratch);
size_t vocr_tsd_resize_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh);

/* The same resize of an RGB(A) frame's integer luma, the luma taken row by row inside the resize: the bits of
 * vocr_tsd_grey then vocr_tsd_resize without the whole grey frame (scratch: vocr_tsd_resize_rgb_scratch bytes). */
void vocr_tsd_resize_rgb(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                         unsigned char *dst, unsigned dw, unsigned dh, int *scratch);
size_t vocr_tsd_resize_rgb_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh);

/* The same resize in bands of output rows (for threads): vocr_tsd_resize_plan fills `plan`
 * (vocr_tsd_resize_plan_bytes, 64-byte aligned) once and returns 0, or -1 when only the whole-frame form applies (a
 * shrink by more than 62: call vocr_tsd_resize(_rgb)); vocr_tsd_resize_band then makes output rows [y0, y1) of a grey
 * (channels 1) or RGB(A) (3, 4) source with `work` (vocr_tsd_resize_band_bytes; one per concurrent band). Any split
 * of the rows gives the bits of vocr_tsd_resize (grey) or vocr_tsd_resize_rgb. */
size_t vocr_tsd_resize_plan_bytes(unsigned w, unsigned h, unsigned dw, unsigned dh);
int vocr_tsd_resize_plan(unsigned w, unsigned h, unsigned dw, unsigned dh, void *plan);
size_t vocr_tsd_resize_band_bytes(unsigned w, unsigned h, unsigned dw, unsigned dh);
void vocr_tsd_resize_band(const void *plan, const unsigned char *src, unsigned stride, unsigned channels,
                          unsigned char *dst, unsigned y0, unsigned y1, void *work);

#ifdef __cplusplus
}
#endif

#endif
