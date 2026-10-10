/* RGBA-row binding for upstream Meiki's horizontal recognition preprocessing. */
#ifndef MEIKI_PREPROCESS_H
#define MEIKI_PREPROCESS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define MEIKI_PREPROCESS_MAX_WIDTH 960u
#define MEIKI_PREPROCESS_MAX_HEIGHT 544u
#define MEIKI_PREPROCESS_ELEMENTS (3u * 32u * 960u)
#define MEIKI_PREPROCESS_SCRATCH_BYTES 13440u

typedef struct {
    void *ud;
    unsigned width, height;
    /* Same callback contract as vita-jp-overlay's VjoOcrImage: packed RGBA,
     * width * 4 bytes per row; return rows copied, or a negative error. */
    int (*rows)(void *ud, unsigned y, unsigned n, unsigned char *rgba);
    int (*cancelled)(void *ud); /* optional, nonzero cancels */
} MeikiRgbaImage;
typedef struct { unsigned x, y, w, h; } MeikiCrop;
typedef struct { unsigned effective_w, effective_h; } MeikiPreprocessInfo;
enum {
    MEIKI_PREPROCESS_OK = 0,
    MEIKI_PREPROCESS_INVALID = -1,
    MEIKI_PREPROCESS_SOURCE = -2,
    MEIKI_PREPROCESS_CANCELLED = -3
};

/* Allocation-free. Caller owns a float32-aligned BGR NCHW output (368640 bytes) and
 * separate scratch (13440 bytes, aligned to at least 2 bytes). Output and
 * scratch must not overlap. RGBA alpha is ignored. A strided framebuffer is
 * supported by copying its rows to the packed callback buffer.
 * On failure, output is partial and must not be submitted for recognition;
 * info is zeroed. Geometry follows Python round (ties to even), OpenCV's
 * uint8 INTER_LINEAR kernel, zero padding and float32 division by 255.
 */
int meiki_preprocess_rgba(const MeikiRgbaImage *image, const MeikiCrop *crop,
                         float *output, size_t output_elements,
                         void *scratch, size_t scratch_bytes,
                         MeikiPreprocessInfo *info);
#ifdef __cplusplus
}
#endif
#endif
