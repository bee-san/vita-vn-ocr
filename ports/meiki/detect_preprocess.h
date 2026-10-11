/* Allocation-free RGBA binding to upstream Meiki detector preprocessing. */
#ifndef MEIKI_DETECT_PREPROCESS_H
#define MEIKI_DETECT_PREPROCESS_H
#include "preprocess.h"
#include "detect_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MEIKI_DETECT_SCRATCH_BYTES 9600u
typedef struct {
    unsigned effective_w, effective_h;
    int32_t target_width, target_height;
} MeikiDetectPreprocessInfo;
/* BGR float32 NCHW320x192, uint8 INTER_LINEAR resize, zero letterbox padding.
 * Output737280 bytes aligned4; scratch9600 bytes aligned2, disjoint. Reuses
 * MeikiRgbaImage row/cancel callbacks and limits960x544. Input can subsequently
 * be reused for the smaller horizontal recognizer tensor. Failure leaves
 * partial output and zero info; it must not be submitted to the detector.
 */
int meiki_detect_preprocess_rgba(const MeikiRgbaImage *image,
                                const MeikiCrop *crop, float *output,
                                size_t output_elements, void *scratch,
                                size_t scratch_bytes,
                                MeikiDetectPreprocessInfo *info);
#ifdef __cplusplus
}
#endif
#endif
