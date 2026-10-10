/* Existing author-trained Meiki v0.1 320x192 detector, bounded MNN binding. */
#ifndef MEIKI_DETECT_RUNTIME_H
#define MEIKI_DETECT_RUNTIME_H
#include "runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MEIKI_DETECT_WIDTH 320u
#define MEIKI_DETECT_HEIGHT 192u
#define MEIKI_DETECT_ELEMENTS (3u * MEIKI_DETECT_HEIGHT * MEIKI_DETECT_WIDTH)
#define MEIKI_DETECT_BOXES 64u
typedef struct {
    int32_t labels[MEIKI_DETECT_BOXES];
    float boxes[MEIKI_DETECT_BOXES][4];
    float scores[MEIKI_DETECT_BOXES];
} MeikiDetectOutput;
/* target_width/height are upstream's virtual unpadded original-size targets,
 * int(320/scale), int(192/scale), not the resized content dimensions. Output
 * remains raw: caller applies score>0.5, clamps/truncates coordinates to its
 * original region, then sorts boxes by y. Uses the same16MiB arena as meiki_run;
 * all detector objects unwind before returning so recognition can reuse it.
 */
int meiki_detect_run(const char *model_path, const float *input,
                     int32_t target_width, int32_t target_height,
                     void *workspace, size_t workspace_bytes,
                     MeikiDetectOutput *output, MeikiStats *stats);
#ifdef __cplusplus
}
#endif
#endif
