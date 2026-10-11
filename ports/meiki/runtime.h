/* C binding for the existing horizontal Meiki model and MNN 3.2.0. */
#ifndef MEIKI_RUNTIME_H
#define MEIKI_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MEIKI_INPUT_ELEMENTS (3u * 32u * 960u)
#define MEIKI_OUTPUT_CHARACTERS 48
#define MEIKI_WORKSPACE_BYTES (16u * 1024u * 1024u)
typedef struct {
    int32_t codes[48];
    float boxes[48 * 4], scores[48];
} MeikiOutput;
typedef struct {
    size_t heap_peak, heap_remaining;
    float session_memory_mib;
    uint64_t load_us, inference_us;
    int allocation_failed;
} MeikiStats;
/* Single caller. The input is upstream Meiki's float32 BGR NCHW tensor.
 * Destroy all interpreter/session objects before returning the workspace.
 * Returns 0 on success; negative values distinguish loading, shape and OOM.
 */
int meiki_run(const char *model_path, const float *input, void *workspace,
              size_t workspace_bytes, MeikiOutput *output, MeikiStats *stats);
#ifdef __cplusplus
}
#endif
#endif
