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
#define MEIKI_MODEL_BYTES 5392856u
#define MEIKI_MODEL_READ_CHUNK 8192u
/* Exact, bounded reads of the pinned recognizer. read must fill every requested
 * byte and return 0, or return a negative I/O/cancellation error. No destination
 * or context pointer may be retained after read/run returns. The descriptor and
 * its caller-owned context stay outside workspace, metadata, input and output.
 */
typedef struct {
    uint32_t size, byte_count;
    void *context;
    int (*read)(void *context, uint32_t offset, void *destination, uint32_t bytes);
} MeikiModelReader;
typedef struct {
    int32_t codes[48];
    float boxes[48 * 4], scores[48];
} MeikiOutput;
typedef struct {
    size_t heap_peak, heap_remaining;
    float session_memory_mib;
    uint64_t load_us, inference_us;
    int allocation_failed;
    size_t model_read_bytes;
    int model_read_error;
} MeikiStats;
/* Single caller. The input is upstream Meiki's float32 BGR NCHW tensor.
 * Destroy all interpreter/session objects before returning the workspace.
 * Returns 0 on success; negative values distinguish loading, shape and OOM.
 */
int meiki_run(const char *model_path, const float *input, void *workspace,
              size_t workspace_bytes, MeikiOutput *output, MeikiStats *stats);
/* Reader path never reopens model_path. Both the temporary input bytes and
 * MNN's owning copy use the same bounded workspace. The temporary is freed
 * before session creation. Returns -14 for a read failure; stats preserves its
 * original code. Calls are synchronous and retain no reader pointers.
 */
int meiki_run_reader(const MeikiModelReader *reader, const float *input,
                     void *workspace, size_t workspace_bytes,
                     MeikiOutput *output, MeikiStats *stats);
#ifdef __cplusplus
}
#endif
#endif
