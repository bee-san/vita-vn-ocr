/* Bounded exact-read adapter, independent of MNN and console I/O. */
#ifndef MEIKI_MODEL_READER_H
#define MEIKI_MODEL_READER_H
#include "runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
int meiki_model_reader_valid(const MeikiModelReader *reader);
int meiki_model_read(const MeikiModelReader *reader, void *destination,
                     size_t capacity, MeikiStats *stats);
#ifdef __cplusplus
}
#endif
#endif
