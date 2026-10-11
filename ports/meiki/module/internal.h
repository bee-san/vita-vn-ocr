#ifndef MEIKI_MODULE_INTERNAL_H
#define MEIKI_MODULE_INTERNAL_H
#include "api.h"
#ifdef __cplusplus
extern "C" {
#endif
void meiki_metadata_prepare(void *base, size_t bytes, int engine_thread);
uint32_t meiki_metadata_brk_peak(void);
uint32_t meiki_metadata_tainted(void);
int meiki_engine_initialize(void *metadata, size_t bytes, MeikiModuleStats *stats);
int meiki_engine_run(const MeikiModuleRequest *request, MeikiModuleStats *stats);
int meiki_engine_detect(const MeikiModuleDetectRequest *request, MeikiModuleStats *stats);
int meiki_engine_shutdown(MeikiModuleStats *stats);
#ifdef __cplusplus
}
#endif
#endif
