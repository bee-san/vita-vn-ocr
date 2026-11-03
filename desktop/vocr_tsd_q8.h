/* vocr_tsd_q8.h - textspot-desktop's int8 network (desktop/vocr_tsd_q8.c), internal to the library: vocr_tsd.c
 * loads a version-2 blob's layers through vocr_tsdq_load and runs the network through vocr_tsdq_run when a model's
 * version is 2. Not a public header (vocr_tsd.h is). */
#ifndef VOCR_TSD_Q8_H
#define VOCR_TSD_Q8_H

#include <stddef.h>

#include "vocr_tsd.h"

/* The layers of a version-2 blob's body (b: the whole blob, its header already read into m); 0 or VOCR_TSD_E_BLOB. */
int vocr_tsdq_load(vocr_tsd_model *m, const unsigned char *b, size_t bytes);
/* Arena bytes of the int8 network on a pw x ph padded input with ntasks tasks. */
size_t vocr_tsdq_maps_bytes(const vocr_tsd_model *m, unsigned pw, unsigned ph, unsigned ntasks);
/* The int8 network on a grey source (w x h, row stride `stride`) read as a pw x ph input with the edge padding (x >= w
 * as w - 1, y >= h as h - 1): *maps points into the arena, float32 [ph / hs][pw / hs][out]. */
int vocr_tsdq_run(const vocr_tsd_model *m, const unsigned char *px, unsigned w, unsigned h, unsigned stride,
                  unsigned pw, unsigned ph, void *arena, size_t bytes, const float **maps, vocr_tsd_runner run,
                  void *ctx, unsigned ntasks);

#ifdef VOCR_TSD_PROF
/* vocr_tsd.c's per-step host timing (profiling builds only): restart the clock; add the time since to slot i; add a
 * measured time to slot i; the clock in us */
void vocr_tsd_prof_start_(void);
void vocr_tsd_prof_mark_(unsigned i, const char *name);
void vocr_tsd_prof_add_(unsigned i, const char *name, double us);
double vocr_tsd_prof_now_(void);
#endif

#endif
