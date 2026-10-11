/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef MEIKI_ARENA_H
#define MEIKI_ARENA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int meiki_arena_begin(void *memory, size_t bytes);
void *meiki_arena_alloc(size_t bytes);
void meiki_arena_free(void *pointer);
size_t meiki_arena_used(void);
size_t meiki_arena_peak(void);
int meiki_arena_failed(void);
void meiki_arena_end(void);
#ifdef __cplusplus
}
#endif
#endif
