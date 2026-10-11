/* SPDX-License-Identifier: Apache-2.0
 * Horizontal postprocessing ported from rtr46/meikiocr 0.3.5, ocr.py.
 */
#ifndef MEIKI_DECODE_H
#define MEIKI_DECODE_H
#include "runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Matches confidence=.1, punct_conf_factor=1 and horizontal NMS in upstream.
 * effective_width is the unpadded input width from its preprocessing.
 * At most 48 UTF-8 characters (193 bytes including NUL). */
int meiki_decode(const MeikiOutput *raw, unsigned crop_width,
                 unsigned effective_width, char *text, size_t text_bytes);
#ifdef __cplusplus
}
#endif
#endif
