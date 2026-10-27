/* vocr_lines_fix.h - the dialogue line finder of vocr_lines.h with the fixes of vita-vn-ocr #129 (#119).
 *
 * The same finders (VOCR_LINES_GRAD, _LEGACY, _TEXTBOX, _TEXTBOX_RECOVERY), arena (vocr_lines_arena_bytes) and
 * return codes as vocr_find_lines() / vocr_find_lines_rows(), plus `fix`, a set of VOCR_FIX_* flags for the GRAD
 * finder and the textbox modes built on it (the legacy finder ignores it). fix = 0 gives the frozen finder's lines,
 * dark flags and return codes bit for bit. Rubi calls the row form with VOCR_FIX_DEFAULT (or the flags its user
 * chose).
 *
 * C99, no allocation, no recursion; vocr_lines_fix.c is vocr_lines_rows.c with the fixes (see its header).
 * docs/finder-fix.md has the method, the measurements on the hooked pages and the development sets, and the cost.
 */
#ifndef VOCR_LINES_FIX_H
#define VOCR_LINES_FIX_H

#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_FIX_COLGAP 0x1u /* a line's parts across a full-width space (gaps up to 4 band heights) stay one line */
#define VOCR_FIX_VALLEY 0x2u /* a band of two touching text rows (outlines, tight leading) is split in two */
#define VOCR_FIX_JUNK 0x4u   /* frame strips on the region's top/bottom edge and repeating ornaments are dropped; it
                              * can drop a text line that reaches the edge of a region drawn tight: opt-in */
#define VOCR_FIX_ALL (VOCR_FIX_COLGAP | VOCR_FIX_VALLEY | VOCR_FIX_JUNK)
#define VOCR_FIX_DEFAULT VOCR_FIX_COLGAP /* the measured default (#129, amendment 3); JUNK and VALLEY are opt-in */

/* vocr_find_lines() with the fixes in `fix` (VOCR_LINES_E_ARG for an unknown flag). */
int vocr_find_lines_fix(const VocrLinesImage *im, int finder, unsigned fix, void *arena, size_t arena_bytes,
                        VocrLineSet *out);

/* vocr_find_lines_rows() with the fixes in `fix`: the same lines as vocr_find_lines_fix() for the same pixels, or
 * VOCR_LINES_E_SOURCE (and no lines) when a row could not be read. */
int vocr_find_lines_fix_rows(const VocrLinesRows *src, int finder, unsigned fix, void *arena, size_t arena_bytes,
                             VocrLineSet *out);

#ifdef __cplusplus
}
#endif

#endif
