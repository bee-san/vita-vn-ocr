/* Stroke split of the dialogue line finder's boxes (vocr_lines.h), for text over busy backgrounds.
 *
 * The GRAD finder bands rows by how much local contrast they hold. Over a translucent dialogue box with a busy
 * scene behind it (or text drawn over the scene with no box), the scene fills the rows between two lines with
 * contrast too, so the finder returns one box for both lines, and the refinement (vocr_lines_refine.h), whose ink
 * mask also takes the scene's pixels, cannot split it (issue #98, docs/translucent-box.md). vocr_split_lines()
 * takes the finder's VocrLineSet and, per box, profiles its rows by thin strokes of both polarities: a text row
 * crosses bright strokes and the dark outlines or gaps beside them; the scene seen through the box is weaker and
 * mostly of one kind, and an edge or a broad shape is no stroke at all. A box whose profile holds two or more
 * text bands, with a valley between them that no stroke crosses, becomes one line per band; any other box stays as
 * it is. Lines keep their box's columns and dark flag and stay in order; nothing is dropped.
 *
 * It runs after the finder and before the refinement, and reads only each box's own pixels, so a caller can split
 * each box on the strip it holds (the box made strip-relative), as it refines them. Flags 0 leave the set
 * untouched, so a caller can A/B it bit for bit.
 *
 * C99, no allocation, integer arithmetic: the caller passes an arena of at least vocr_split_arena_bytes() bytes,
 * which never exceeds vocr_lines_arena_bytes() for the same region, so the finder's arena serves it.
 * synth/lines_split.py is a bit-exact Python twin.
 */
#ifndef VOCR_LINES_SPLIT_H
#define VOCR_LINES_SPLIT_H

#include <stddef.h>

#include "vocr_lines.h"
#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_SPLIT_STROKE 0x01u
#define VOCR_SPLIT_ALL 0x01u
#define VOCR_SPLIT_DEFAULT VOCR_SPLIT_STROKE

/* Arena bytes vocr_split_lines needs for any region up to max_width x max_height (0 if either is 0); never more
 * than vocr_lines_arena_bytes(max_width, max_height). */
size_t vocr_split_arena_bytes(unsigned max_width, unsigned max_height);

/* Splits set (the finder's output for im) in place. Returns VOCR_LINES_OK, VOCR_LINES_E_ARG (bad image, flags or
 * a line outside the image) or VOCR_LINES_E_ARENA; on an error the set is unchanged. rows_read, if not NULL,
 * receives the number of image rows read (each box of 16 rows or more is read twice, top to bottom). */
int vocr_split_lines(const VocrLinesImage *im, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *set,
                     unsigned long *rows_read);

/* The same through a row callback (vocr_lines_rows.h): each box's rows are asked in increasing order, twice,
 * boxes in set order. VOCR_LINES_E_SOURCE when a row could not be read (the set is then unchanged). */
int vocr_split_lines_rows(const VocrLinesRows *src, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *set,
                          unsigned long *rows_read);

#ifdef __cplusplus
}
#endif

#endif
