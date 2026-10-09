/* cases_dump.c - host driver for tests/test_vita.py: parses a cases.bin with the Vita app's
 * own parser (vita/app/cases.c) and prints what it read, one tab-separated record per line:
 *   header  n_models n_cases max_case_bytes max_lines max_frames n_classes max_rw max_rh
 *   model   name file mode blob_bytes blob_crc host_arena
 *   case    id kind flags w h ch gt pixel_sum
 *   expect  model_index text crc frames n_lines then x y w h dark per line
 * Texts are printed with tab, newline and backslash escaped as \t, \n and \\.
 * Not part of the console build.
 */
#include "../app/cases.h"

#include <stdio.h>
#include <stdlib.h>

static void put_text(const char *s)
{
    for (; *s; s++) {
        if (*s == '\t') fputs("\\t", stdout);
        else if (*s == '\n') fputs("\\n", stdout);
        else if (*s == '\\') fputs("\\\\", stdout);
        else putchar(*s);
    }
}

int main(int argc, char **argv)
{
    unsigned char hdr[VC_HEADER_BYTES], rec[VC_MODEL_BYTES], lb[4], *buf;
    static char strings[VC_STRINGS_BYTES];
    VcHeader h;
    VcCase c;
    unsigned i, m, j;
    FILE *f;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) {
        fprintf(stderr, "usage: cases_dump cases.bin\n");
        return 2;
    }
    if (fread(hdr, 1, sizeof hdr, f) != sizeof hdr || vc_header(hdr, &h)) {
        fprintf(stderr, "bad header\n");
        return 1;
    }
    printf("header\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n", h.n_models, h.n_cases, h.max_case_bytes, h.max_lines,
           h.max_frames, h.n_classes, h.max_rw, h.max_rh);
    for (m = 0; m < h.n_models; m++) {
        VcModel vm;
        if (fread(rec, 1, sizeof rec, f) != sizeof rec) return 1;
        vc_model(rec, &vm);
        printf("model\t%s\t%s\t%u\t%u\t%u\t%u\n", vm.name, vm.file, vm.mode, vm.blob_bytes, vm.blob_crc,
               vm.host_arena);
    }
    buf = (unsigned char *)malloc(h.max_case_bytes);
    for (i = 0; i < h.n_cases; i++) {
        uint32_t len;
        unsigned long sum = 0;
        size_t k;
        if (!buf || fread(lb, 1, 4, f) != 4 || (len = vc_rd32(lb)) < 4 || len > h.max_case_bytes ||
            fread(buf, 1, len - 4, f) != len - 4 || vc_case(buf, len - 4, h.n_models, strings, &c)) {
            fprintf(stderr, "bad record %u\n", i);
            free(buf);
            return 1;
        }
        for (k = 0; k < (size_t)c.w * c.h * c.ch; k++) sum = (sum + c.px[k]) & 0xFFFFFFFFul;
        printf("case\t%s\t%u\t%u\t%u\t%u\t%u\t", c.id, c.kind, c.flags, c.w, c.h, c.ch);
        put_text(c.gt);
        printf("\t%lu\n", sum);
        for (m = 0; m < h.n_models; m++) {
            printf("expect\t%u\t", m);
            put_text(c.exp[m].text);
            printf("\t%u\t%u\t%u", c.exp[m].crc, c.exp[m].frames, c.exp[m].n_lines);
            for (j = 0; j < c.exp[m].n_lines; j++)
                printf("\t%u\t%u\t%u\t%u\t%u", c.exp[m].box[j][0], c.exp[m].box[j][1], c.exp[m].box[j][2],
                       c.exp[m].box[j][3], c.exp[m].box[j][4]);
            printf("\n");
        }
    }
    free(buf);
    fclose(f);
    return 0;
}
