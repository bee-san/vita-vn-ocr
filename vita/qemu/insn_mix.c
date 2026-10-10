/* insn_mix.c - qemu TCG plugin: how often each guest instruction executed, and how often
 * execution switched between the Advanced SIMD (NEON) and VFP units.
 *
 *   qemu-arm -plugin libinsn_mix.so,out=FILE[,classes=MAP][,lo=HEX,hi=HEX] PROGRAM ARGS...
 *
 * At exit, FILE gets one line "address count" (hex, decimal) per executed instruction
 * whose address is in [lo, hi) (default: all). vita/a9_cycles.py joins the counts with
 * objdump's disassembly of PROGRAM, which gives the exact dynamic instruction mix.
 *
 * MAP (optional) lists "address N" for Advanced SIMD (NEON), "address V" for VFP and
 * "address L" for extension-register loads and stores (VLDR, VSTR, VLDM, VSTM, VPUSH, VPOP,
 * which both instruction sets share), from the same disassembly. The Cortex-A9 cannot run
 * the NEON and VFP units at once: every instruction of one completes before the next one of
 * the other issues (DDI 0409I, 3.4.10). The plugin counts those switches along the executed
 * instruction stream, integer instructions in between ignored, twice: with L instructions
 * ignored ("# switches_l_free N") and with L counted as VFP ("# switches_l_vfp N").
 *
 * Counting is per translation block (one callback per executed block, every instruction of
 * the block counted once), so it assumes blocks run to their end, which holds for a
 * single-threaded user-mode program between system calls.
 * Built by vita/scripts/build_qemu_plugins.sh; host-only, never part of the console build.
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

typedef struct {
    uint64_t count;    /* executions of this block */
    uint32_t switches[2];  /* NEON <-> VFP switches inside the block: L free, L as VFP */
    char first[2], last[2]; /* first and last unit class of the block per rule (0: none) */
    size_t n;          /* instructions of the block inside [lo, hi) */
    uint64_t addr[];   /* their addresses */
} Block;

static GPtrArray *blocks;
static GHashTable *classes; /* address -> 'N' or 'V' */
static GMutex lock;
static uint64_t lo, hi = UINT64_MAX, switches[2];
static char prev_class[2];
static const char *out_path = "insn_mix.txt";

static void tb_exec(unsigned int vcpu, void *udata)
{
    Block *b = (Block *)udata;
    int r;
    (void)vcpu;
    b->count++;
    for (r = 0; r < 2; r++) {
        if (!b->first[r]) continue;
        if (prev_class[r] && prev_class[r] != b->first[r]) switches[r]++;
        switches[r] += b->switches[r];
        prev_class[r] = b->last[r];
    }
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t i, n = qemu_plugin_tb_n_insns(tb), k = 0;
    Block *b = g_malloc0(sizeof *b + n * sizeof b->addr[0]);
    (void)id;
    for (i = 0; i < n; i++) {
        uint64_t va = qemu_plugin_insn_vaddr(qemu_plugin_tb_get_insn(tb, i));
        if (classes) {
            gpointer c = g_hash_table_lookup(classes, &va);
            char cls = c ? (char)GPOINTER_TO_INT(c) : 0;
            int r;
            for (r = 0; r < 2; r++) {
                char u = cls == 'L' ? (r ? 'V' : 0) : cls; /* rule 0: L free; rule 1: L is VFP */
                if (!u) continue;
                if (!b->first[r]) b->first[r] = u;
                else if (b->last[r] != u) b->switches[r]++;
                b->last[r] = u;
            }
        }
        if (va >= lo && va < hi) b->addr[k++] = va;
    }
    b->n = k;
    if (!k && !b->first[0] && !b->first[1]) {
        g_free(b);
        return;
    }
    g_mutex_lock(&lock);
    g_ptr_array_add(blocks, b);
    g_mutex_unlock(&lock);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, b);
}

static gint cmp_addr(gconstpointer a, gconstpointer b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void at_exit(qemu_plugin_id_t id, void *udata)
{
    GHashTable *counts = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    GList *keys, *l;
    FILE *f;
    guint i;
    size_t j;
    (void)id;
    (void)udata;
    for (i = 0; i < blocks->len; i++) {
        Block *b = g_ptr_array_index(blocks, i);
        if (!b->count) continue;
        for (j = 0; j < b->n; j++) {
            uint64_t *c = g_hash_table_lookup(counts, &b->addr[j]);
            if (!c) {
                uint64_t *k = g_new(uint64_t, 1);
                *k = b->addr[j];
                c = g_new0(uint64_t, 1);
                g_hash_table_insert(counts, k, c);
            }
            *c += b->count;
        }
    }
    f = fopen(out_path, "w");
    if (!f) {
        qemu_plugin_outs("insn_mix: cannot write the output file\n");
        return;
    }
    keys = g_list_sort(g_hash_table_get_keys(counts), cmp_addr);
    for (l = keys; l; l = l->next)
        fprintf(f, "%" PRIx64 " %" PRIu64 "\n", *(uint64_t *)l->data,
                *(uint64_t *)g_hash_table_lookup(counts, l->data));
    if (classes)
        fprintf(f, "# switches_l_free %" PRIu64 "\n# switches_l_vfp %" PRIu64 "\n", switches[0], switches[1]);
    fclose(f);
    g_list_free(keys);
    g_hash_table_destroy(counts);
}

static int load_classes(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[64];
    if (!f) return 1;
    classes = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, NULL);
    while (fgets(line, sizeof line, f)) {
        char *end;
        uint64_t *k = g_new(uint64_t, 1);
        *k = strtoull(line, &end, 16);
        while (*end == ' ') end++;
        if (*end != 'N' && *end != 'V' && *end != 'L') {
            g_free(k);
            continue;
        }
        g_hash_table_insert(classes, k, GINT_TO_POINTER((int)*end));
    }
    fclose(f);
    return 0;
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                                           int argc, char **argv)
{
    int i;
    (void)info;
    for (i = 0; i < argc; i++) {
        char *v = strchr(argv[i], '=');
        if (!v) return 1;
        *v++ = 0;
        if (!strcmp(argv[i], "out")) out_path = v;
        else if (!strcmp(argv[i], "lo")) lo = strtoull(v, NULL, 16);
        else if (!strcmp(argv[i], "hi")) hi = strtoull(v, NULL, 16);
        else if (!strcmp(argv[i], "classes")) {
            if (load_classes(v)) return 1;
        } else return 1;
    }
    blocks = g_ptr_array_new_with_free_func(g_free);
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
