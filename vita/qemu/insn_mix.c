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
 *
 * Per-layer attribution (vita/a9_layers.py), all optional:
 *   marker=HEX   the entry address of a function the guest calls with a phase id in r0
 *                (vita/qemu/prof_mark.c); every instruction executed after a call, up to the
 *                next one, is counted for that id (0-127; 0 before the first call)
 *   layers=FILE  at exit: "id address count" per executed instruction and id, then per id
 *                "# id switches_l_free N", "# id switches_l_vfp N" and, with l1d=on,
 *                "# id data_accesses N", "# id l1d_misses N", "# id l2_misses N"
 *   l1d=on       a data cache model per access: 32 KiB 4-way L1 and 2 MiB 8-way L2, 32-byte
 *                lines, LRU, by the access's start address, as contrib/plugins/cache.c with
 *                vita/a9_cycles.py's CACHE_ARGS (so the totals match its numbers closely)
 *   pld=FILE     with l1d=on: "address reg offset" per PLD instruction (reg 0-14 = r0-r12, sp,
 *                lr): qemu runs PLD as a no-op, so the model reads the base register and fills
 *                the line into both caches as the A9's PLD does, without a demand access; the
 *                id's "# id prefetches N" and "# id prefetch_fills N" (lines not yet in L1)
 * FILE (out=) is unchanged by these options.
 * Built by vita/scripts/build_qemu_plugins.sh; host-only, never part of the console build.
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define N_IDS 128u

typedef struct {
    uint64_t count;    /* executions of this block */
    uint64_t *by_id;   /* with layers=: executions per phase id (N_IDS), else NULL */
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

/* per-layer attribution */
static const char *layers_path;
static uint64_t marker = UINT64_MAX;
static unsigned cur_id;
static uint64_t id_switches[N_IDS][2];
static struct qemu_plugin_register *r0_handle;
static GByteArray *r0_buf;

/* the data cache model (l1d=on): sets x ways of line tags, LRU by a use counter */
typedef struct {
    unsigned sets, ways;
    uint64_t *tag, *used, tick;
} Cache;
static int cache_on;
static Cache l1d, l2;
static uint64_t id_access[N_IDS], id_l1m[N_IDS], id_l2m[N_IDS], id_pld[N_IDS], id_pld_fill[N_IDS];
static GHashTable *plds;   /* address -> (reg << 16 | (offset & 0xFFFF)) + 1 */
static struct qemu_plugin_register *core_regs[15];
static const char *const core_names[15] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9",
                                           "r10", "r11", "r12", "sp", "lr"};

static void cache_init(Cache *c, unsigned bytes, unsigned ways)
{
    c->ways = ways;
    c->sets = bytes / 32u / ways;
    c->tag = g_new0(uint64_t, (size_t)c->sets * ways);
    c->used = g_new0(uint64_t, (size_t)c->sets * ways);
}

/* 1 on a hit; on a miss the least recently used way gets the line */
static int cache_touch(Cache *c, uint64_t line)
{
    unsigned s = (unsigned)(line % c->sets), w, lru = 0;
    uint64_t *t = c->tag + (size_t)s * c->ways, *u = c->used + (size_t)s * c->ways;
    c->tick++;
    for (w = 0; w < c->ways; w++)
        if (u[w] && t[w] == line) {
            u[w] = c->tick;
            return 1;
        }
    for (w = 1; w < c->ways; w++)
        if (u[w] < u[lru]) lru = w;
    t[lru] = line;
    u[lru] = c->tick;
    return 0;
}

static void mem_access(unsigned int vcpu, qemu_plugin_meminfo_t info, uint64_t vaddr, void *udata)
{
    uint64_t line = vaddr >> 5;
    (void)vcpu;
    (void)info;
    (void)udata;
    id_access[cur_id]++;
    if (cache_touch(&l1d, line)) return;
    id_l1m[cur_id]++;
    if (!cache_touch(&l2, line)) id_l2m[cur_id]++;
}

static void vcpu_init(qemu_plugin_id_t id, unsigned int vcpu)
{
    GArray *regs = qemu_plugin_get_registers();
    guint i;
    (void)id;
    (void)vcpu;
    for (i = 0; regs && i < regs->len; i++) {
        qemu_plugin_reg_descriptor *d = &g_array_index(regs, qemu_plugin_reg_descriptor, i);
        unsigned r;
        if (!strcmp(d->name, "r0")) r0_handle = d->handle;
        for (r = 0; r < 15; r++)
            if (!strcmp(d->name, core_names[r])) core_regs[r] = d->handle;
    }
    if (regs) g_array_free(regs, TRUE);
}

/* the marker function's first block: its r0 is the phase id */
static void marker_exec(unsigned int vcpu, void *udata)
{
    (void)vcpu;
    (void)udata;
    if (!r0_handle) return;
    g_byte_array_set_size(r0_buf, 0);
    if (qemu_plugin_read_register(r0_handle, r0_buf) >= 4) {
        uint32_t v;
        memcpy(&v, r0_buf->data, 4); /* little-endian guest and host */
        cur_id = v < N_IDS ? v : N_IDS - 1u;
    }
}

/* a PLD: the line of base register + offset into L1 and L2, as a fill, not an access */
static void pld_exec(unsigned int vcpu, void *udata)
{
    const uintptr_t e = (uintptr_t)udata - 1u;
    const unsigned reg = (unsigned)(e >> 16);
    const int32_t off = (int16_t)(e & 0xFFFFu);
    uint32_t base;
    (void)vcpu;
    if (reg >= 15 || !core_regs[reg]) return;
    g_byte_array_set_size(r0_buf, 0);
    if (qemu_plugin_read_register(core_regs[reg], r0_buf) < 4) return;
    memcpy(&base, r0_buf->data, 4);
    id_pld[cur_id]++;
    if (!cache_touch(&l1d, (uint64_t)(uint32_t)(base + (uint32_t)off) >> 5)) {
        id_pld_fill[cur_id]++;
        (void)cache_touch(&l2, (uint64_t)(uint32_t)(base + (uint32_t)off) >> 5);
    }
}

static void tb_exec(unsigned int vcpu, void *udata)
{
    Block *b = (Block *)udata;
    int r;
    (void)vcpu;
    b->count++;
    if (b->by_id) b->by_id[cur_id]++;
    for (r = 0; r < 2; r++) {
        uint32_t sw = 0;
        if (!b->first[r]) continue;
        if (prev_class[r] && prev_class[r] != b->first[r]) sw++;
        sw += b->switches[r];
        switches[r] += sw;
        id_switches[cur_id][r] += sw;
        prev_class[r] = b->last[r];
    }
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t i, n = qemu_plugin_tb_n_insns(tb), k = 0;
    Block *b = g_malloc0(sizeof *b + n * sizeof b->addr[0]);
    (void)id;
    if (n && qemu_plugin_insn_vaddr(qemu_plugin_tb_get_insn(tb, 0)) == marker)
        qemu_plugin_register_vcpu_tb_exec_cb(tb, marker_exec, QEMU_PLUGIN_CB_R_REGS, NULL);
    for (i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t va = qemu_plugin_insn_vaddr(insn);
        if (cache_on) qemu_plugin_register_vcpu_mem_cb(insn, mem_access, QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, NULL);
        if (cache_on && plds) {
            gpointer e = g_hash_table_lookup(plds, &va);
            if (e) qemu_plugin_register_vcpu_insn_exec_cb(insn, pld_exec, QEMU_PLUGIN_CB_R_REGS, e);
        }
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
    if (layers_path) b->by_id = g_new0(uint64_t, N_IDS);
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

/* layers=FILE: "id address count" per executed instruction of each id, then the per-id totals */
static void write_layers(void)
{
    FILE *f = fopen(layers_path, "w");
    unsigned id;
    guint i;
    size_t j;
    if (!f) {
        qemu_plugin_outs("insn_mix: cannot write the layers file\n");
        return;
    }
    for (id = 0; id < N_IDS; id++) {
        GHashTable *counts = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
        GList *keys, *l;
        for (i = 0; i < blocks->len; i++) {
            Block *b = g_ptr_array_index(blocks, i);
            if (!b->by_id || !b->by_id[id]) continue;
            for (j = 0; j < b->n; j++) {
                uint64_t *c = g_hash_table_lookup(counts, &b->addr[j]);
                if (!c) {
                    uint64_t *key = g_new(uint64_t, 1);
                    *key = b->addr[j];
                    c = g_new0(uint64_t, 1);
                    g_hash_table_insert(counts, key, c);
                }
                *c += b->by_id[id];
            }
        }
        keys = g_list_sort(g_hash_table_get_keys(counts), cmp_addr);
        for (l = keys; l; l = l->next)
            fprintf(f, "%u %" PRIx64 " %" PRIu64 "\n", id, *(uint64_t *)l->data,
                    *(uint64_t *)g_hash_table_lookup(counts, l->data));
        g_list_free(keys);
        g_hash_table_destroy(counts);
        if (id_switches[id][0] || id_switches[id][1])
            fprintf(f, "# %u switches_l_free %" PRIu64 "\n# %u switches_l_vfp %" PRIu64 "\n", id,
                    id_switches[id][0], id, id_switches[id][1]);
        if (cache_on && id_access[id])
            fprintf(f, "# %u data_accesses %" PRIu64 "\n# %u l1d_misses %" PRIu64 "\n# %u l2_misses %" PRIu64 "\n",
                    id, id_access[id], id, id_l1m[id], id, id_l2m[id]);
        if (id_pld[id])
            fprintf(f, "# %u prefetches %" PRIu64 "\n# %u prefetch_fills %" PRIu64 "\n", id, id_pld[id], id,
                    id_pld_fill[id]);
    }
    fclose(f);
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
    if (layers_path) write_layers();
}

static int load_plds(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[96];
    if (!f) return 1;
    plds = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, NULL);
    while (fgets(line, sizeof line, f)) {
        char *end;
        uint64_t *k = g_new(uint64_t, 1);
        long reg, off;
        *k = strtoull(line, &end, 16);
        reg = strtol(end, &end, 10);
        off = strtol(end, &end, 10);
        if (reg < 0 || reg >= 15 || off < -32768 || off > 32767) {
            g_free(k);
            continue;
        }
        g_hash_table_insert(plds, k, (gpointer)(((uintptr_t)reg << 16 | ((uintptr_t)off & 0xFFFFu)) + 1u));
    }
    fclose(f);
    return 0;
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
        else if (!strcmp(argv[i], "marker")) marker = strtoull(v, NULL, 16);
        else if (!strcmp(argv[i], "layers")) layers_path = v;
        else if (!strcmp(argv[i], "l1d")) cache_on = !strcmp(v, "on");
        else if (!strcmp(argv[i], "pld")) {
            if (load_plds(v)) return 1;
        }
        else if (!strcmp(argv[i], "classes")) {
            if (load_classes(v)) return 1;
        } else return 1;
    }
    blocks = g_ptr_array_new_with_free_func(g_free);
    if (cache_on) {
        cache_init(&l1d, 32768u, 4u);
        cache_init(&l2, 2097152u, 8u);
    }
    if (marker != UINT64_MAX || plds) {
        r0_buf = g_byte_array_new();
        qemu_plugin_register_vcpu_init_cb(id, vcpu_init);
    }
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
