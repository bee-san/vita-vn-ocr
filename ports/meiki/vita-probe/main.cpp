// Manually started, offline test of the existing Meiki recognizer.
#include "decode.h"
#include "runtime.h"
extern "C" {
#include "screen.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <psp2/ctrl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>

extern "C" {
unsigned int _newlib_heap_size_user = 4u * 1024u * 1024u;
}
static constexpr size_t POOL_BYTES = MEIKI_WORKSPACE_BYTES;
static constexpr size_t RESERVE_BYTES = 8u * 1024u * 1024u;
static MeikiOutput output;
static MeikiOutput first_outputs[4];
static float data[MEIKI_INPUT_ELEMENTS];
static unsigned char font[16384] __attribute__((aligned(4)));

static unsigned buttons() {
    SceCtrlData pad = {};
    sceCtrlPeekBufferPositive(0, &pad, 1);
    return pad.buttons;
}
static bool exists(const char *path) {
    SceIoStat info = {};
    return sceIoGetstat(path, &info) >= 0;
}
static void idle() {
    sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
    sceKernelDelayThread(20000);
}
static int keep_awake(SceSize, void *) {
    for (;;) {
        sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
        sceKernelDelayThread(1000000);
    }
    return 0;
}
static int finish(bool automatic, int rc) {
    if (!automatic) {
        scr_head(3, SCR_WHITE,
                 "Results: ux0:data/MeikiProbe/results.jsonl   CIRCLE exits");
        while (buttons() & SCE_CTRL_CIRCLE)
            idle();
        while (!(buttons() & SCE_CTRL_CIRCLE))
            idle();
    }
    sceKernelExitProcess(rc);
    return rc;
}

int main() {
    bool automatic = exists("ux0:data/MeikiProbe/emulator-autoexit.flag");
    int display_rc = scr_init();
    if (display_rc < 0)
        return 1;
    FILE *face = fopen("app0:data/font.bin", "rb");
    if (face) {
        size_t bytes = fread(font, 1, sizeof(font), face);
        fclose(face);
        scr_load_font(font, bytes);
    }
    scr_head(0, SCR_CYAN, "Meiki OCR manual test v1.01");
    scr_head(1, SCR_WHITE,
             "Offline recognizer, four real crops, two repetitions");
    scr_head(
        2, SCR_WHITE,
        "CROSS starts. CIRCLE exits. No network or control plugins required.");
    scr_status(SCR_YELLOW, "Ready", 0, 8);
    if (!automatic) {
        while (buttons() & (SCE_CTRL_CROSS | SCE_CTRL_CIRCLE))
            idle();
        while (!(buttons() & (SCE_CTRL_CROSS | SCE_CTRL_CIRCLE)))
            idle();
        if (buttons() & SCE_CTRL_CIRCLE)
            return finish(true, 0);
    }
    SceUID ticker = sceKernelCreateThread("MeikiKeepAwake", keep_awake,
                                          0x10000100, 4096, 0, 0, nullptr);
    if (ticker < 0 || sceKernelStartThread(ticker, 0, nullptr) < 0) {
        if (ticker >= 0)
            sceKernelDeleteThread(ticker);
        scr_status(SCR_RED,
                   "Cannot start the keep-awake thread; no inference started",
                   1, 1);
        return finish(automatic, 5);
    }
    sceIoMkdir("ux0:data/MeikiProbe", 0777);
    freopen("ux0:data/MeikiProbe/runtime.log", "w", stdout);
    freopen("ux0:data/MeikiProbe/runtime-errors.log", "w", stderr);
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    FILE *log = fopen("ux0:data/MeikiProbe/results.jsonl", "w");
    FILE *inputs = fopen("app0:data/cases.tsv", "r");
    if (!log || !inputs) {
        if (log)
            fclose(log);
        if (inputs)
            fclose(inputs);
        scr_status(SCR_RED, "Cannot open bundled cases or results file", 1, 1);
        return finish(automatic, 2);
    }
    setvbuf(log, nullptr, _IONBF, 0);
    SceKernelFreeMemorySizeInfo before = {sizeof(before)};
    int free_rc = sceKernelGetFreeMemorySize(&before);
    if (free_rc < 0 || before.size_user < POOL_BYTES + RESERVE_BYTES) {
        fprintf(
            log,
            "{\"status\":\"memory_guard\",\"code\":%d,\"free_before\":%u}\n",
            free_rc, before.size_user);
        fclose(inputs);
        fclose(log);
        scr_status(SCR_RED,
                   "Not enough free application memory; no inference started",
                   1, 1);
        return finish(automatic, 3);
    }
    void *pool = nullptr;
    SceUID uid = sceKernelAllocMemBlock("MeikiWorkspace",
                                        SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
                                        POOL_BYTES, nullptr);
    if (uid < 0 || sceKernelGetMemBlockBase(uid, &pool) < 0 || !pool) {
        fprintf(log,
                "{\"status\":\"allocation_failed\",\"code\":%d,\"free_before\":"
                "%u}\n",
                uid, before.size_user);
        if (uid >= 0)
            sceKernelFreeMemBlock(uid);
        fclose(inputs);
        fclose(log);
        scr_status(SCR_RED, "Workspace allocation failed; no inference started",
                   1, 1);
        return finish(automatic, 3);
    }
    fprintf(log,
            "{\"status\":\"started\",\"pid\":%d,\"free_before\":%u,\"pool_"
            "bytes\":%u,\"newlib_heap_bytes\":%u}\n",
            sceKernelGetProcessId(), before.size_user, (unsigned)POOL_BYTES,
            _newlib_heap_size_user);
    char line[512], status[160];
    int count = 0, failed = 0, matches = 0, repeat_differences = 0;
    size_t peak = 0;
    while (fgets(line, sizeof(line), inputs)) {
        char *fields[4];
        unsigned fields_n = 0;
        fields[fields_n++] = line;
        for (char *p = line; *p; ++p) {
            if (*p == '\t' && fields_n < 4) {
                *p = 0;
                fields[fields_n++] = p + 1;
            } else if (*p == '\n' || *p == '\r') {
                *p = 0;
                break;
            }
        }
        if (fields_n != 4) {
            failed = 1;
            break;
        }
        char path[256];
        snprintf(path, sizeof(path), "app0:data/%s", fields[0]);
        unsigned crop_w = (unsigned)strtoul(fields[1], nullptr, 10);
        unsigned effective_w = (unsigned)strtoul(fields[2], nullptr, 10);
        FILE *input = fopen(path, "rb");
        if (!input) {
            failed = 1;
            break;
        }
        bool exact = fread(data, 1, sizeof(data), input) == sizeof(data) &&
                     fgetc(input) == EOF;
        fclose(input);
        if (!exact) {
            failed = 1;
            break;
        }
        snprintf(status, sizeof(status), "Running case %d of 8...", count + 1);
        scr_status(SCR_YELLOW, status, (unsigned)count, 8);
        idle();
        MeikiStats stats;
        unsigned heap_before = (unsigned)mallinfo().uordblks;
        int rc = meiki_run("app0:data/meiki-stream-int8.mnn", data, pool,
                           POOL_BYTES, &output, &stats);
        unsigned heap_after = (unsigned)mallinfo().uordblks;
        char text[193] = {};
        int decoded =
            rc ? -1
               : meiki_decode(&output, crop_w, effective_w, text, sizeof(text));
        bool match = !rc && decoded >= 0 && !strcmp(text, fields[3]);
        if (count < 4)
            first_outputs[count] = output;
        else if (count < 8 &&
                 memcmp(&first_outputs[count - 4], &output, sizeof(output)))
            ++repeat_differences;
        if (match)
            ++matches;
        if (stats.heap_peak > peak)
            peak = stats.heap_peak;
        fprintf(log,
                "{\"index\":%d,\"rc\":%d,\"matched\":%s,\"load_us\":%llu,"
                "\"inference_us\":%llu,\"heap_peak\":%u,\"heap_remaining\":%u,"
                "\"session_memory_mib\":%.6f,\"allocation_failed\":%d,\"text_"
                "hex\":\"",
                count++, rc, match ? "true" : "false",
                (unsigned long long)stats.load_us,
                (unsigned long long)stats.inference_us,
                (unsigned)stats.heap_peak, (unsigned)stats.heap_remaining,
                stats.session_memory_mib, stats.allocation_failed);
        for (const unsigned char *p = (const unsigned char *)text; *p; ++p)
            fprintf(log, "%02x", *p);
        fprintf(
            log,
            "\",\"newlib_used_before\":%u,\"newlib_used_after\":%u,\"codes\":[",
            heap_before, heap_after);
        for (int i = 0; i < 48; ++i)
            fprintf(log, "%s%d", i ? "," : "", output.codes[i]);
        fprintf(log, "],\"boxes\":[");
        for (int i = 0; i < 192; ++i)
            fprintf(log, "%s%.9g", i ? "," : "", output.boxes[i]);
        fprintf(log, "],\"scores\":[");
        for (int i = 0; i < 48; ++i)
            fprintf(log, "%s%.9g", i ? "," : "", output.scores[i]);
        fprintf(log, "]}\n");
        scr_log(match ? SCR_GREEN : SCR_RED, text);
        if (rc || decoded <= 0 || stats.heap_remaining) {
            failed = 1;
            break;
        }
        idle();
    }
    int free_workspace_rc = sceKernelFreeMemBlock(uid);
    if (free_workspace_rc < 0)
        failed = 1;
    SceKernelFreeMemorySizeInfo after = {sizeof(after)};
    sceKernelGetFreeMemorySize(&after);
    if (count != 8 || repeat_differences)
        failed = 1;
    fprintf(log,
            "{\"status\":\"complete\",\"cases\":%d,\"matches\":%d,\"failed\":%"
            "d,\"repeat_differences\":%d,\"workspace_free_rc\":%d,\"free_"
            "after\":%u}\n",
            count, matches, failed, repeat_differences, free_workspace_rc,
            after.size_user);
    fclose(inputs);
    fclose(log);
    snprintf(status, sizeof(status),
             "%s %d/8 runs. PC text %d/8. Repeat differences %d. Pool peak "
             "%.2f MiB.",
             failed ? "FAIL" : "COMPLETE", count, matches, repeat_differences,
             peak / 1048576.);
    scr_status(failed             ? SCR_RED
               : matches != count ? SCR_YELLOW
                                  : SCR_GREEN,
               status, (unsigned)count, 8);
    return finish(automatic, failed ? 4 : 0);
}
