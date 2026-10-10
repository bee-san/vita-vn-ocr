/* Offline loader/ABI probe. This is separate from the v1.01 manual baseline. */
#include "../module/api.h"
#include "../decode.h"
#include "../detect_preprocess.h"
#include "screen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <psp2/ctrl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>

unsigned int _newlib_heap_size_user = 1u * 1024u * 1024u;
static float data[MEIKI_DETECT_ELEMENTS] __attribute__((aligned(64)));
static MeikiOutput output, first_outputs[4], references[4];
static unsigned char resize_scratch[MEIKI_PREPROCESS_SCRATCH_BYTES] __attribute__((aligned(64)));
static MeikiDetectOutput detector_output, first_detector_outputs[2];
static unsigned detector_cases, detector_repeat_differences;
static MeikiOutput first_pipeline_outputs[2][8];
static unsigned detected_line_cases, pipeline_repeat_differences;
static unsigned char font[16384] __attribute__((aligned(4)));
static MeikiModuleApi api;
static MeikiModuleRequest concurrent_request;
static volatile int concurrent_run_rc, concurrent_stop_rc;

static int exists(const char *path) {
    SceIoStat info = {0};
    return sceIoGetstat(path, &info) >= 0;
}
static unsigned buttons(void) {
    SceCtrlData pad = {0};
    sceCtrlPeekBufferPositive(0, &pad, 1);
    return pad.buttons;
}
static void idle(void) {
    sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
    sceKernelDelayThread(20000);
}
static int keep_awake(SceSize args, void *argp) {
    (void)args; (void)argp;
    for (;;) {
        sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
        sceKernelDelayThread(1000000);
    }
    return 0;
}
static int concurrent_calls(SceSize args, void *argp) {
    (void)args; (void)argp;
    MeikiModuleStats local;
    // First inference is substantially longer than this delay in Vita3K.
    // Record actual BUSY responses; scheduling cannot manufacture a pass.
    sceKernelDelayThread(50000);
    concurrent_run_rc = api.run(&concurrent_request, &local);
    concurrent_stop_rc = api.stop(&local);
    return 0;
}
static int finish(int automatic, int rc) {
    if (!automatic) {
        scr_head(3, SCR_WHITE, "ux0:data/MeikiModuleProbe/results.jsonl   CIRCLE exits");
        while (buttons() & SCE_CTRL_CIRCLE) idle();
        while (!(buttons() & SCE_CTRL_CIRCLE)) idle();
    }
    sceKernelExitProcess(rc);
    return rc;
}
static SceUID allocate(const char *name, size_t bytes, void **base) {
    SceUID uid = sceKernelAllocMemBlock(name, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, bytes, NULL);
    *base = NULL;
    if (uid >= 0 && sceKernelGetMemBlockBase(uid, base) < 0) {
        sceKernelFreeMemBlock(uid);
        return -1;
    }
    return uid;
}
static void raw_output(FILE *log, const MeikiOutput *out) {
    fprintf(log, "\"codes\":[");
    for (unsigned i = 0; i < 48; ++i) fprintf(log, "%s%d", i ? "," : "", out->codes[i]);
    fprintf(log, "],\"boxes\":[");
    for (unsigned i = 0; i < 192; ++i) fprintf(log, "%s%.9g", i ? "," : "", out->boxes[i]);
    fprintf(log, "],\"scores\":[");
    for (unsigned i = 0; i < 48; ++i) fprintf(log, "%s%.9g", i ? "," : "", out->scores[i]);
    fprintf(log, "]");
}


typedef struct { FILE *file; unsigned width; } RgbaSource;
static int rgba_rows(void *ud, unsigned y, unsigned n, unsigned char *rgba) {
    RgbaSource *source = (RgbaSource *)ud;
    if (fseek(source->file, (long)(8u + y * source->width * 4u), SEEK_SET)) return -1;
    return fread(rgba, source->width * 4u, n, source->file) == n ? (int)n : -1;
}
static int detected_lines(unsigned cycle, unsigned detector_index, const char *name,
                          unsigned w, unsigned h, void *workspace, FILE *log,
                          size_t *peak_pool, unsigned *peak_metadata) {
    MeikiCrop crops[8];
    unsigned count = 0;
    for (unsigned i = 0; i < 64; ++i) {
        if (!(detector_output.scores[i] > 0.5f)) continue;
        unsigned bounds[4];
        for (unsigned j = 0; j < 4; ++j) {
            float value = detector_output.boxes[i][j];
            unsigned limit = j & 1u ? h : w;
            if (value < 0) value = 0;
            if (value > limit) value = (float)limit;
            bounds[j] = (unsigned)value;
        }
        if (bounds[2] <= bounds[0] || bounds[3] <= bounds[1]) continue;
        if (count == 8) return 1;
        MeikiCrop crop = {bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]};
        unsigned place = count;
        while (place && crops[place - 1].y > crop.y) { crops[place] = crops[place - 1]; place--; }
        crops[place] = crop; count++;
    }
    char path[256];
    snprintf(path, sizeof(path), "app0:data/%s.rgba", name);
    FILE *rgba = fopen(path, "rb");
    if (!rgba) return 1;
    RgbaSource source = {rgba, w};
    MeikiRgbaImage image = {&source, w, h, rgba_rows, NULL};
    int failed = 0;
    for (unsigned i = 0; i < count && !failed; ++i) {
        MeikiPreprocessInfo prepared = {0};
        int prep_rc = meiki_preprocess_rgba(&image, &crops[i], data, MEIKI_DETECT_ELEMENTS,
            resize_scratch, sizeof(resize_scratch), &prepared);
        MeikiModuleRequest request = {sizeof(request), "app0:data/meiki-stream-int8.mnn",
            data, workspace, MEIKI_WORKSPACE_BYTES, &output};
        MeikiModuleStats stats = {0};
        int rc = prep_rc ? -999 : api.run(&request, &stats);
        unsigned different = 0;
        if (!rc && !cycle) first_pipeline_outputs[detector_index][i] = output;
        else if (!rc) different = memcmp(&first_pipeline_outputs[detector_index][i], &output, sizeof(output)) != 0;
        pipeline_repeat_differences += different;
        if (stats.runtime.heap_peak > *peak_pool) *peak_pool = stats.runtime.heap_peak;
        if (stats.metadata_brk_peak > *peak_metadata) *peak_metadata = stats.metadata_brk_peak;
        char text[193] = {0};
        int decoded = rc ? -1 : meiki_decode(&output, crops[i].w, prepared.effective_w, text, sizeof(text));
        fprintf(log, "{\"status\":\"detected_line\",\"cycle\":%u,\"detector_index\":%u,\"line_index\":%u,\"bbox\":[%u,%u,%u,%u],\"rc\":%d,\"prep_rc\":%d,\"repeat_different\":%u,\"heap_peak\":%u,\"heap_remaining\":%u,\"metadata_brk_peak\":%u,\"tainted\":%u,\"text_hex\":\"", cycle, detector_index, i, crops[i].x, crops[i].y, crops[i].w, crops[i].h, rc, prep_rc, different, (unsigned)stats.runtime.heap_peak, (unsigned)stats.runtime.heap_remaining, stats.metadata_brk_peak, stats.tainted);
        for (const unsigned char *p = (const unsigned char *)text; *p; ++p) fprintf(log, "%02x", *p);
        fprintf(log, "\"}\n");
        detected_line_cases++;
        if (rc || decoded <= 0 || stats.runtime.heap_remaining || stats.tainted || different) failed = 1;
    }
    fclose(rgba);
    return failed || !count;
}
static int detector_tests(unsigned cycle, void *workspace, FILE *log,
                          size_t *peak_pool, unsigned *peak_metadata) {
    FILE *cases = fopen("app0:data/detect.tsv", "r");
    if (!cases) return 0; /* Optional diagnostic fixtures, never baseline data. */
    char line[256], name[128], path[256];
    unsigned index = 0, w, h;
    int failed = 0;
    while (!failed && fgets(line, sizeof(line), cases)) {
        if (sscanf(line, "%127[^\t]\t%u\t%u", name, &w, &h) != 3 || index >= 2) { failed = 1; break; }
        snprintf(path, sizeof(path), "app0:data/%s.rgba", name);
        FILE *rgba = fopen(path, "rb");
        unsigned dimensions[2];
        if (!rgba || fread(dimensions, 4, 2, rgba) != 2 || dimensions[0] != w || dimensions[1] != h) {
            if (rgba) fclose(rgba);
            failed = 1; break;
        }
        RgbaSource source = {rgba, w};
        MeikiRgbaImage image = {&source, w, h, rgba_rows, NULL};
        MeikiCrop crop = {0, 0, w, h};
        MeikiDetectPreprocessInfo prepared = {0};
        int prep_rc = meiki_detect_preprocess_rgba(&image, &crop, data,
            MEIKI_DETECT_ELEMENTS, resize_scratch, sizeof(resize_scratch), &prepared);
        fclose(rgba);
        snprintf(path, sizeof(path), "app0:data/%s.f32", name);
        FILE *expected = fopen(path, "rb");
        unsigned prepared_different = prep_rc || !expected;
        size_t offset = 0;
        while (expected && !prepared_different && offset < sizeof(data)) {
            size_t bytes = sizeof(data) - offset;
            if (bytes > sizeof(resize_scratch)) bytes = sizeof(resize_scratch);
            if (fread(resize_scratch, 1, bytes, expected) != bytes ||
                memcmp(resize_scratch, (const unsigned char *)data + offset, bytes)) prepared_different = 1;
            offset += bytes;
        }
        if (expected) { if (fgetc(expected) != EOF) prepared_different = 1; fclose(expected); }
        MeikiModuleDetectRequest request = {sizeof(request), "app0:data/meiki-detect-int8.mnn",
            data, prepared.target_width, prepared.target_height, workspace,
            MEIKI_WORKSPACE_BYTES, &detector_output};
        MeikiModuleStats stats = {0};
        int rc = prepared_different ? -999 : api.detect(&request, &stats);
        unsigned repeat_different = 0;
        if (!rc && !cycle) first_detector_outputs[index] = detector_output;
        else if (!rc) repeat_different = memcmp(&first_detector_outputs[index], &detector_output, sizeof(detector_output)) != 0;
        detector_repeat_differences += repeat_different;
        if (stats.runtime.heap_peak > *peak_pool) *peak_pool = stats.runtime.heap_peak;
        if (stats.metadata_brk_peak > *peak_metadata) *peak_metadata = stats.metadata_brk_peak;
        fprintf(log, "{\"status\":\"detector\",\"cycle\":%u,\"detector_index\":%u,\"rc\":%d,\"prepared_different\":%u,\"repeat_different\":%u,\"width\":%u,\"height\":%u,\"target_width\":%d,\"target_height\":%d,\"heap_peak\":%u,\"heap_remaining\":%u,\"allocation_failed\":%d,\"metadata_brk_peak\":%u,\"metadata_used_before\":%u,\"metadata_used_after\":%u,\"tainted\":%u,\"labels\":[", cycle,index,rc,prepared_different,repeat_different,w,h,prepared.target_width,prepared.target_height,(unsigned)stats.runtime.heap_peak,(unsigned)stats.runtime.heap_remaining,stats.runtime.allocation_failed,stats.metadata_brk_peak,stats.metadata_used_before,stats.metadata_used_after,stats.tainted);
        for (unsigned i = 0; i < 64; i++) fprintf(log, "%s%d", i ? "," : "", detector_output.labels[i]);
        fprintf(log, "],\"boxes\":[");
        for (unsigned i = 0; i < 256; i++) fprintf(log, "%s%.9g", i ? "," : "", ((float *)detector_output.boxes)[i]);
        fprintf(log, "],\"scores\":[");
        for (unsigned i = 0; i < 64; i++) fprintf(log, "%s%.9g", i ? "," : "", detector_output.scores[i]);
        fprintf(log, "]}\n");
        if (rc || stats.runtime.heap_remaining || stats.tainted || repeat_different || prepared_different) failed = 1;
        if (!failed) failed = detected_lines(cycle, index, name, w, h, workspace, log, peak_pool, peak_metadata);
        detector_cases++; index++;
    }
    fclose(cases);
    return failed || index != 2;
}

int main(void) {
    int automatic = exists("ux0:data/MeikiModuleProbe/emulator-autoexit.flag");
    if (scr_init() < 0) return 1;
    FILE *face = fopen("app0:data/font.bin", "rb");
    if (face) { size_t n = fread(font, 1, sizeof(font), face); fclose(face); scr_load_font(font, n); }
    scr_head(0, SCR_CYAN, "Meiki private-module loader probe v0.01");
    scr_head(1, SCR_WHITE, "Two load/start/run/stop/unload cycles. Four crops twice per cycle.");
    scr_head(2, SCR_WHITE, "Offline. CROSS starts. CIRCLE exits. No remote-control plugin.");
    scr_status(SCR_YELLOW, "Ready", 0, 16);
    if (!automatic) {
        while (buttons() & (SCE_CTRL_CROSS | SCE_CTRL_CIRCLE)) idle();
        while (!(buttons() & (SCE_CTRL_CROSS | SCE_CTRL_CIRCLE))) idle();
        if (buttons() & SCE_CTRL_CIRCLE) return finish(1, 0);
    }
    SceUID ticker = sceKernelCreateThread("MeikiModuleAwake", keep_awake, 0x10000100, 4096, 0, 0, NULL);
    if (ticker < 0 || sceKernelStartThread(ticker, 0, NULL) < 0) {
        scr_status(SCR_RED, "Keep-awake thread failed", 0, 16);
        return finish(automatic, 2);
    }
    sceIoMkdir("ux0:data/MeikiModuleProbe", 0777);
    FILE *log = fopen("ux0:data/MeikiModuleProbe/results.jsonl", "w");
    FILE *baseline = fopen("app0:data/reference.raw", "rb");
    if (!log || !baseline || fread(references, 1, sizeof(references), baseline) != sizeof(references) || fgetc(baseline) != EOF) {
        if (baseline) fclose(baseline);
        if (log) fclose(log);
        scr_status(SCR_RED, "Cannot open inputs/results/reference", 0, 16);
        return finish(automatic, 2);
    }
    fclose(baseline);
    setvbuf(log, NULL, _IONBF, 0);
    int failed = 0, count = 0, matches = 0, repeat_differences = 0, baseline_differences = 0, cycles = 0;
    size_t peak_pool = 0;
    unsigned peak_metadata = 0;
    unsigned total_cycles = exists("app0:data/meiki-detect-oom.mnn") ? 3u : 2u;
    for (unsigned cycle = 0; cycle < total_cycles && !failed; ++cycle) {
        void *metadata = NULL, *workspace = NULL;
        SceUID metadata_uid = allocate("MeikiModuleMetadata", MEIKI_MODULE_METADATA_BYTES, &metadata);
        SceUID workspace_uid = allocate("MeikiModuleWorkspace", MEIKI_WORKSPACE_BYTES, &workspace);
        if (metadata_uid < 0 || workspace_uid < 0 || !metadata || !workspace) {
            fprintf(log, "{\"status\":\"buffer_allocation_failed\",\"cycle\":%u,\"metadata_uid\":%d,\"workspace_uid\":%d}\n", cycle, metadata_uid, workspace_uid);
            if (metadata_uid >= 0) sceKernelFreeMemBlock(metadata_uid);
            if (workspace_uid >= 0) sceKernelFreeMemBlock(workspace_uid);
            failed = 1;
            break;
        }
        memset(&api, 0, sizeof(api));
        MeikiModuleStats initial = {0};
        MeikiModuleStart start = {sizeof(start), MEIKI_MODULE_MAGIC, MEIKI_MODULE_ABI, metadata, MEIKI_MODULE_METADATA_BYTES, &api, &initial};
        int start_status = -1;
        SceUID module = sceKernelLoadStartModule("app0:data/meiki-engine.suprx", sizeof(start), &start, 0, NULL, &start_status);
        fprintf(log, "{\"status\":\"loaded\",\"cycle\":%u,\"module\":%d,\"start_status\":%d,\"api_abi\":%u,\"engine_thread\":%d,\"caller_thread\":%d,\"metadata_brk_peak\":%u,\"metadata_used_after\":%u}\n", cycle, module, start_status, api.abi, initial.engine_thread_id, sceKernelGetThreadId(), initial.metadata_brk_peak, initial.metadata_used_after);
        if (module < 0 || start_status != 0 || api.abi != MEIKI_MODULE_ABI || api.size != sizeof(api) || !api.run || !api.stop || !api.detect) {
            int safe = module < 0;
            if (module >= 0 && start_status == SCE_KERNEL_START_FAILED)
                safe = sceKernelStopUnloadModule(module, 0, NULL, 0, NULL, &start_status) >= 0 && !start_status;
            if (safe) {
                sceKernelFreeMemBlock(workspace_uid);
                sceKernelFreeMemBlock(metadata_uid);
            }
            failed = 1;
            break;
        }
        MeikiModuleStats probe_stats = {0};
        MeikiModuleRequest request = {sizeof(request), "app0:data/missing-model.mnn", data, workspace, MEIKI_WORKSPACE_BYTES, &output};
        if (cycle == 2) {
            MeikiModuleDetectRequest fault = {sizeof(fault), "app0:data/meiki-detect-oom.mnn",
                data, 960, 544, workspace, MEIKI_WORKSPACE_BYTES, &detector_output};
            int fault_rc = api.detect(&fault, &probe_stats);
            int rejected_rc = api.run(&request, NULL);
            fprintf(log, "{\"status\":\"detector_oom\",\"rc\":%d,\"rejected_rc\":%d,\"heap_peak\":%u,\"heap_remaining\":%u,\"allocation_failed\":%d,\"metadata_brk_peak\":%u,\"tainted\":%u}\n", fault_rc, rejected_rc, (unsigned)probe_stats.runtime.heap_peak, (unsigned)probe_stats.runtime.heap_remaining, probe_stats.runtime.allocation_failed, probe_stats.metadata_brk_peak, probe_stats.tainted);
            if (fault_rc != MEIKI_MODULE_TAINTED || rejected_rc != MEIKI_MODULE_TAINTED ||
                !probe_stats.runtime.allocation_failed || !probe_stats.tainted) failed = 1;
            goto cleanup_module;
        }
        int missing_rc = api.run(&request, &probe_stats);
        request.workspace_bytes--;
        int invalid_rc = api.run(&request, &probe_stats);
        request.workspace_bytes++;
        fprintf(log, "{\"status\":\"guard_checks\",\"cycle\":%u,\"missing_model_rc\":%d,\"invalid_workspace_rc\":%d,\"tainted\":%u}\n", cycle, missing_rc, invalid_rc, probe_stats.tainted);
        if (missing_rc != -2 || invalid_rc != MEIKI_MODULE_INVALID || probe_stats.tainted) failed = 1;
        MeikiModuleDetectRequest guard = {sizeof(guard), "app0:data/missing-model.mnn",
            data, 320, 192, workspace, MEIKI_WORKSPACE_BYTES, &detector_output};
        int detector_missing_rc = api.detect(&guard, &probe_stats);
        guard.target_width = 0;
        int detector_invalid_rc = api.detect(&guard, &probe_stats);
        guard.target_width = 320;
        guard.model_path = "app0:data/meiki-stream-int8.mnn";
        int detector_shape_rc = api.detect(&guard, &probe_stats);
        fprintf(log, "{\"status\":\"detector_guards\",\"missing_model_rc\":%d,\"invalid_target_rc\":%d,\"wrong_model_rc\":%d,\"heap_remaining\":%u,\"tainted\":%u}\n", detector_missing_rc, detector_invalid_rc, detector_shape_rc, (unsigned)probe_stats.runtime.heap_remaining, probe_stats.tainted);
        if (detector_missing_rc != -2 || detector_invalid_rc != MEIKI_MODULE_INVALID ||
            detector_shape_rc != -4 || probe_stats.runtime.heap_remaining || probe_stats.tainted) failed = 1;
        if (!failed) failed = detector_tests(cycle, workspace, log, &peak_pool, &peak_metadata);
        FILE *inputs = fopen("app0:data/cases.tsv", "r");
        unsigned index = 0;
        char line[512], status[180];
        request.model_path = "app0:data/meiki-stream-int8.mnn";
        while (inputs && !failed && fgets(line, sizeof(line), inputs)) {
            char *fields[4] = {line, NULL, NULL, NULL};
            unsigned fields_n = 1;
            for (char *p = line; *p; ++p) {
                if (*p == '\t' && fields_n < 4) { *p = 0; fields[fields_n++] = p + 1; }
                else if (*p == '\n' || *p == '\r') { *p = 0; break; }
            }
            if (fields_n != 4 || index >= 8) { failed = 1; break; }
            char path[256];
            snprintf(path, sizeof(path), "app0:data/%s", fields[0]);
            FILE *input = fopen(path, "rb");
            if (!input || fread(data, 1, MEIKI_INPUT_ELEMENTS * sizeof(float), input) != MEIKI_INPUT_ELEMENTS * sizeof(float) || fgetc(input) != EOF) {
                if (input) fclose(input);
                failed = 1; break;
            }
            fclose(input);
            unsigned crop_w = (unsigned)strtoul(fields[1], NULL, 10), effective_w = (unsigned)strtoul(fields[2], NULL, 10);
            snprintf(status, sizeof(status), "Cycle %u, case %u of 8", cycle + 1, index + 1);
            scr_status(SCR_YELLOW, status, (unsigned)count, 16);
            idle();
            SceUID concurrent = -1;
            if (!cycle && !index) {
                concurrent_request = request;
                concurrent_run_rc = concurrent_stop_rc = 999;
                concurrent = sceKernelCreateThread("MeikiBusyProbe", concurrent_calls, 0x10000100, 16384, 0, 0, NULL);
                if (concurrent < 0 || sceKernelStartThread(concurrent, 0, NULL) < 0) { failed = 1; break; }
            }
            MeikiModuleStats stats = {0};
            int rc = api.run(&request, &stats);
            if (concurrent >= 0) {
                int thread_status;
                sceKernelWaitThreadEnd(concurrent, &thread_status, NULL);
                sceKernelDeleteThread(concurrent);
                fprintf(log, "{\"status\":\"busy_checks\",\"run_rc\":%d,\"stop_rc\":%d}\n", concurrent_run_rc, concurrent_stop_rc);
                if (concurrent_run_rc != MEIKI_MODULE_BUSY || concurrent_stop_rc != MEIKI_MODULE_BUSY) failed = 1;
            }
            char text[193] = {0};
            int decoded = rc ? -1 : meiki_decode(&output, crop_w, effective_w, text, sizeof(text));
            int match = !rc && decoded > 0 && !strcmp(text, fields[3]);
            matches += match;
            int repeated = 0, baseline_diff = memcmp(&references[index % 4], &output, sizeof(output)) != 0;
            baseline_differences += baseline_diff;
            if (!cycle && index < 4) first_outputs[index] = output;
            else repeated = memcmp(&first_outputs[index % 4], &output, sizeof(output)) != 0;
            repeat_differences += repeated;
            if (stats.runtime.heap_peak > peak_pool) peak_pool = stats.runtime.heap_peak;
            if (stats.metadata_brk_peak > peak_metadata) peak_metadata = stats.metadata_brk_peak;
            fprintf(log, "{\"cycle\":%u,\"index\":%u,\"rc\":%d,\"matched\":%s,\"repeated_different\":%s,\"baseline_different\":%s,\"heap_peak\":%u,\"heap_remaining\":%u,\"allocation_failed\":%d,\"metadata_capacity\":%u,\"metadata_brk_peak\":%u,\"metadata_used_before\":%u,\"metadata_used_after\":%u,\"tainted\":%u,\"engine_thread\":%d,\"text_hex\":\"", cycle, index, rc, match ? "true" : "false", repeated ? "true" : "false", baseline_diff ? "true" : "false", (unsigned)stats.runtime.heap_peak, (unsigned)stats.runtime.heap_remaining, stats.runtime.allocation_failed, stats.metadata_capacity, stats.metadata_brk_peak, stats.metadata_used_before, stats.metadata_used_after, stats.tainted, stats.engine_thread_id);
            for (const unsigned char *p = (const unsigned char *)text; *p; ++p) fprintf(log, "%02x", *p);
            fprintf(log, "\","); raw_output(log, &output); fprintf(log, "}\n");
            scr_log(match ? SCR_GREEN : SCR_YELLOW, text);
            count++; index++;
            if (rc || decoded <= 0 || stats.runtime.heap_remaining || stats.tainted || stats.metadata_brk_peak > MEIKI_MODULE_METADATA_BYTES) failed = 1;
        }
        if (inputs) fclose(inputs);
        if (index != 8) failed = 1;
cleanup_module:;
        MeikiModuleStats stopped = {0};
        int stop_rc = api.stop(&stopped), unload_status = -1;
        int unload_rc = stop_rc ? -1 : sceKernelStopUnloadModule(module, 0, NULL, 0, NULL, &unload_status);
        fprintf(log, "{\"status\":\"unloaded\",\"cycle\":%u,\"stop_rc\":%d,\"unload_rc\":%d,\"unload_status\":%d,\"metadata_brk_peak\":%u,\"metadata_used_before\":%u,\"metadata_used_after\":%u,\"tainted\":%u}\n", cycle, stop_rc, unload_rc, unload_status, stopped.metadata_brk_peak, stopped.metadata_used_before, stopped.metadata_used_after, stopped.tainted);
        // A failed stop/unload retains both buffers until process exit.
        if (stop_rc || unload_rc < 0 || unload_status) { failed = 1; break; }
        int free_workspace_rc = sceKernelFreeMemBlock(workspace_uid), free_metadata_rc = sceKernelFreeMemBlock(metadata_uid);
        fprintf(log, "{\"status\":\"buffers_freed\",\"cycle\":%u,\"workspace_rc\":%d,\"metadata_rc\":%d}\n", cycle, free_workspace_rc, free_metadata_rc);
        if (free_workspace_rc || free_metadata_rc) failed = 1;
        if (cycle < 2) cycles++;
    }
    if (count != 16 || cycles != 2 || repeat_differences || baseline_differences || detector_repeat_differences || pipeline_repeat_differences) failed = 1;
    fprintf(log, "{\"status\":\"detector_summary\",\"cases\":%u,\"repeat_differences\":%u,\"detected_line_cases\":%u,\"pipeline_repeat_differences\":%u}\n", detector_cases, detector_repeat_differences, detected_line_cases, pipeline_repeat_differences);
    fprintf(log, "{\"status\":\"complete\",\"cases\":%d,\"cycles\":%d,\"matches\":%d,\"failed\":%d,\"repeat_differences\":%d,\"baseline_differences\":%d,\"pool_peak\":%u,\"metadata_brk_peak\":%u}\n", count, cycles, matches, failed, repeat_differences, baseline_differences, (unsigned)peak_pool, peak_metadata);
    fclose(log);
    char status[180];
    snprintf(status, sizeof(status), "%s %d/16, loads %d/2. Raw diff %d; repeats %d. Pool %.2f MiB; metadata %.2f MiB.", failed ? "FAIL" : "COMPLETE", count, cycles, baseline_differences, repeat_differences, peak_pool / 1048576., peak_metadata / 1048576.);
    scr_status(failed ? SCR_RED : (matches != count ? SCR_YELLOW : SCR_GREEN), status, (unsigned)count, 16);
    return finish(automatic, failed ? 4 : 0);
}
