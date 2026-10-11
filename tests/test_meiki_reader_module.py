"""Actual module bridge ownership with synchronous host kernel/engine shims."""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parents[1] / "ports/meiki"

DRIVER = r"""
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "module/bridge.c"
static MeikiModuleApi api;
static MeikiModuleStats initial;
static unsigned char heap[MEIKI_MODULE_METADATA_BYTES] __attribute__((aligned(64)));
static unsigned char pool[MEIKI_WORKSPACE_BYTES] __attribute__((aligned(64)));
static unsigned char model[MEIKI_MODEL_BYTES];
static float input[MEIKI_INPUT_ELEMENTS] __attribute__((aligned(64)));
static MeikiOutput output;
static unsigned calls, stops, thread_deletes, event_deletes;
static int locked, failure, join_error, delete_error;
static MeikiModelReader *original_reader;
static MeikiModuleRequest *original_request;

int sceKernelCreateMutex(const char *n, int a, int b, void *p) { (void)n;(void)a;(void)b;(void)p;return 1; }
int sceKernelTryLockMutex(int uid, int n) { (void)uid;(void)n;if(locked)return -1;locked=1;return 0; }
int sceKernelUnlockMutex(int uid, int n) { (void)uid;(void)n;locked=0;return 0; }
int sceKernelDeleteMutex(int uid) { (void)uid;return 0; }
int sceKernelCreateEventFlag(const char *n,int a,int b,void *p) {
    static int uid=1;(void)n;(void)a;(void)b;(void)p;return ++uid;
}
int sceKernelDeleteEventFlag(int uid) { (void)uid;++event_deletes;return 0; }
int sceKernelClearEventFlag(int uid, unsigned bits) { (void)uid;(void)bits;return 0; }
int sceKernelWaitEventFlag(int uid,unsigned bits,int flags,unsigned *out,unsigned *timeout) {
    (void)uid;(void)bits;(void)flags;(void)out;(void)timeout;assert(0);return -1;
}
int sceKernelDelayThread(unsigned us) { (void)us;return 0; }
int sceKernelCreateThread(const char *n,int(*entry)(unsigned,void*),int p,unsigned s,int a,int c,void *o) {
    (void)n;(void)entry;(void)p;(void)a;(void)c;(void)o;assert(s==128*1024);return 7;
}
int meiki_engine_initialize(void *p,size_t bytes,MeikiModuleStats *out) {
    assert(p==heap && bytes==sizeof(heap));memset(out,0,sizeof(*out));out->size=sizeof(*out);return 0;
}
int sceKernelStartThread(int uid,unsigned args,void *p) {
    (void)uid;(void)args;(void)p;complete(meiki_engine_initialize(metadata,metadata_bytes,&stats));return 0;
}
int sceKernelWaitThreadEnd(int uid,int *status,void *p) {
    (void)uid;(void)status;(void)p;if(join_error){join_error=0;return -1;}return 0;
}
int sceKernelDeleteThread(int uid) { (void)uid;++thread_deletes;if(delete_error){delete_error=0;return -1;}return 0; }
int meiki_engine_shutdown(MeikiModuleStats *out) { (void)out;++stops;return 0; }
static int read_model(void *context,uint32_t offset,void *target,uint32_t bytes) {
    assert(context==&calls && bytes && bytes<=8192 && bytes<=MEIKI_MODEL_BYTES-offset);
    ++calls;
    MeikiModuleRequest nested=*original_request;
    MeikiModelReader copied=*request.reader;
    nested.reader=&copied;
    assert(api.run(&nested,NULL)==MEIKI_MODULE_BUSY);
    assert(api.stop(NULL)==MEIKI_MODULE_BUSY);
    if(failure && offset==8192)return -17;
    memset(target,0,bytes);return 0;
}
int meiki_engine_run(const MeikiModuleRequest *in,MeikiModuleStats *out) {
    assert(in->reader==&model_reader && !in->model_path);
    assert(in->reader->read==read_model); // descriptor copied before dispatch
    return meiki_model_read(in->reader,model,sizeof(model),&out->runtime);
}
int meiki_engine_detect(const MeikiModuleDetectRequest *in,MeikiModuleStats *out) { (void)in;(void)out;return -1; }
int sceKernelSetEventFlag(int uid,unsigned bits) {
    (void)bits;if(uid==command_event){
        int next=__atomic_exchange_n(&command,0,__ATOMIC_ACQ_REL);
        if(next==COMMAND_RUN){
            original_reader->read=NULL; // caller descriptor is no longer consulted
            memset(&stats,0,sizeof(stats));complete(meiki_engine_run(&request,&stats));
        } else if(next==COMMAND_STOP)complete(meiki_engine_shutdown(&stats));
        else assert(0);
    }return 0;
}
int main(void) {
    MeikiModuleStart start={sizeof(start),MEIKI_MODULE_MAGIC,MEIKI_MODULE_ABI,
        heap,sizeof(heap),&api,&initial};
    start.abi=2;assert(module_start(sizeof(start),&start)==SCE_KERNEL_START_FAILED);
    start.abi=3;assert(module_start(sizeof(start),&start)==SCE_KERNEL_START_SUCCESS);
    assert(api.abi==3);
    MeikiModelReader reader={sizeof(reader),MEIKI_MODEL_BYTES,&calls,read_model};
    MeikiModuleRequest in={sizeof(in),NULL,input,pool,sizeof(pool),&output,&reader};
    original_reader=&reader;original_request=&in;
    MeikiModuleStats out __attribute__((aligned(8)));
    for(unsigned i=0;i<3;++i){
        reader.read=read_model;calls=0;
        assert(api.run(&in,&out)==0 && calls==659);
        assert(out.runtime.model_read_bytes==MEIKI_MODEL_BYTES);
        assert(!request.reader && !model_reader.read && !locked);
    }
    reader.read=read_model;failure=1;calls=0;
    assert(api.run(&in,&out)==-14 && calls==2 && out.runtime.model_read_error==-17);
    assert(!request.reader && !model_reader.context);
    failure=0;reader.read=read_model;calls=0;
    assert(api.run(&in,&out)==0 && calls==659);
    reader.read=read_model;
    reader.byte_count--;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);reader.byte_count++;
    reader.context=pool;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);
    reader.context=heap;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);
    reader.context=input;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);
    reader.context=&output;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);
    reader.context=&out;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);reader.context=&calls;
    MeikiModelReader *overlap=(MeikiModelReader*)pool;memcpy(overlap,&reader,sizeof(reader));
    in.reader=overlap;assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);in.reader=&reader;
    unsigned char unaligned[sizeof(reader)+8] __attribute__((aligned(8)));
    memcpy(unaligned+1,&reader,sizeof(reader));
    in.reader=(MeikiModelReader*)(unaligned+1);
    assert(api.run(&in,&out)==MEIKI_MODULE_INVALID);in.reader=&reader;
    assert(module_stop(0,NULL)==SCE_KERNEL_STOP_FAIL);
    unsigned saved_calls=calls;
    join_error=1;assert(api.stop(&out)==MEIKI_MODULE_KERNEL && stops==1);
    delete_error=1;assert(api.stop(&out)==MEIKI_MODULE_KERNEL && stops==1);
    assert(api.stop(&out)==0 && stops==1 && thread_deletes==2);
    assert(calls==saved_calls && api.run(&in,&out)==MEIKI_MODULE_STOPPED);
    assert(module_stop(0,NULL)==SCE_KERNEL_STOP_SUCCESS && event_deletes==2);
    puts("ABI3 copied reader, bounded reads, BUSY ownership, cancellation, pointer clearing, "
         "alignment/overlap rejection and retryable stop passed");
    return 0;
}
"""


def test_module_reader_synchronous_ownership(tmp_path):
    cc = shutil.which(os.environ.get("CC", "cc"))
    if not cc:
        pytest.skip("C compiler unavailable")
    shim = tmp_path / "psp2/kernel"
    shim.mkdir(parents=True)
    (shim / "clib.h").write_text(
        "#include <stddef.h>\n#include <string.h>\n"
        "#define sceClibMemset memset\n#define sceClibMemcpy memcpy\n"
        "#define sceClibStrnlen strnlen\n")
    (shim / "modulemgr.h").write_text(
        "#define SCE_KERNEL_START_FAILED -1\n#define SCE_KERNEL_START_SUCCESS 0\n"
        "#define SCE_KERNEL_STOP_FAIL -1\n#define SCE_KERNEL_STOP_SUCCESS 0\n")
    (shim / "threadmgr.h").write_text(
        "typedef int SceUID; typedef unsigned SceSize; typedef unsigned SceUInt;\n"
        "#define SCE_EVENT_WAITAND 1\n#define SCE_EVENT_WAITCLEAR_PAT 2\n"
        "int sceKernelCreateMutex(const char*,int,int,void*);\n"
        "int sceKernelTryLockMutex(int,int); int sceKernelUnlockMutex(int,int);\n"
        "int sceKernelDeleteMutex(int);\n"
        "int sceKernelCreateEventFlag(const char*,int,int,void*);\n"
        "int sceKernelDeleteEventFlag(int); int sceKernelClearEventFlag(int,unsigned);\n"
        "int sceKernelSetEventFlag(int,unsigned);\n"
        "int sceKernelWaitEventFlag(int,unsigned,int,unsigned*,unsigned*);\n"
        "int sceKernelDelayThread(unsigned);\n"
        "int sceKernelCreateThread(const char*,int(*)(unsigned,void*),int,unsigned,int,int,void*);\n"
        "int sceKernelStartThread(int,unsigned,void*);\n"
        "int sceKernelWaitThreadEnd(int,int*,void*); int sceKernelDeleteThread(int);\n")
    source, binary = tmp_path / "driver.c", tmp_path / "bridge-test"
    source.write_text(DRIVER)
    subprocess.run([cc, "-D_POSIX_C_SOURCE=200809L", "-std=c99", "-O1", "-g",
                    "-Wall", "-Wextra", "-Werror", *shlex.split(os.environ.get("CFLAGS", "")),
                    f"-I{tmp_path}", f"-I{PORT}", str(source), str(PORT / "model_reader.c"),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
