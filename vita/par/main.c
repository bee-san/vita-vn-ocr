/* vita/par: the runtime's host drivers on the PS Vita, for the bit-identity check in Vita3K (vita/vita3k_parity.py).
 *
 * The app links runtime/tools/vocr_cli.c (its main renamed vocr_cli_main) and runtime/vocr_textspot6_test.c (main
 * renamed ts6_main) with the Vita build of the runtime (vita/CMakeLists.txt), and runs the jobs of
 * ux0:data/vocrpar/jobs.txt one by one, one per line: a tool name ("cli" or "ts6") and its arguments, separated by
 * single spaces (no quoting; paths are absolute ux0: paths). Job k's stdout goes to ux0:data/vocrpar/out_k.txt and its
 * stderr to err_k.txt; after it, "k rc" is appended to status.txt. When every job has run, the app writes done.txt and
 * exits, so the emulator quits with it. The outputs are those of the same command on the host, byte for byte, when the
 * runtime computes the same results there: vita/vita3k_parity.py compares them.
 *
 * Not a console tool: it shows nothing on the screen and is meant for the emulator (on a PCH-1000 it would run the
 * same way, without a display). */
#include <psp2/kernel/processmgr.h>

#include <malloc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIR "ux0:data/vocrpar/"
#define MAX_ARGS 64

int vocr_cli_main(int argc, char **argv);
int ts6_main(int argc, char **argv);

/* newlib's heap (sbrk) for the packs, records and arenas the drivers malloc */
int _newlib_heap_size_user = 200 * 1024 * 1024;

/* VitaSDK's newlib declares posix_memalign but does not define it; vocr_cli.c uses it for the blob and the arena */
int posix_memalign(void **p, size_t align, size_t n)
{
    void *q = memalign(align, n ? n : align);
    if (!q) return 12; /* ENOMEM */
    *p = q;
    return 0;
}

static void append(const char *path, const char *text)
{
    FILE *f = fopen(path, "a");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

int main(void)
{
    static char line[8192];
    FILE *jobs = fopen(DIR "jobs.txt", "r");
    unsigned k = 0;
    if (!jobs) {
        append(DIR "status.txt", "no jobs.txt\n");
    } else {
        while (fgets(line, sizeof line, jobs)) {
            char *argv[MAX_ARGS + 1], path[128], msg[64];
            int argc = 0, rc;
            char *p = line, *e;
            line[strcspn(line, "\r\n")] = 0;
            while (*p && argc < MAX_ARGS) {
                while (*p == ' ') p++;
                if (!*p) break;
                argv[argc++] = p;
                e = strchr(p, ' ');
                if (!e) break;
                *e = 0;
                p = e + 1;
            }
            argv[argc] = NULL;
            if (!argc) continue;
            snprintf(path, sizeof path, DIR "out_%u.txt", k);
            rc = freopen(path, "w", stdout) ? 0 : -101;
            snprintf(path, sizeof path, DIR "err_%u.txt", k);
            if (!rc && !freopen(path, "w", stderr)) rc = -102;
            if (rc) {
            } else if (!strcmp(argv[0], "cli")) {
                rc = vocr_cli_main(argc, argv);
            } else if (!strcmp(argv[0], "ts6")) {
                rc = ts6_main(argc, argv);
            } else {
                rc = -100;
            }
            fflush(stdout);
            fflush(stderr);
            snprintf(msg, sizeof msg, "%u %d\n", k, rc);
            append(DIR "status.txt", msg);
            k++;
        }
        fclose(jobs);
    }
    fclose(stdout);
    fclose(stderr);
    append(DIR "done.txt", "done\n");
    sceKernelExitProcess(0);
    return 0;
}
