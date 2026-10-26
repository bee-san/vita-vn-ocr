# psp/Makefile - the PSP EBOOT test app (vocr_test) and the PSP build of runtime/vocr.c.
#
#   make -C psp                 EBOOT.PBP: runtime/vocr.c + main.c, Allegrex, -O2 -G0, scalar FPU path
#   make -C psp VFPU=1          the same with the VFPU matrix products (opt-in)
#   make -C psp LARGEMEM=1      PSP_LARGE_MEMORY = 1: on a 64 MB PSP (2000 / 3000 / Go) the app gets the
#                               larger user partition (about 52 MB instead of 24 MB); ignored on a PSP-1000
#   make -C psp TITLE="..."     the XMB title (default "vocr OCR test", with " (VFPU)" and " 64MB" as built)
#   make -C psp size            code size of vocr.o and the line finder, and vocr.o's worst-case stack chain
#   make -C psp clean
#
# PSPDEV defaults to psp/tools/pspdev (scripts/fetch_pspdev.sh). vocr.c gets the runtime's
# strict flags (C99, -Werror, -ffp-contract=off so a*b+c is never fused: the PSP build must
# compute the host's bits). Without LARGEMEM=1 the app gets the 24 MB user partition every
# PSP has, so what it measures is the smallest PSP's budget. The line finder and its refinement
# (runtime/vocr_lines.c, vocr_lines_refine.c) are linked for the app's finder mode (main.c).

PSPDEV ?= $(abspath tools/pspdev)
export PATH := $(PSPDEV)/bin:$(PATH)
PSPSDK = $(PSPDEV)/psp/sdk
PYTHON ?= python3

TARGET = vocr_test
OBJS = main.o vocr.o vocr_lines.o vocr_lines_refine.o
INCDIR = ../runtime
WARN = -std=c99 -ffp-contract=off -Wall -Wextra -Werror -Wshadow -Wpointer-arith \
       -Wstrict-prototypes -Wmissing-prototypes
CFLAGS = -O2 -G0 $(WARN) -isystem $(PSPSDK)/include -isystem $(PSPDEV)/psp/include
# VFPU=1: the float matrix products on the VFPU (runtime/vocr.c, VOCR_PSP_VFPU); opt-in
ifeq ($(VFPU),1)
CFLAGS += -DVOCR_PSP_VFPU
TITLE_BASE = vocr OCR test (VFPU)
else
TITLE_BASE = vocr OCR test
endif
ifeq ($(LARGEMEM),1)
PSP_LARGE_MEMORY = 1
CFLAGS += -DVOCR_PSP_LARGE_MEMORY
PSP_EBOOT_TITLE = $(TITLE_BASE) 64MB
else
PSP_LARGE_MEMORY = 0
PSP_EBOOT_TITLE = $(TITLE_BASE)
endif
ifneq ($(TITLE),)
PSP_EBOOT_TITLE = $(TITLE)
endif
VOCR_FLAGS = -Wpedantic -Wcast-align -fstack-usage -fcallgraph-info=su
LIBS = -lpsppower
BUILD_PRX = 1
EXTRA_TARGETS = EBOOT.PBP
EXTRA_CLEAN = vocr.su vocr.ci main.su main.ci

ifneq ($(wildcard $(PSPSDK)/lib/build.mak),)
include $(PSPSDK)/lib/build.mak
else
all:
	@echo "no PSPSDK at $(PSPSDK): run psp/scripts/fetch_pspdev.sh, or set PSPDEV" >&2; exit 1
endif

vocr.o: ../runtime/vocr.c ../runtime/vocr.h
	$(CC) $(CFLAGS) $(VOCR_FLAGS) -c -o $@ $<

vocr_lines.o: ../runtime/vocr_lines.c ../runtime/vocr_lines.h
	$(CC) $(CFLAGS) -c -o $@ $<

vocr_lines_refine.o: ../runtime/vocr_lines_refine.c ../runtime/vocr_lines_refine.h ../runtime/vocr_lines_rows.h \
                     ../runtime/vocr_lines.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c ../runtime/vocr.h ../runtime/vocr_lines.h ../runtime/vocr_lines_refine.h

.PHONY: size
size: vocr.o vocr_lines.o vocr_lines_refine.o
	psp-size vocr.o vocr_lines.o vocr_lines_refine.o
	$(PYTHON) stack.py vocr.ci

# host tool: the D-cache misses of vocr_run() (cachesim.c); not part of the PSP build
HOSTCC ?= gcc
KASAN = -fsanitize=kernel-address --param asan-instrumentation-with-call-threshold=0 \
        --param asan-stack=0 --param asan-globals=0
.PHONY: cachesim
cachesim: build/cachesim
build/cachesim: cachesim.c ../runtime/vocr.c ../runtime/vocr.h
	mkdir -p build
	$(HOSTCC) -std=c99 -O2 -ffp-contract=off $(KASAN) -c -o build/vocr_kasan.o ../runtime/vocr.c
	$(HOSTCC) $(WARN) -O2 -o $@ cachesim.c build/vocr_kasan.o -Wl,--wrap=memcpy -Wl,--wrap=memset -lm
