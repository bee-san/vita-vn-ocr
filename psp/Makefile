# psp/Makefile - the PSP EBOOT test app (vocr_test) and the PSP build of runtime/vocr.c.
#
#   make -C psp                 EBOOT.PBP: runtime/vocr.c + main.c, Allegrex, -O2 -G0, scalar FPU path
#   make -C psp size            code size of vocr.o and its worst-case stack chain (stack.py)
#   make -C psp clean
#
# PSPDEV defaults to psp/tools/pspdev (scripts/fetch_pspdev.sh). vocr.c gets the runtime's
# strict flags (C99, -Werror, -ffp-contract=off so a*b+c is never fused: the PSP build must
# compute the host's bits). PSP_LARGE_MEMORY=0: the app gets the 24 MB user partition every
# PSP has, PSP-1000 included, so what it measures is the PSP-1000 budget.

PSPDEV ?= $(abspath tools/pspdev)
export PATH := $(PSPDEV)/bin:$(PATH)
PSPSDK = $(PSPDEV)/psp/sdk
PYTHON ?= python3

TARGET = vocr_test
OBJS = main.o vocr.o
INCDIR = ../runtime
WARN = -std=c99 -ffp-contract=off -Wall -Wextra -Werror -Wshadow -Wpointer-arith \
       -Wstrict-prototypes -Wmissing-prototypes
CFLAGS = -O2 -G0 $(WARN) -isystem $(PSPSDK)/include -isystem $(PSPDEV)/psp/include
# VFPU=1: the float matrix products on the VFPU (runtime/vocr.c, VOCR_PSP_VFPU); opt-in
ifeq ($(VFPU),1)
CFLAGS += -DVOCR_PSP_VFPU
PSP_EBOOT_TITLE = vocr OCR test (VFPU)
else
PSP_EBOOT_TITLE = vocr OCR test
endif
VOCR_FLAGS = -Wpedantic -Wcast-align -fstack-usage -fcallgraph-info=su
LIBS = -lpsppower
BUILD_PRX = 1
PSP_LARGE_MEMORY = 0
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

main.o: main.c ../runtime/vocr.h

.PHONY: size
size: vocr.o
	psp-size vocr.o
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
