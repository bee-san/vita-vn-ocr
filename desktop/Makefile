# desktop/Makefile - VOCR's desktop C path (vocr_desk.h; docs/desktop.md): the desktop line detector and the page
# pipeline with VOCR's recognizer (runtime/vocr.c, unchanged), as a shared library and a command line.
#
#   make                 build/libvocrdesk.so and build/vocr_desk_cli, AVX2 + FMA (x86-64-v3: 2013+ x86 CPUs)
#   make ARCH=           the same without SIMD flags (portable C; the plain-C kernels)
#   make ARCH=-march=native
#   make OUT=build-o2 OPT=-O2 ARCH=    a plain -O2 build in another directory
#
# runtime/vocr.c keeps its own contract (-std=c99 -ffp-contract=off: the same W8I bits at every optimisation level,
# docs/onnx.md and issue #164); the detector's kernels use fused multiply-adds when ARCH allows them.

CC ?= gcc
OUT ?= build
OPT ?= -O3
ARCH ?= -march=x86-64-v3
WARN = -std=c99 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes
RT = ../runtime
SRC = vocr_desk.c vocr_desk_page.c $(RT)/vocr.c $(RT)/vocr_lines.c
CFLAGS_ALL = $(WARN) -ffp-contract=off $(OPT) $(ARCH) -I$(RT) -I. $(EXTRA)

all: $(OUT)/libvocrdesk.so $(OUT)/vocr_desk_cli

$(OUT):
	mkdir -p $(OUT)

$(OUT)/libvocrdesk.so: $(SRC) vocr_desk.h | $(OUT)
	$(CC) $(CFLAGS_ALL) -fPIC -shared -o $@ $(SRC) -lm -lpthread

$(OUT)/vocr_desk_cli: vocr_desk_cli.c $(SRC) vocr_desk.h | $(OUT)
	$(CC) $(CFLAGS_ALL) -o $@ vocr_desk_cli.c $(SRC) -lm -lpthread

clean:
	rm -rf $(OUT)

.PHONY: all clean

# textspot-desktop's box refinement (vocr_tsd_refine.h, issue #173; its reference: tools/tsdesk_refine.py): linked
# into the library and the command line (the recipes expand SRC when they run), and built alone as
# build/libvocrtsdr.so for tools/tsdesk_refine.CRefine and tests/test_tsdesk_refine.py (make refine)
SRC += vocr_tsd_refine.c
$(OUT)/libvocrdesk.so $(OUT)/vocr_desk_cli: vocr_tsd_refine.c vocr_tsd_refine.h

$(OUT)/libvocrtsdr.so: vocr_tsd_refine.c vocr_tsd_refine.h | $(OUT)
	$(CC) $(CFLAGS_ALL) -fPIC -shared -o $@ vocr_tsd_refine.c -lm

refine: $(OUT)/libvocrtsdr.so

.PHONY: refine
