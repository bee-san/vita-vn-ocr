# desktop/Makefile - VOCR's desktop C path (vocr_desk.h; docs/desktop.md): the desktop line detector and the page
# pipeline with VOCR's recognizer (runtime/vocr.c, unchanged), as a shared library and a command line; and
# textspot-desktop's detector (vocr_tsd.h; issue #173) in the same library, with its page pipeline, its frame luma and
# resize (vocr_tsd_resize.h: vocr_desk_grey's and vocr_desk_resize's rules with SIMD paths), its box refinement
# (vocr_tsd_refine.h), its post-pass and reading filter (vocr_tsd_post.h) and a bench.
#
#   make                 build/libvocrdesk.so, build/vocr_desk_cli and build/vocr_tsd_bench, AVX2 + FMA
#                        (x86-64-v3: 2013+ x86 CPUs), and build/libvocrtsdr.so (the box refinement alone)
#   make ARCH=           the same without SIMD flags (portable C; the plain-C kernels)
#   make ARCH=-march=x86-64-v4    textspot-desktop's AVX-512 kernels (AVX-512 F/BW/CD/DQ/VL CPUs; the maps are the
#                        AVX2 build's bit for bit; -DVOCR_TSD_NO_AVX512 keeps the AVX2 kernels in such a build)
#   make ARCH=-march=native
#   make OUT=build-o2 OPT=-O2 ARCH=    a plain -O2 build in another directory
#   make OUT=build-scalar EXTRA=-DVOCR_TSD_SCALAR    textspot-desktop's plain-C kernels in an AVX2 build
#   make OUT=build-prof EXTRA=-DVOCR_TSD_PROF        per-step host timing in vocr_tsd_bench (not thread safe)
#
# runtime/vocr.c keeps its own contract (-std=c99 -ffp-contract=off: the same W8I bits at every optimisation level,
# docs/onnx.md and issue #164); the detectors' kernels use fused multiply-adds when ARCH allows them.

CC ?= gcc
OUT ?= build
OPT ?= -O3
ARCH ?= -march=x86-64-v3
WARN = -std=c99 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes
RT = ../runtime
SRC = vocr_desk.c vocr_desk_page.c vocr_tsd.c vocr_tsd_page.c vocr_tsd_resize.c vocr_tsd_refine.c vocr_tsd_post.c \
      $(RT)/vocr.c $(RT)/vocr_lines.c $(RT)/vocr_charbox.c
HDR = vocr_desk.h vocr_tsd.h vocr_tsd_resize.h vocr_tsd_refine.h vocr_tsd_post.h
CFLAGS_ALL = $(WARN) -ffp-contract=off $(OPT) $(ARCH) -I$(RT) -I. $(EXTRA)

all: $(OUT)/libvocrdesk.so $(OUT)/vocr_desk_cli $(OUT)/vocr_tsd_bench $(OUT)/libvocrtsdr.so

$(OUT):
	mkdir -p $(OUT)

$(OUT)/libvocrdesk.so: $(SRC) $(HDR) | $(OUT)
	$(CC) $(CFLAGS_ALL) -fPIC -shared -o $@ $(SRC) -lm -lpthread

# textspot-desktop's box refinement alone (tools/tsdesk_refine.CRefine, tests/test_tsdesk_refine.py)
$(OUT)/libvocrtsdr.so: vocr_tsd_refine.c vocr_tsd_refine.h | $(OUT)
	$(CC) $(CFLAGS_ALL) -fPIC -shared -o $@ vocr_tsd_refine.c -lm

$(OUT)/vocr_desk_cli: vocr_desk_cli.c $(SRC) $(HDR) | $(OUT)
	$(CC) $(CFLAGS_ALL) -o $@ vocr_desk_cli.c $(SRC) -lm -lpthread

$(OUT)/vocr_tsd_bench: vocr_tsd_bench.c $(SRC) $(HDR) | $(OUT)
	$(CC) $(CFLAGS_ALL) -o $@ vocr_tsd_bench.c $(SRC) -lm -lpthread

clean:
	rm -rf $(OUT)

.PHONY: all clean
