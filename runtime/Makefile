# runtime/Makefile - host builds, tests and ARM cross builds of the vocr runtime.
#
#   make                  build/vocr_cli and build/libvocr.so (release flags; vocr.c + vocr_lm.c + vocr_post.c)
#   make check            the line finder's checks (check-lines, vocr_lines.mk), the stroke split's
#                         (check-split, vocr_lines_split.mk: strict compile, its C self-test and its
#                         parity with synth/lines_split.py), the text detector's
#                         (check-textdet, vocr_textdet.mk: strict compile, its C self-test and its
#                         parity with synth/textdet_twin.py), the text-presence gate's (check-textgate,
#                         vocr_textgate.mk: the same against eval/textgate.py), the vertical column finder's
#                         (check-vertical, vocr_vertical.mk: strict compile, self-test, parity with
#                         synth/vertical_twin.py), the full-frame text spotter's (check-textspot,
#                         vocr_textspot.mk: strict compile, its C self-test and its parity with
#                         synth/textspot_twin.py), the spotter of VTSP version 1 and 2 (d2s) blobs'
#                         (check-textspot2, vocr_textspot2.mk: strict compile, its C self-test, its parity
#                         with synth/textspot2_twin.py and, for version 1, with vocr_textspot.c), the
#                         spotter of VTSP version 1, 2 and 3 (multi-scale) blobs' (check-textspot3,
#                         vocr_textspot3.mk: strict compile, its C self-test, its parity with
#                         synth/textspot3_twin.py and, for versions 1 and 2, with vocr_textspot2.c), the counter
#                         post-processing's (check-post, vocr_post.mk: strict compile, its C self-test and its
#                         parity with eval/vocr_post.py), the edge continuation's (check-widen,
#                         vocr_lines_widen.mk: strict compile, its C self-test and its parity with
#                         synth/lines_widen.py), the per-character positions' (check-charbox,
#                         vocr_charbox.mk: strict compile, its C self-test and its parity with
#                         eval/charbox.py), then the
#                         recognizer's: C unit tests and the PyTorch parity test through a
#                         vocr_cli built with $(CC) $(CFLAGS) $(LDFLAGS) (default: ASan + UBSan),
#                         plus the same parity on stride-16 and stride-12 fixture models and on one
#                         with the attention op (docs/vocr6-attn.md)
#   make check-all        check with gcc and clang, plus strict release builds of both
#   make arm              static Cortex-A9 CLIs for qemu-arm: NEON and scalar
#   make check-arm        the C unit tests as Cortex-A9 NEON and scalar builds, under qemu-arm
#   make size             ARM code size of vocr.o (NEON and scalar)
#
# `check` follows CI's contract (ci/runtime.sh): CFLAGS / LDFLAGS given on the command
# line are kept and only extended, and the parity test reads VOCR_CKPT (a train.py
# checkpoint; default: a random-init fixture), VOCR_SHARD (a render.py shard; default:
# fixture crops) and writes VOCR_REPORT. SAN_ENV prefixes sanitized programs, e.g.
# SAN_ENV="env LD_LIBRARY_PATH=/path/to/libasan" when gcc's libasan is not installed.
# ARM_CC defaults to the Arm GNU toolchain (scripts/fetch_arm_tools.sh).

ifeq ($(origin CC),default)
CC = gcc
endif
CLANG ?= clang
PYTHON ?= python3
OUT ?= build
WARN = -std=c99 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -Wshadow \
       -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Wpointer-arith
OPT ?= -O2
SAN = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ifeq ($(origin CFLAGS),undefined)
CFLAGS = -O2 -g $(SAN)
endif
ifeq ($(origin LDFLAGS),undefined)
LDFLAGS = $(SAN)
endif
LDLIBS = -lm
SAN_ENV ?=
VOCR_CKPT ?=
VOCR_SHARD ?=
VOCR_REPORT ?= $(OUT)/parity_report.json
PARITY_ARGS ?=
ARM_CC ?= arm-none-linux-gnueabihf-gcc
ARM_FLAGS = -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard
TOOLS ?= $(abspath scripts/tools)
QEMU ?= $(TOOLS)/bin/qemu-arm

SRC = vocr.c vocr.h vocr_lm.c vocr_lm.h vocr_post.c vocr_post.h
LIBSRC = vocr.c vocr_lm.c vocr_post.c
CLI = tools/vocr_cli.c

.PHONY: all check check-vocr check-all arm check-arm size parity-arm clean

all: $(OUT)/vocr_cli $(OUT)/libvocr.so

$(OUT):
	mkdir -p $(OUT)/arm

# ---- release builds (no sanitizers; libvocr.so is what tests/parity.py loads by default)
$(OUT)/vocr_cli: $(SRC) $(CLI) | $(OUT)
	$(CC) $(WARN) $(OPT) -o $@ $(LIBSRC) $(CLI) $(LDLIBS)

$(OUT)/libvocr.so: $(SRC) | $(OUT)
	$(CC) $(WARN) $(OPT) -fPIC -shared -o $@ $(LIBSRC) $(LDLIBS)

# ---- check programs: $(CC) $(CFLAGS) ... $(LDFLAGS); rebuilt whenever CC or the flags change
#      (CFLAGS are kept and extended with $(WARN), which must come last: -ffp-contract=off)
FLAGS_STAMP = $(OUT)/.check-flags
$(shell mkdir -p $(OUT)/arm; echo '$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS)' | cmp -s - $(FLAGS_STAMP) || \
        echo '$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS)' > $(FLAGS_STAMP))

$(OUT)/test_vocr_check: $(SRC) tests/test_vocr.c $(FLAGS_STAMP) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -o $@ vocr.c tests/test_vocr.c $(LDFLAGS) $(LDLIBS)

$(OUT)/test_vocr_lm_check: $(SRC) tests/test_vocr_lm.c $(FLAGS_STAMP) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -o $@ $(LIBSRC) tests/test_vocr_lm.c $(LDFLAGS) $(LDLIBS)

$(OUT)/vocr_cli_check: $(SRC) $(CLI) $(FLAGS_STAMP) | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -o $@ $(LIBSRC) $(CLI) $(LDFLAGS) $(LDLIBS)

check: check-lines check-lines-fix check-split check-textdet check-textgate check-vertical check-textspot \
       check-textspot2 check-textspot3 check-post check-widen check-charbox check-vocr

include vocr_lines.mk
include vocr_lines_fix.mk
include vocr_lines_split.mk
include vocr_textdet.mk
include vocr_textgate.mk
include vocr_vertical.mk
include vocr_textspot.mk
include vocr_textspot2.mk
include vocr_textspot3.mk
include vocr_post.mk
include vocr_lines_widen.mk
include vocr_charbox.mk

check-vocr: $(OUT)/test_vocr_check $(OUT)/test_vocr_lm_check $(OUT)/vocr_cli_check
	$(SAN_ENV) ./$(OUT)/test_vocr_check
	$(SAN_ENV) ./$(OUT)/test_vocr_lm_check
	$(PYTHON) tests/parity.py $(if $(VOCR_CKPT),--ckpt "$(VOCR_CKPT)",--fixture) \
	    $(if $(VOCR_SHARD),--shard "$(VOCR_SHARD)") $(PARITY_ARGS) \
	    --cli "$(SAN_ENV) ./$(OUT)/vocr_cli_check" --report "$(VOCR_REPORT)"
	for s in 16 12; do \
	  $(PYTHON) tests/parity.py --fixture --stride $$s --n 24 --tile-lines 12 \
	      --cli "$(SAN_ENV) ./$(OUT)/vocr_cli_check" --report "$(OUT)/parity_report_s$$s.json" > /dev/null \
	  || { echo "stride-$$s fixture parity failed: $(OUT)/parity_report_s$$s.json"; exit 1; }; \
	  echo "stride-$$s fixture parity: PARITY_OK"; done
	$(PYTHON) tests/parity.py --fixture --attn 1 --n 24 --tile-lines 12 \
	    --cli "$(SAN_ENV) ./$(OUT)/vocr_cli_check" --report "$(OUT)/parity_report_attn.json" > /dev/null \
	  || { echo "attention fixture parity failed: $(OUT)/parity_report_attn.json"; exit 1; }
	@echo "attention fixture parity: PARITY_OK"

check-all:
	$(MAKE) --no-print-directory clean
	$(MAKE) --no-print-directory CC=gcc all
	$(CLANG) $(WARN) $(OPT) -o $(OUT)/vocr_cli_clang $(LIBSRC) $(CLI) $(LDLIBS)
	$(MAKE) --no-print-directory CC=gcc OUT=$(OUT)/gcc check
	$(MAKE) --no-print-directory CC=$(CLANG) OUT=$(OUT)/clang check SAN_ENV=

# ---- ARM (Cortex-A9, as on the PS Vita), run under qemu-arm
$(OUT)/arm/vocr_cli_neon: $(SRC) $(CLI) | $(OUT)
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -static -o $@ $(LIBSRC) $(CLI) $(LDLIBS)

$(OUT)/arm/vocr_cli_scalar: $(SRC) $(CLI) | $(OUT)
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -DVOCR_NO_NEON -static -o $@ $(LIBSRC) $(CLI) $(LDLIBS)

arm: $(OUT)/arm/vocr_cli_neon $(OUT)/arm/vocr_cli_scalar

$(OUT)/arm/test_vocr_neon: $(SRC) tests/test_vocr.c | $(OUT)
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -static -o $@ vocr.c tests/test_vocr.c $(LDLIBS)

$(OUT)/arm/test_vocr_scalar: $(SRC) tests/test_vocr.c | $(OUT)
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -DVOCR_NO_NEON -static -o $@ vocr.c tests/test_vocr.c $(LDLIBS)

check-arm: $(OUT)/arm/test_vocr_neon $(OUT)/arm/test_vocr_scalar
	$(QEMU) -cpu cortex-a9 ./$(OUT)/arm/test_vocr_neon
	$(QEMU) -cpu cortex-a9 ./$(OUT)/arm/test_vocr_scalar

parity-arm: arm
	for v in neon scalar; do \
	  $(PYTHON) tests/parity.py $(if $(VOCR_CKPT),--ckpt "$(VOCR_CKPT)",--fixture) \
	      $(if $(VOCR_SHARD),--shard "$(VOCR_SHARD)") $(PARITY_ARGS) \
	      --cli "$(QEMU) -cpu cortex-a9 ./$(OUT)/arm/vocr_cli_$$v" --report $(OUT)/arm/parity_$$v.json \
	  || exit 1; done

size: | $(OUT)
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -fstack-usage -c -o $(OUT)/arm/vocr_neon.o vocr.c
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -fstack-usage -c -o $(OUT)/arm/vocr_lm.o vocr_lm.c
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -fstack-usage -c -o $(OUT)/arm/vocr_post.o vocr_post.c
	$(ARM_CC) $(WARN) $(ARM_FLAGS) -fstack-usage -DVOCR_NO_NEON -c -o $(OUT)/arm/vocr_scalar.o vocr.c
	$(ARM_CC:gcc=size) $(OUT)/arm/vocr_neon.o $(OUT)/arm/vocr_scalar.o $(OUT)/arm/vocr_lm.o $(OUT)/arm/vocr_post.o
	sort -t'	' -k2 -n -r $(OUT)/arm/vocr_neon.su | head -5
	sort -t'	' -k2 -n -r $(OUT)/arm/vocr_lm.su | head -5
	sort -t'	' -k2 -n -r $(OUT)/arm/vocr_post.su | head -5

clean:
	rm -rf $(OUT)
