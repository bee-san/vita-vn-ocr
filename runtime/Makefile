# runtime/Makefile - host builds, sanitizers and ARM cross builds of the vocr runtime.
#
#   make                 build/vocr_cli and build/libvocr.so (gcc, -O2)
#   make check           gcc + clang -Werror builds, ASan/UBSan CLIs, C unit tests
#   make arm             static ARMv7 (Cortex-A9) CLIs: NEON and scalar
#   make size            code size of vocr.o for the ARM target
#
# Tools can be overridden: CC, CLANG, ARM_CC (default: the Arm GNU toolchain prefix
# arm-none-linux-gnueabihf-, see scripts/fetch_arm_tools.sh), QEMU.

ifeq ($(origin CC),default)
CC = gcc
endif
CLANG ?= clang
OUT ?= build
CSTD = -std=c99 -ffp-contract=off
WARN = -Wall -Wextra -Werror -Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
       -Wcast-align -Wpointer-arith
OPT ?= -O2
SAN = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ARM_CC ?= arm-none-linux-gnueabihf-gcc
ARM_FLAGS = -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -mtune=cortex-a9
QEMU ?= qemu-arm

SRC = vocr.c vocr.h
CLI = tools/vocr_cli.c

.PHONY: all check arm size clean

all: $(OUT)/vocr_cli $(OUT)/libvocr.so

$(OUT):
	mkdir -p $(OUT) $(OUT)/arm

$(OUT)/vocr_cli: $(SRC) $(CLI) | $(OUT)
	$(CC) $(CSTD) $(WARN) $(OPT) -o $@ vocr.c $(CLI) -lm

$(OUT)/libvocr.so: $(SRC) | $(OUT)
	$(CC) $(CSTD) $(WARN) $(OPT) -fPIC -shared -o $@ vocr.c -lm

$(OUT)/vocr_cli_clang: $(SRC) $(CLI) | $(OUT)
	$(CLANG) $(CSTD) $(WARN) $(OPT) -o $@ vocr.c $(CLI) -lm

$(OUT)/vocr_cli_san: $(SRC) $(CLI) | $(OUT)
	$(CC) $(CSTD) $(WARN) $(SAN) -o $@ vocr.c $(CLI) -lm

$(OUT)/vocr_cli_clang_san: $(SRC) $(CLI) | $(OUT)
	$(CLANG) $(CSTD) $(WARN) $(SAN) -o $@ vocr.c $(CLI) -lm

$(OUT)/test_vocr: $(SRC) tests/test_vocr.c | $(OUT)
	$(CC) $(CSTD) $(WARN) $(SAN) -o $@ vocr.c tests/test_vocr.c -lm

$(OUT)/test_vocr_clang: $(SRC) tests/test_vocr.c | $(OUT)
	$(CLANG) $(CSTD) $(WARN) $(SAN) -o $@ vocr.c tests/test_vocr.c -lm

$(OUT)/test_vocr_neon_off: $(SRC) tests/test_vocr.c | $(OUT)
	$(CC) $(CSTD) $(WARN) $(OPT) -DVOCR_NO_NEON -o $@ vocr.c tests/test_vocr.c -lm

# SAN_ENV: e.g. LD_LIBRARY_PATH=... when gcc's libasan is not installed system-wide
check: $(OUT)/vocr_cli $(OUT)/libvocr.so $(OUT)/vocr_cli_clang $(OUT)/vocr_cli_san \
       $(OUT)/vocr_cli_clang_san $(OUT)/test_vocr $(OUT)/test_vocr_clang
	$(SAN_ENV) ./$(OUT)/test_vocr
	./$(OUT)/test_vocr_clang

arm: $(OUT)/arm/vocr_cli_neon $(OUT)/arm/vocr_cli_scalar

$(OUT)/arm/vocr_cli_neon: $(SRC) $(CLI) | $(OUT)
	$(ARM_CC) $(CSTD) $(WARN) $(ARM_FLAGS) -static -o $@ vocr.c $(CLI) -lm

$(OUT)/arm/vocr_cli_scalar: $(SRC) $(CLI) | $(OUT)
	$(ARM_CC) $(CSTD) $(WARN) $(ARM_FLAGS) -DVOCR_NO_NEON -static -o $@ vocr.c $(CLI) -lm

size: | $(OUT)
	$(ARM_CC) $(CSTD) $(WARN) $(ARM_FLAGS) -c -o $(OUT)/arm/vocr_neon.o vocr.c
	$(ARM_CC) $(CSTD) $(WARN) $(ARM_FLAGS) -DVOCR_NO_NEON -c -o $(OUT)/arm/vocr_scalar.o vocr.c
	$(ARM_CC:gcc=size) $(OUT)/arm/vocr_neon.o $(OUT)/arm/vocr_scalar.o

clean:
	rm -rf $(OUT)
