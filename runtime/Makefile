# runtime/Makefile - host checks of the C code in runtime/.
#
#   make check      the line finder's checks (check-lines, in vocr_lines.mk)
#
# CI (ci/runtime.sh) runs `make -C runtime check CC=... CFLAGS=... LDFLAGS=... PYTHON=...` and
# expects test programs built with $(CC) $(CFLAGS) ... $(LDFLAGS): flags given on the command
# line are kept and only extended. Without them the defaults below build with ASan + UBSan.
# SAN_ENV prefixes the commands that run sanitized programs, for example when gcc's libasan is
# not on the loader path:  make check SAN_ENV="env LD_LIBRARY_PATH=/path/to/gcc/lib"
# The recognizer's Makefile keeps `include vocr_lines.mk` and makes `check` depend on check-lines.

ifeq ($(origin CC),default)
CC = gcc
endif
PYTHON ?= python3
OUT ?= build
SAN = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ifeq ($(origin CFLAGS),undefined)
CFLAGS = -O2 -g $(SAN)
endif
ifeq ($(origin LDFLAGS),undefined)
LDFLAGS = $(SAN)
endif
SAN_ENV ?=

.PHONY: check clean

check: check-lines

include vocr_lines.mk

clean:
	rm -rf $(OUT)
