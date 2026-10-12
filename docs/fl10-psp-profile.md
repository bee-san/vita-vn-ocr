# Released FL10 / VOCR2 on the PSP emulator

The released VOCR2 weights pass 192 generated-input comparisons against the host
runtime in PPSSPP. The existing opt-in VFPU float path with tile 4 uses a 118,432-byte
arena and 47.29% less emulated time than scalar integer inference with tile 4.
This qualifies an existing execution option with the current model; it introduces
no new recognition model or kernel optimization and changes no deployment default.

This is **not a real-screenshot accuracy test or a physical PSP measurement**.
For measured recognition changes, see the CLANNAD comparison linked from
[issue #42](https://github.com/bee-san/vita-vn-ocr/issues/42).

## Fixed measurement

Measured on 2026-10-12. The model is `VOCR2_FL10_w8.vocr` from
[the VOCR2.1 weight release](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.3-vocr2.1),
SHA-256 `d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766`.
It has 3,501 output classes including the CTC blank and a 657,664-byte blob.
The runtime source is based on `62676de1e5ec976b952dab3a3845549336e1dd7c`.

- PPSSPPHeadless 1.20.4, distribution package `ppsspp 1.20.4-4.1`, JIT,
  emulated CPU 333 MHz, 24 MB user partition.
- PSP toolchain: the repository's SHA-pinned pspdev `v20261001`, GCC 15.2.0,
  archive SHA-256 `a86efe624e770005290919617859e9df24bd8fcd3e364f4587f23ec7364b0acd`.
- Linux x86-64, glibc 2.44, Python 3.11.15, NumPy 2.4.6. The JSON records the
  exact platform, emulator, EBOOT, host library, source and input-pack hashes.
- Twelve procedural grey/RGBA images, including blank inputs, heights 1–64,
  widths 1–1,920, and both inversion settings. These contain no game data.
  Maximum recognition width is 1,024; head tile is 16.
- Four fixed execution profiles, two fresh boots per polarity: 16 boots,
  192 checked cases. Each case has one initial numerical check followed by
  three timed repetitions. Times below weight each boot by its CTC-frame count.
- Every case must match its mode's host text and float-logit CRC. This does
  not assert that integer and float modes have identical outputs to each other.
  Missing cases, mismatched execution settings or model allocation, and an
  exhausted worker stack fail the run.

The complete small derived result is
[`eval/fl10_psp_profile_20261012.json`](../eval/fl10_psp_profile_20261012.json).
Repeated boots for each profile and polarity had identical per-case timing.
The ranges below cover both polarities and both boots, not hardware uncertainty.

| PSP build / mode | Tile | Arena bytes | Emulated ms / CTC frame | Range | Numerical matches |
|---|---:|---:|---:|---:|---:|
| Scalar / W8I | 4 | 124,832 | 13.587951 | 13.587949–13.587954 | 48 / 48 |
| Scalar / W8F | 4 | 118,432 | 10.589078 | 10.588662–10.589494 | 48 / 48 |
| VFPU / W8F | 4 | 118,432 | 7.162195 | 7.162191–7.162198 | 48 / 48 |
| VFPU / W8F | 16 | 353,536 | 6.285608 | 6.285602–6.285614 | 48 / 48 |

With tile 4, VFPU W8F takes **47.29% less emulated time** than scalar W8I,
with **6,400 fewer arena bytes**. Against scalar W8F, VFPU W8F takes **32.36%
less emulated time** with the same arena. Tile 16 saves a further 12.24% of
emulated time but adds 235,104 arena bytes. Tile 4 is the memory-first choice
to investigate on hardware. No physical-console speedup follows from these figures.

## Allocation and stack limits

All four profiles passed with a **4,096-byte worker stack**. The emulator reported
2,992 bytes used, leaving 1,104 bytes of observed headroom. This is a qualification
of these fixed model/input/build combinations; the test application's default
16 KB stack is unchanged. The model and runtime context add 3,756 bytes.

`psp-size` reported 18,219 bytes of recognizer code for scalar and 18,759 bytes for
VFPU, with zero object data/BSS. The blob, VFPU tile-4 arena, context, worker stack
and recognizer code total **802,707 bytes**. This is a component sum, not total
plugin RAM: it excludes the line finder, framebuffer/crop storage, caller stack,
allocator overhead, application code and system libraries.

The test app additionally allocates a 1,806,516-byte full-logit output buffer for
numerical checking and up to 491,520 bytes for its largest generated input.
A streaming overlay does not need the full-logit output buffer. Neither this app
nor the reported free-memory counters measure spare RAM next to a running VN.

## Reproduce

Use the repository's pinned Python dependencies and PSP toolchain. Keep model
blobs, EBOOTs, packs and detailed predictions outside Git. Obtain the exact model
from the linked release and verify its hash before running.

```sh
psp/scripts/fetch_pspdev.sh /absolute/path/to/psp-tools
export PSPDEV=/absolute/path/to/psp-tools/pspdev
mkdir -p build/fl10-psp-profile
make -C psp clean
make -C psp VFPU=0
cp psp/EBOOT.PBP build/fl10-psp-profile/EBOOT_scalar.PBP
make -C psp clean
make -C psp VFPU=1
cp psp/EBOOT.PBP build/fl10-psp-profile/EBOOT_vfpu.PBP
gcc -std=c99 -O2 -shared -fPIC -ffp-contract=off -DVOCR_NO_NEON \
  -Wall -Wextra -Werror runtime/vocr.c \
  -o build/fl10-psp-profile/libvocr-scalar.so -lm
python psp/fl10_profile.py \
  --model /absolute/path/to/VOCR2_FL10_w8.vocr \
  --model-sha256 d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766 \
  --scalar-eboot build/fl10-psp-profile/EBOOT_scalar.PBP \
  --vfpu-eboot build/fl10-psp-profile/EBOOT_vfpu.PBP \
  --library build/fl10-psp-profile/libvocr-scalar.so \
  --ppsspp /absolute/path/to/PPSSPPHeadless \
  --out build/fl10-psp-profile/new-run
python -m pytest -q tests/test_psp.py tests/test_psp_profile.py
```

`fl10_profile.py` refuses to overwrite an output directory. It pins the inputs before
execution and checks for changes afterward. Its generated pack is reproducible
from the recorded seed; private per-boot files retain the actual predictions and
emulator log. The public JSON retains derived counts, hashes, timings and allocation
measurements. The driver uses PPSSPP's explicit `host0:` mount via `--root` and
does not replace the caller's `HOME`. Emulated copies use a `vocr-emulator-app/` directory:
PPSSPP redirects paths containing `PSP/GAME` to its real `ms0:` mount. The bundle
entry point was also exercised with these generated inputs; all 12 text and
logit checks passed with a 118,432-byte arena and 2,992-byte observed stack use.
An existing copy is preserved; use a fresh `--keep` directory for another run.

Local full-suite verification after the profiler filename fix passed 1,269 tests
with one optional local-data test skipped. The final focused PSP suite passed
28 tests after the host-mount adjustment; Ruff and source/result audits passed.

PPSSPP does not model the PSP's cache and pipeline timing. The VFPU option still
needs numerical checking on a physical PSP, especially its handling of denormals.
No physical-console run or plugin integration is claimed here.
