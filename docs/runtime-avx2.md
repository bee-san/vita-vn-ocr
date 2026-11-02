# runtime-avx2: AVX2 kernels for x86 builds of the recognizer and textspot8, bit for bit

This is not part of `desktop/` (#175, #176), whose detectors have their own AVX2 + FMA kernels. It is the layer under
it: `runtime/vocr.c`, the recognizer that `desktop/`'s page pipeline and every other x86 build read lines with, and
`runtime/vocr_textspot6.c` (textspot8, Rubi's line finder). No model and no output changes.

The runtime has hand-written NEON for the Vita and nothing for x86, so an x86 build ran plain C: gcc `-O2` leaves
its loops unvectorised, and `-O3 -march=native` vectorises only part of them (`docs/c-speed.md`, "Host speed"). This
change adds AVX2 paths to the hot kernels of those two files. Every output stays the same bit for bit, the arenas and
the API do not change, and every build without AVX2 (the Vita's, the PSP's, plain x86 `-O2`) compiles to `main`'s
exact machine code.

**Measured effect** (one thread, a Ryzen 7 5800X3D, a whole screenshot read the way Rubi reads it; details in "Host
speed" below): about 3-4x faster than the best plain-C build (gcc `-O3 -march=native`) and about 4.5x faster than
clang `-O2`, the spotter 5-6x.

## When the kernels are compiled in

`__AVX2__` defined (gcc and clang: `-mavx2`, or a `-march` that includes AVX2, such as `-march=native` on an AVX2 CPU:
Intel Haswell, AMD Excavator / Zen and later; MSVC: `/arch:AVX2`, not tested), not a NEON build, and no
`-DVOCR_NO_AVX2`. The guard is `VOCR_AVX2` in
`vocr.c` and `TS_AVX2` in `vocr_textspot6.c`, each defined once at the top of its file, as `VOCR_NEON` and `TS_NEON`
are. There is no runtime CPU dispatch: a binary built with `-mavx2` needs an AVX2 CPU, and one built without it never
runs the kernels.

`desktop/` (#175, #176) builds `runtime/vocr.c` with `-march=x86-64-v3` by default, so its page pipeline reads lines
with these kernels; `make -C desktop ARCH=` or `EXTRA=-DVOCR_NO_AVX2` builds it without them. The kernels keep no
state, so its one arena per thread is still enough. `desktop/` does not link `vocr_textspot6.c`, and its detectors'
own AVX2 + FMA kernels are not touched here.

## What changed

### `runtime/vocr.c` (AVX2 builds only)

| kernel | AVX2 form | why the bits are the same |
|---|---|---|
| `mm_i16`, the W8I matrix product (every pointwise layer, the head, the gate and the escape) | 4 positions × 8 outputs per pass. Each pair of weight rows (k, k+1) is interleaved (w[k][0], w[k+1][0], w[k][1], …) and widened to int16 once per pass; the input pair (x[k], x[k+1]) is broadcast, and one `VPMADDWD` per position adds w[k][j]·x[k] + w[k+1][j]·x[k+1] into output j's int32 lane. Then the chunk's float conversion and scale, ReLU (`max` with 0) and the residual add | the int32 sums are exact in any order: `W8I_CHUNK`'s bound (512 × 127 × 32767 < 2^31) covers the pair sums too, since they are partial sums of the same chunk. The float steps are the scalar code's, in its order |
| `k_conv1` (the stem) | 8 channels per register, 4 registers (32 channels) per pass, then 8 at a time; the bias loaded as the starting sums, each tap's input broadcast, ReLU as `max` on the way out | each output is still bias, then + w · x over the taps in the scalar order, every product and every sum rounded on its own (`MULPS` then `ADDPS`, no FMA) |
| `k_dw` (the depthwise layers) | as `k_conv1`'s, the input a vector of channels too | the same |

Whatever the AVX2 loop leaves (fewer than 8 channels) takes the scalar loop that follows it, as before.

### `runtime/vocr_textspot6.c` (AVX2 builds only)

An `#elif TS_AVX2` branch beside `#if TS_NEON` in the existing block-kernel chain, so `produce` and its dispatch do
not change. A block's 8 output pixels are 8 int32 lanes of one register.

| kernel | AVX2 form | why the bits are the same |
|---|---|---|
| 3×3 layers (`K3_BLOCK`: `k3_block_s1`, `k3_block_s2`) | the 9 taps of an input channel as 5 int16 pairs (the last paired with zeros), each through one `VPMADDWD`. The 9 weights are masked (a kernel row outside the input gets 0) and widened in one register, each pair broadcast from it. Not depthwise: 4 output channels per pass share the widened inputs. Stride 2 loads 16 bytes and keeps the even ones (`PSHUFB`) | every product of two int8 codes and every int32 sum is exact, so any order gives the scalar sums; a masked tap adds 0, as the scalar code's masked weight does |
| 1×1 layers (`k1_block`) | the block's input channels widened once into int16 pairs (an odd last one with zeros); 4 output channels per pass, their weights 16 at a time (never read past a weight row), then 2 at a time; the lone output channels past a multiple of 4 one at a time | the same exact int32 sums |
| the requantisation (`rq8`, `store_acc8`, `store_acc8x4`) | for a shift of 32 or more (every channel of a trained blob): bias · m + 2^(sh − 1) + sum · m in int64 (`VPMULDQ` on the even lanes, then the odd), the high word, an arithmetic shift by sh − 32, the clamp to [lo, 127]. Four channels of a full block are packed to int8 together (two `PACKSSDW`, one `PACKSSWB`, one `VPERMD` back to channel order). Any other shift, and a partial block, go through `requant` lane by lane | `requant`'s own sh ≥ 32 branch on 8 lanes: `bq_of` + sum · m in int64, floor by 2^32 (the high word), floor by 2^(sh − 32), the clamp. The packs saturate within a range the clamp has already applied |

## Rules for later edits

These are what the bit identity rests on. A change to the kernels has to keep each of them.

- **Float order.** `k_conv1`, `k_dw` and `mm_i16`'s float steps add in the scalar code's order, one rounding per
  operation. No FMA intrinsics (`_mm256_fmadd_ps`), and `-ffp-contract=off` stays in every build (the Makefile's
  `WARN`, `ci/runtime.sh`).
- **Integer sums.** Reordering the int32 sums is safe only while they stay exact: `W8I_CHUNK`'s bound in `vocr.c`, and
  in the finder at most `MAXCH` (64) input channels × 9 taps of int8 × int8 products.
- **Stride-2 reads.** The stride-2 load reads 16 bytes for 8 pixels, past the last pixel of a block; the plane's slack
  covers it, as it does NEON's `vld2`. A change to the plane layout or its slack must keep at least that much.
- **Plain builds unchanged.** All AVX2 code stays inside `VOCR_AVX2` / `TS_AVX2`. Without them, `vocr.o` and
  `vocr_textspot6.o` must stay byte-identical to `main`'s (checked below for x86 `-O2` and Cortex-A9 NEON).
- **The checks.** On x86, `make check` builds with `AVX2=-mavx2` (the default when the CPU has AVX2; `ci/runtime.sh`
  passes it): `check-vocr` runs the unit tests and every fixture parity (default, stride 16 and 12, attention)
  through AVX2 builds, and `check-textspot6` compiles `vocr_textspot6.c` strictly with it and runs
  `tests/textspot6_parity.py` through an AVX2 driver. `ci/runtime.sh` also compiles the sources with AVX2 kernels with
  `-mavx2`, warning-free. `vocr.c` also has to build under `desktop/`'s stricter `WARN` (`-Werror -Wshadow
  -Wmissing-prototypes`, `-O3 -march=x86-64-v3`), and `tests/test_desktop.py` reads its page lines through an
  `x86-64-v3` build and checks them against pyvocr's.

## Bit identity

- **Objects.** gcc 13.3 `-O2` on x86-64: `vocr.o` and `vocr_textspot6.o` have `main`'s exact `.text`, and so do the
  `-mavx2 -DVOCR_NO_AVX2` builds against `main`'s `-mavx2` ones. clang 21.1 (zig cc 0.16) for the Cortex-A9 with
  NEON (`-mcpu=cortex_a9 -mfpu=neon -mfloat-abi=hard -O2`): both objects have `main`'s exact `.text`.
- **Outputs.** A hash tool (off-repo; below) hashed every output of every tier on two screenshots: S, N and L on 20
  random line crops per frame in both polarities, once with the head gate off and the full logits, once with Rubi's
  options (top 8, blank logits, conf, gated and escaped counts); and every textspot8 map row, box and block, grey and
  RGBA, at the default and the forced 1× scale. The AVX2 and scalar builds print the same hashes.
- **Texts.** Every build timed below read the same text on both screenshots, in every run.
- **Unit tests.** `test_vocr` passes built with AVX2, without it, and with AVX2 and `-DVOCR_NO_ATTN`.
- **`desktop/`.** `make -C desktop` builds warning-free (gcc 13.3) with the kernels, with `EXTRA=-DVOCR_NO_AVX2`, and
  plain at `OPT=-O2 ARCH=`.

## Host speed

One thread, a Ryzen 7 5800X3D. `ocr_page` (off-repo, below) reads a whole screenshot the way Rubi does: the textspot8
spotter, Normal's greedy read with its head gate and two-level head, the re-decode, the beam with the 1 MB LM, and the
cascade to Large. Two Vita screenshots (960×544 native, 1200×680 upscaled), v0.5.0's blobs. Median of 30 warm
repeats, 3 rounds; ms per screenshot, the spotter's share in brackets.

gcc 13.3 (Ubuntu 24.04 under WSL 2):

| | plain C `-O2` | plain C `-O3 -march=native` | AVX2 `-O2 -mavx2` | AVX2 `-O3 -march=native` |
|---|---:|---:|---:|---:|
| 960×544 | 135.1 (30.1) | 41.9-45.9 (18.6) | 15.0-16.3 (3.5) | 14.2-15.3 (3.2) |
| 1200×680 | 82.4 (43.7) | 37.5 (26.9) | 10.3 (5.1) | 9.4 (4.6) |

clang 21.1 (zig cc 0.16, Windows 11): plain C `-O2` 63.1 (16.9) and 45.3 (26.7); AVX2 `-O2 -mavx2` 14.1 (3.8) and
10.4 (5.9). clang's `-O3 -march=native` without the kernels is no faster than its `-O2` (62.8 and 45.0).

These are two screenshots on one machine. The repo's own host timing, `eval/c_speed_host_ms.py`, now also builds
`-O2 -mavx2` and `-O3 -march=native -DVOCR_NO_AVX2` on x86, so the figures can be redone on the development lines and
regions next to `docs/c-speed.md`'s.

## Memory

No new allocations and no change to any arena: the kernels use registers and the existing buffers. Code size, gcc 13.3
`-O2 -mavx2`, `.text`: `vocr.o` 32,381 → 33,053 B, `vocr_textspot6.o` 31,140 → 36,260 B. The console builds are
unchanged (above).

## Reproduce

```sh
make -C runtime check AVX2=-mavx2               # x86 with AVX2; AVX2= turns the AVX2 runs off
python eval/c_speed_host_ms.py --base <main's runtime/> --new runtime --models <the v0.5.0 blobs> --sub sub24.bin \
    --regions dev_regions.bin --out host_ms.json   # adds O2avx2 and O3n_noavx2 on x86
```

`ocr_page` and the hash tool (`parity`) are on the `bench-avx2` branch of github.com/Mansive/vita-vn-ocr (`bench/`),
not in this repo; the screenshots are copyrighted and not committed.

## Not measured

- How much each technique contributes: only whole builds were timed.
- `tests/parity.py` and `tests/textspot6_parity.py` locally (no PyTorch on the measuring machine); CI runs both
  through the AVX2 builds.
- The development lines and regions (`eval/c_speed_host_ms.py`, `vita/speed_parity.py`): not available to the author.
- ONNX on this machine. In `docs/onnx.md` ("Speed", a different host) the exact ONNX files ran about as fast per line
  as gcc `-O3 -march=native`, so these kernels should be ahead of them; that is an inference across two machines.
  `docs/desktop.md` puts VOCR6.2-L at 6.5 ms per line in exact ONNX, 6.3 ms in C at `-O3 -march=native` and 1.8 ms in
  fast ONNX (W8F arithmetic, not these bits); per-line times with these kernels were not taken.
- `desktop/`'s page times with these kernels: its detector blobs are on releases not downloaded here. Its timings in
  `docs/desktop.md` were taken before this change.
- MSVC, AVX-512, and CPUs other than Zen 3.
