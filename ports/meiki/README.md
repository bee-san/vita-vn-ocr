# Existing Meiki OCR on Vita

This binds and converts rtr46's existing horizontal Meiki recognizer and its
author-trained 320x192 text detector. No new recognizer or detector is trained.
The overlay defaults to recognizing one selected horizontal line; optional
`meiki_layout = dialogue_box` detects line crops inside a selected region.
The vertical recognizer is omitted.

The standalone test app and optional isolated-module binding are built locally.
Game and Adrenaline integration and physical-console memory/speed are still unverified.
After repeated console crashes, the user requested local testing followed by
manual device tests. The PC controller was stopped.

## Measured locally

The shared, hand-checked real screenshot sets contain 206 Vita lines and
245 PSP lines. Using upstream preprocessing and horizontal postprocessing:

| Recognizer | Vita character error | PSP character error |
|---|---:|---:|
| Original Meiki ONNX | 37 / 2,564 (1.443%) | 92 / 2,510 (3.665%) |
| Prepared, weight-only int8 MNN | 36 / 2,564 (1.404%) | 89 / 2,510 (3.546%) |
| Same recognizer on ARM in Vita3K | 37 / 2,564 (1.443%) | 92 / 2,510 (3.665%) |

The graph rewrite preserves all 451 decoded texts and Unicode code outputs.
Float MNN conversion preserves all 451 decoded texts. Some floating-point
box/score values differ; this is not a claim of bit-exact raw output.
Int8 changes 22 decoded texts, with both
improvements and regressions. The small aggregate change does not establish
general accuracy improvement. PSP speaker names and furigana remain weaker.
See `results-host.json` for the scoped results.

`verify_runtime.py` additionally checks repeated model load/unload inside
the same bounded binding used by the Vita app, and compares its C decoder
with upstream on all 451 crops. The workspace is 16 MiB; process metadata and
code are separate. The app reserves another 4 MiB for its newlib heap, and
2 MiB of CDRAM for its test display. Its static input is 360 KiB.
The exact final host and emulator measurements are recorded in
`results-runtime.json`; emulator partition sizes and timings are not Vita
measurements.

## Why the allocation changed

The original encoder-class head creates an 11,017,440-byte score tensor.
`prepare.py` computes the same maximum in 128-class chunks, bounding each
such tensor to 322,560 bytes. This is an individual intermediate bound,
not total inference memory.

MNN's low-memory CPU setting retains int8 weight storage and dynamically
quantizes convolution activations to int8. Tensor interfaces and the remaining
operations use float32. The ARM backend differs from the PC on one of the four
emulator test crops, involving PSP punctuation, on both repetitions. All eight
inferences complete and raw outputs repeat exactly within the ARM backend.
The full ARM run completes all 451 crops with 1.443% Vita / 3.665% PSP CER.
Fifteen decoded texts differ from the PC; raw candidates are not identical
between architectures. Every C preprocessing tensor matches upstream exactly,
four native repeats match their raw baselines, and all neural allocations are
released. The metadata break highwater is 2,764,800 bytes inside its separate
4 MiB reservation, with 40,232 bytes live after every job and no growth.
See `results-arm.json`: this full matrix uses the frozen recognizer-only
ABI1 module. The final ABI2 module separately reproduces its 16 baseline
raw outputs while also exercising sequential detection and recognition.

`preprocess.c` provides the framebuffer-to-model binding without allocation.
Its scalar resize, BGR normalization and padding match upstream Meiki bit for
bit on all 451 crops. One reusable float input and resize scratch need 382,080
bytes, separate from the neural workspace. Alpha is ignored. Geometry, failed
reads and cancellation are checked; no new detector or recognizer is trained.
See `results-preprocess.json` and `verify_preprocess.py`.
The direct selected-line overlay path matches the model's host crop error
counts. The existing white-text line finder performs poorly on whole dialogue
regions (40.822% Vita / 65.060% PSP CER); the first Meiki overlay path bypasses
it. Selection margins and adjacent lines still affect recognition accuracy.
The smaller detector uses a 737,280-byte input and the same 13,440-byte
scratch, then releases its model/session so recognition can reuse the same
16 MiB neural arena. The original 960x544 detector fails session creation
in that arena. Complete-region accuracy must be measured through the bounded
overlay adapter, including its eight-crop limit and refused requests.

Its model, packed weights and tensors use an explicit arena through three linker
wrappers. The TopK patch also puts its large sorting scratch into that arena.
The aligned allocation wrappers throw on OOM before MNN operations can
dereference a null buffer; the C boundary catches the exception and reports
the failure after cleanup. MNN must be compiled with
`MNN_ENABLE_EXCEPTIONS=ON`: unwinding through its previous `-fno-exceptions`
frames could strand mutexes. The binding rejects that build before loading.
This avoids overflowing a separate small C++/newlib heap during inference.
The app's pthread linkage explicitly retains weakly referenced initialization
functions from VitaSDK's static archive.

The overlay source we inspected asks for a 64 MiB USER memblock.
The live Shell partition measured only 50 MiB total, so that particular
allocation cannot fit there. Partition free space is different from free
space inside Paf's already reserved heap. This observation does not establish
the cause of every console crash or of another installed OCR configuration.

## Reproduce

Pins, URLs, hashes and licenses are in `dependencies.json`. Build in ignored
`build/` directories; neither weights nor screenshot crops belong in Git.
Install the host requirements with the accompanying hash-locked file.

```sh
uv venv --python 3.11 build/meiki-venv
uv pip install --python build/meiki-venv/bin/python --require-hashes -r ports/meiki/requirements.lock
git clone https://github.com/rtr46/meikiocr build/meiki-source
git -C build/meiki-source checkout 52aa607ecdc1038c7bf390f37cd9a8b306e8a08a
git clone https://github.com/alibaba/MNN build/mnn-source
git -C build/mnn-source checkout ebdada82634300956e08bd4056ecfeb1e4f23b32
git -C build/mnn-source apply ../../ports/meiki/mnn-vita.patch
build/meiki-venv/bin/python ports/meiki/prepare.py
```

The preparation script verifies the source model hash. Model conversion uses
MNN 3.2.0's converter with:

```sh
MNNConvert -f ONNX --modelFile models/meiki/meiki-stream.onnx \
  --MNNModel models/meiki/meiki-stream-int8.mnn --bizCode meiki-vita --weightQuantBits 8
build/meiki-venv/bin/python ports/meiki/normalize_mnn_uuid.py \
  --model recognizer models/meiki/meiki-stream-int8.mnn
```

MNN writes a random UUID into converted-model metadata. The normalization
helper pins that field and verifies the complete expected model hash without
changing weights or inference outputs. The overlay accepts this exact evaluated
conversion.

For optional dialogue-box mode, `prepare_detect.py` verifies the author's
existing 320x192 ONNX, fixes batch one, simplifies it and runs the same pinned
converter. It normalizes metadata and verifies both output hashes. Detector
source, license, pins and tensor shapes are in `detector-dependencies.json`.

```sh
build/meiki-venv/bin/python ports/meiki/prepare_detect.py \
  --source /path/to/meiki.text.detect.v0.1.320x192.onnx \
  --out models/meiki-detector --converter /path/to/MNNConvert
cp models/meiki-detector/meiki-detect-int8.mnn models/meiki/
```

Configure MNN's host and Vita static builds with `MNN_LOW_MEMORY=ON`,
`MNN_ENABLE_EXCEPTIONS=ON`,
`MNN_FORBID_MULTI_THREAD=ON`, `MNN_ARM82=OFF`, `MNN_OPENMP=OFF`,
`MNN_USE_THREAD_POOL=OFF`, and tools/converter/demo/test/shared libraries off.
Enable the converter only in its separate host build.
The Vita build uses the pinned VitaSDK toolchain, ARMv7-A NEON and hard-float.
`build.sh` builds the bounded host probe from a static MNN library.

```sh
ports/meiki/build.sh build/mnn-source build/mnn-host-static/libMNN.a build/meiki-host
build/meiki-venv/bin/python eval/fetch.py --set all
build/meiki-venv/bin/python ports/meiki/evaluate.py \
  --upstream build/meiki-source \
  --original models/meiki/meiki.text.rec.v0.960x32.onnx \
  --prepared models/meiki/meiki-stream.onnx \
  --mnn-probe build/meiki-host/meiki-eval-probe --mnn-model models/meiki/meiki-stream-int8.mnn \
  --output build/meiki-eval
build/meiki-venv/bin/python ports/meiki/verify_runtime.py \
  --upstream build/meiki-source --model models/meiki/meiki-stream-int8.mnn \
  --probe build/meiki-host/meiki-bound-probe \
  --reference build/meiki-eval/meiki-stream-int8.jsonl --out build/meiki-bound
```

The bound probe links static MNN with `--wrap=MNNMemoryAllocAlign`,
`--wrap=MNNMemoryCallocAlign` and `--wrap=MNNMemoryFreeAlign`.
The separate evaluation probe uses the same library without allocator wrapping.
The C binding accepts upstream's BGR float32 NCHW input; the standalone test
bundle uses upstream's resize and padding code. The separate C preprocessor
is checked against it using `verify_preprocess.py`.

```sh
build/meiki-venv/bin/python ports/meiki/bundle.py \
  --upstream build/meiki-source --model models/meiki/meiki-stream-int8.mnn \
  --probe build/meiki-host/meiki-bound-probe --out build/meiki-manual-data
cmake -S ports/meiki/vita-probe -B build/meiki-vita \
  -DMNN_SOURCE="$PWD/build/mnn-source" \
  -DMNN_LIBRARY="$PWD/build/mnn-vita/libMNN.a" \
  -DMEIKI_BUNDLE_DIR="$PWD/build/meiki-manual-data"
cmake --build build/meiki-vita
build/meiki-venv/bin/python ports/meiki/run_vita3k.py \
  --vpk build/meiki-vita/meiki-probe.vpk --out build/meiki-emulator
```

The emulator test uses an isolated config and filesystem under its output
directory. The special auto-start/exit flag exists only in that sandbox.
The physical app waits for CROSS, then runs four dialogue crops twice,
checks text against host MNN, reports peak workspace use, and saves a JSONL log.
COMPLETE requires eight successful, nonempty inferences, identical ARM repeats,
released arenas, and a successful workspace release. PC text matches are
reported separately: the local ARM run completes with 6/8 matches and zero
repeat differences. The accuracy table comes from the full host benchmark.

## Manual console test

1. In VitaShell, back up `ur0:tai/config.txt`. Under the added
   `# OCR live testing` block, comment out these three added lines:

   ```ini
   # ur0:tai/nosleep.skprx
   # ur0:tai/vita_agent_kernel.skprx
   # ur0:tai/vita_agent_loader.suprx
   ```

   Preserve the other plugin lines and section headers, then reboot manually.
   The test app keeps itself awake while open.
2. Install the new `meiki-probe.vpk` in VitaShell, replacing an older version of
   `Meiki OCR probe` if prompted. Its data and model are inside the VPK.
3. Open the app yourself and press CROSS. Report the final COMPLETE/FAIL line,
   displayed peak and any error code. CIRCLE exits.
   Logs are `ux0:data/MeikiProbe/results.jsonl`, `runtime.log` and
   `runtime-errors.log`.

This first test runs as a standalone app. It establishes a clean baseline
before the recognizer is loaded beside a game. No automatic launcher,
pairing or network service is part of the package.

## Isolated overlay module

`module/` builds `meiki-engine.suprx`. It receives a caller-owned, 64-byte
aligned 4 MiB metadata heap and 16 MiB neural workspace through a versioned
C ABI. It creates no default SDK USER heap. Newlib, C++ constructors, MNN
and teardown all execute on a dedicated thread, keeping the SDK's fixed
newlib TLS slot away from Shell's threads. No STL object crosses the boundary.
`run` is synchronous; `stop` must join that thread before module unload and
buffer release. A tainted instance requires unload and reinitialization.

```sh
cmake -S ports/meiki/module -B build/meiki-module \
  -DMNN_SOURCE="$PWD/build/mnn-source" \
  -DMNN_LIBRARY="$PWD/build/mnn-vita/libMNN.a"
cmake --build build/meiki-module
```

The Shell adapter borrows the two heaps and preprocessing buffers from Paf,
loads only the pinned, checksum-verified model on a manual OCR request, and
unloads before dictionary lookup. Code, thread stacks, capture/result buffers
and fragmentation are additional to these reservations. Available Paf space
and coexistence with a game must be measured on hardware.

`results-module.json` records the final ABI2 module: two fresh healthy loads
complete four detector jobs, twelve detector-derived recognitions and sixteen
baseline recognitions. All stop/unload and caller-buffer releases succeed.
A third load with the larger detector returns an allocation error, refuses
reuse, then stops/unloads before its buffers are released. The module's ELF
sections occupy 1,777,240 bytes and its engine stack reserves 128 KiB, separate
from the caller buffers; page and kernel overhead remain unmeasured.
`results-detector-preprocess.json` checks 177 real regions and ten additional
geometry/stride cases against upstream without tensor differences.

`verify_failures.py` exercises sampled model allocations, C++ exceptions,
metadata exceptions, repeated missing/invalid models and recovery using the
same host pool. This is a bounded fault test of the pinned recognizer, not an
exhaustive proof for every MNN model or metadata failure.

## Licenses

The project and arena binding are GPL-3.0-or-later. The decoder adaptation and
Meiki/MNN code are Apache-2.0; model weights and their conversion are LGPL-3.0.
The resize adaptation retains the OpenCV contributors' BSD notices.
The private test display's M+ 1p font is OFL-1.1, with its license in the VPK.
The private VPK contains game screenshot crops: keep it local, and do not
publish it. A public model release must include the model's license and the
pinned original source and conversion instructions.
