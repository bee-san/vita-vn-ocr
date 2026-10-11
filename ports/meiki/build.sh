#!/usr/bin/env bash
# Build both host probes from the pinned, patched MNN static library.
#   build.sh MNN_SOURCE MNN_LIBRARY OUTPUT_DIRECTORY
set -euo pipefail
[ "$#" -eq 3 ] || { echo "usage: $0 MNN_SOURCE MNN_LIBRARY OUTPUT_DIRECTORY" >&2; exit 2; }
port=$(cd "$(dirname "$0")" && pwd)
source_dir=$(realpath "$1")
library=$(realpath "$2")
out=$(realpath -m "$3")
[ -f "$source_dir/include/MNN/Interpreter.hpp" ] && [ -f "$library" ] || {
  echo "MNN headers or static library missing" >&2
  exit 2
}
mkdir -p "$out"
flags=(-O2 -ffp-contract=off -Wall -Wextra -Werror "-I$port" "-I$source_dir/include")
"${CC:-cc}" -std=c99 "${flags[@]}" -c "$port/arena.c" -o "$out/arena.o"
"${CC:-cc}" -std=c99 "${flags[@]}" -c "$port/decode.c" -o "$out/decode.o"
"${CC:-cc}" -std=c99 "${flags[@]}" -c "$port/model_reader.c" -o "$out/model_reader.o"
"${CXX:-c++}" -std=c++17 "${flags[@]}" "$port/bound_probe.cpp" "$port/runtime.cpp" \
  "$out/arena.o" "$out/decode.o" "$out/model_reader.o" "$library" -pthread -lm -ldl \
  -Wl,--wrap=MNNMemoryAllocAlign -Wl,--wrap=MNNMemoryCallocAlign -Wl,--wrap=MNNMemoryFreeAlign \
  -o "$out/meiki-bound-probe"
"${CXX:-c++}" -std=c++17 "${flags[@]}" "$port/mnn_probe.cpp" "$library" \
  -pthread -lm -ldl -o "$out/meiki-eval-probe"
