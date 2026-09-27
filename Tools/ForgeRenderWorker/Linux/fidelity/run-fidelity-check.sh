#!/usr/bin/env bash
#
# Compare a worker frame (CUDA, or --backend cpu-emulation) with the shipping
# Metal shader source run on the CPU.
#
#   run-fidelity-check.sh UNIFORMS.bin FRAME.png [MAX_MEAN_ABS] [JSON_OUT]
#
# UNIFORMS.bin comes from `ForgeRenderWorker --dump-uniforms DIR` for the same
# frame. The oracle (MandelNewtonCPUReference.cpp) compiles
# CyanescentForge/Shaders/ForgeMandelNewton.h and ForgePost.h unmodified,
# renders the frame from those exact uniforms, and the two PNGs are compared.
# Default tolerance: mean absolute difference <= 0.01 (0..1 scale), PSNR >= 35 dB.
# JSON_OUT (optional) receives the metrics as one JSON line.
#
# The oracle is independent of the CUDA build: it uses
# Tools/CPUReference/ForgeCPUCompat.h, not cuda/ForgeCUDACompat.cuh.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX="$(dirname "$HERE")"
ROOT="$(cd "$LINUX/../../.." && pwd)"
BUILD="${FORGE_LINUX_BUILD:-$ROOT/.build/linux-cuda-$(uname -m)}"
CXX="${CXX:-c++}"
UNIFORMS="$1"
FRAME="$2"
LIMIT="${3:-0.01}"
JSON_OUT="${4:-}"

mkdir -p "$BUILD"
ORACLE="$BUILD/MandelNewtonCPUReference"
OPENMP=(-fopenmp)
echo 'int main(){}' | "$CXX" -x c++ -fopenmp - -o /dev/null 2>/dev/null || OPENMP=()
"$CXX" -std=c++17 -O2 -ffp-contract=off -fno-fast-math "${OPENMP[@]}" \
  -o "$ORACLE" "$HERE/MandelNewtonCPUReference.cpp" "$LINUX/src/PNGWriter.cpp" -lz

REFERENCE="${FRAME%.png}_cpu_reference.png"
"$ORACLE" "$UNIFORMS" "$REFERENCE"
JSON_ARGS=()
[ -n "$JSON_OUT" ] && JSON_ARGS=(--json "$JSON_OUT")
python3 "$LINUX/tools/png_inspect.py" compare "$FRAME" "$REFERENCE" --max-mean-abs "$LIMIT" --min-psnr 35 "${JSON_ARGS[@]}"
