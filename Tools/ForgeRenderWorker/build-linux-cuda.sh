#!/usr/bin/env bash
#
# Build the Linux CUDA ForgeRenderWorker (headless NVIDIA), V0.1.
#
#   ./Tools/ForgeRenderWorker/build-linux-cuda.sh [OUTPUT_DIR]
#
# Output (default .build/linux-cuda-<arch>/):
#   ForgeRenderWorker        the worker (CUDA runtime linked statically; needs
#                            only the NVIDIA driver's libcuda.so.1 at run time)
#   CUDAWorkerSelfTests      host unit tests (run by this script)
#   cuda-kernels.txt         ptxas register/spill report per kernel and arch
#
# Needs: a C++17 compiler, zlib headers, python3, and the CUDA toolkit (nvcc,
# 11.7 or newer). Nothing Vulkan, OpenGL, EGL or X11 — and the script fails
# if the executable links any of them. Ubuntu:
#   apt-get install build-essential zlib1g-dev python3
#   CUDA: an nvidia/cuda:*-devel image, a RunPod "PyTorch/CUDA devel" image,
#   NVIDIA's cuda-nvcc-12-x + cuda-cudart-dev-12-x packages, or Ubuntu's
#   nvidia-cuda-toolkit.
#
# Environment:
#   NVCC                    nvcc to use (default: PATH, then /usr/local/cuda*/bin)
#   FORGE_CUDA_ARCHS        compute capabilities to build, e.g. "89" or "75 80 86 89 90"
#                           (default: the GPUs nvidia-smi reports, else a portable list;
#                           PTX for the highest is always added for newer GPUs)
#   FORGE_CUDA_HOST_CXX     host compiler for nvcc (default: the first of c++, g++-13 ...
#                           g++-9 that this nvcc accepts)
#   FORGE_BUILD_WITHOUT_CUDA=1  build without nvcc: CPU-emulation backend only (CI/dev)
#   CXX                     host compiler for the portable sources (default c++)
#   FORGE_WORKER_SOURCE_REVISION  revision string to embed (default: git; for builds without .git)
#
# The macOS worker is built by build.sh and is not affected by this script.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LINUX="$ROOT/Tools/ForgeRenderWorker/Linux"
OUT="${1:-$ROOT/.build/linux-cuda-$(uname -m)}"
CXX="${CXX:-c++}"
WITHOUT_CUDA="${FORGE_BUILD_WITHOUT_CUDA:-0}"

say() { printf '%s\n' "$*"; }
die() { printf 'build-linux-cuda.sh: %s\n' "$*" >&2; exit 1; }

command -v "$CXX" >/dev/null 2>&1 || die "no C++ compiler '$CXX' (apt-get install build-essential)"
command -v python3 >/dev/null 2>&1 || die "python3 is required (apt-get install python3)"
printf '#include <zlib.h>\nint main(){return zlibVersion()[0]==0;}\n' | "$CXX" -x c++ - -lz -o /dev/null 2>/dev/null \
  || die "zlib development files not found (apt-get install zlib1g-dev)"

# --- nvcc ---------------------------------------------------------------------
NVCC="${NVCC:-}"
if [ "$WITHOUT_CUDA" != "1" ]; then
  if [ -z "$NVCC" ]; then
    NVCC="$(command -v nvcc 2>/dev/null || true)"
  fi
  if [ -z "$NVCC" ]; then
    for candidate in /usr/local/cuda/bin/nvcc $(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r); do
      if [ -x "$candidate" ]; then NVCC="$candidate"; break; fi
    done
  fi
  [ -n "$NVCC" ] && [ -x "$NVCC" ] || die "nvcc not found. Use a CUDA *devel* image, install the CUDA toolkit \
(e.g. apt-get install cuda-nvcc-12-4 cuda-cudart-dev-12-4 from NVIDIA's repository), set NVCC=/path/to/nvcc, \
or build the CPU-emulation-only worker with FORGE_BUILD_WITHOUT_CUDA=1."
  NVCC_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  NVCC_MAJOR="${NVCC_VERSION%%.*}"; NVCC_MINOR="${NVCC_VERSION##*.}"
  if [ "$NVCC_MAJOR" -lt 11 ] || { [ "$NVCC_MAJOR" -eq 11 ] && [ "$NVCC_MINOR" -lt 7 ]; }; then
    die "nvcc $NVCC_VERSION is too old; CUDA 11.7 or newer is required (__grid_constant__)."
  fi
fi

mkdir -p "$OUT/obj"
OUT="$(cd "$OUT" && pwd)"
SCRATCH="$OUT/obj"

REVISION="${FORGE_WORKER_SOURCE_REVISION:-}"
if [ -z "$REVISION" ]; then
  REVISION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [ -n "$(git -C "$ROOT" status --porcelain -- CyanescentForge/Shaders Tools/ForgeRenderWorker/Linux 2>/dev/null)" ]; then
    REVISION="$REVISION-modified"
  fi
fi

# --- 1. Layout: every ForgeShaderTypes.h field at the host's (Metal) offset --
LAYOUT="$SCRATCH/ForgeUniformLayoutAsserts.h"
python3 "$LINUX/tools/check_cuda_layout.py" "$CXX" "$SCRATCH" "$LAYOUT"

# --- 2. Portable sources + CPU emulation (host compiler) ---------------------
# -ffp-contract=off: Swift never fuses multiply-add, and the frame state must
# match the macOS worker bit for bit. No -ffast-math, no -march: portable.
CXXFLAGS=(-std=c++17 -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra -Wno-unused-parameter
          -Wno-missing-field-initializers -pthread "-DFORGE_WORKER_SOURCE_REVISION=\"$REVISION\""
          "-DFORGE_LAYOUT_ASSERTS=\"$LAYOUT\"")
objects=()
for src in "$LINUX"/src/*.cpp "$LINUX"/cuda/CPUEmulationBackend.cpp; do
  obj="$SCRATCH/$(basename "${src%.*}").o"
  "$CXX" "${CXXFLAGS[@]}" -c "$src" -o "$obj"
  objects+=("$obj")
done

# --- 3. CUDA backend (nvcc) --------------------------------------------------
if [ "$WITHOUT_CUDA" = "1" ]; then
  "$CXX" "${CXXFLAGS[@]}" -c "$LINUX/cuda/CUDAUnavailable.cpp" -o "$SCRATCH/CUDAUnavailable.o"
  objects+=("$SCRATCH/CUDAUnavailable.o")
  "$CXX" "${objects[@]}" -pthread -lz -o "$OUT/ForgeRenderWorker"
  ARCH_LIST="none (built without CUDA)"
  say "Built WITHOUT CUDA (FORGE_BUILD_WITHOUT_CUDA=1): only --backend cpu-emulation can render."
else
  # A host compiler this nvcc accepts (each CUDA release caps the gcc version).
  probe="$SCRATCH/probe.cu"
  printf '__global__ void k(int *p) { p[0] = 1; }\nint main() { return 0; }\n' > "$probe"
  HOST_CXX="${FORGE_CUDA_HOST_CXX:-}"
  if [ -z "$HOST_CXX" ]; then
    for candidate in "$CXX" g++ g++-14 g++-13 g++-12 g++-11 g++-10 g++-9; do
      command -v "$candidate" >/dev/null 2>&1 || continue
      if "$NVCC" -ccbin "$candidate" -c "$probe" -o "$SCRATCH/probe.o" >/dev/null 2>&1; then
        HOST_CXX="$candidate"; break
      fi
    done
  fi
  [ -n "$HOST_CXX" ] || die "nvcc $NVCC_VERSION accepts none of the installed host compilers \
(install the g++ version it supports, e.g. g++-12 for CUDA 12.0-12.3, or set FORGE_CUDA_HOST_CXX)."

  # Architectures: what the machine has, else a portable list; always PTX for
  # the newest so later GPUs can JIT it.
  supported="$("$NVCC" --list-gpu-arch 2>/dev/null | sed -n 's/^compute_\([0-9]*\)$/\1/p' | tr '\n' ' ')"
  archs="${FORGE_CUDA_ARCHS:-}"
  arch_source="FORGE_CUDA_ARCHS"
  if [ -z "$archs" ] && command -v nvidia-smi >/dev/null 2>&1; then
    archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. ' | sort -u | tr '\n' ' ' || true)"
    arch_source="this machine's GPUs (nvidia-smi)"
  fi
  if [ -z "${archs// /}" ]; then
    archs=""
    for a in 75 80 86 89 90 100 120; do
      if [ -z "$supported" ]; then
        [ "$a" -le 90 ] && archs="$archs $a"   # nvcc without --list-gpu-arch: CUDA 11.x-era list
      else
        case " $supported " in *" $a "*) archs="$archs $a" ;; esac
      fi
    done
    arch_source="portable list (no GPU detected)"
  fi
  GENCODE=()
  ARCH_LIST=""
  highest=0
  for a in $archs; do
    [ "$a" -ge 70 ] || die "compute capability $a is below the minimum 7.0"
    if [ -n "$supported" ]; then
      case " $supported " in *" $a "*) ;; *) die "nvcc $NVCC_VERSION cannot target sm_$a (it supports: $supported); \
use a newer CUDA toolkit." ;; esac
    fi
    GENCODE+=(-gencode "arch=compute_$a,code=sm_$a")
    ARCH_LIST="$ARCH_LIST sm_$a"
    [ "$a" -gt "$highest" ] && highest="$a"
  done
  GENCODE+=(-gencode "arch=compute_$highest,code=compute_$highest")
  ARCH_LIST="${ARCH_LIST# } compute_$highest"
  say "nvcc $NVCC_VERSION ($NVCC), host compiler $HOST_CXX"
  say "CUDA architectures: $ARCH_LIST — from $arch_source"

  # No --use_fast_math: Metal renders Forge with fastMathEnabled = false.
  NVCCFLAGS=(-std=c++17 -O3 -lineinfo -ccbin "$HOST_CXX" "${GENCODE[@]}"
             -Xcompiler -Wall,-Wno-unused-parameter,-Wno-unused-function
             -Xptxas -v
             "-DFORGE_WORKER_SOURCE_REVISION=\"$REVISION\""
             "-DFORGE_LAYOUT_ASSERTS=\"$LAYOUT\""
             "-DFORGE_CUDA_ARCHS=\"$ARCH_LIST\"")
  started=$(date +%s)
  if ! "$NVCC" "${NVCCFLAGS[@]}" -c "$LINUX/cuda/CUDABackend.cu" -o "$SCRATCH/CUDABackend.o" \
       > "$OUT/cuda-kernels.txt" 2>&1; then
    grep -v 'Program is doing double precision' "$OUT/cuda-kernels.txt" | grep -v '^ptxas info' | tail -40 >&2
    die "nvcc failed (full log: $OUT/cuda-kernels.txt)"
  fi
  say "CUDA kernels compiled in $(( $(date +%s) - started )) s; register report:"
  # One line per accumulate/resolve kernel and architecture.
  awk '/Compiling entry function/ { match($0, /forge(Accumulate(R[0-9]+)?|Resolve)/); k = substr($0, RSTART, RLENGTH);
                                   match($0, /sm_[0-9]+/); a = substr($0, RSTART, RLENGTH) }
       /bytes stack frame/ && k != "" { s = $0 }
       /Used [0-9]+ registers/ && k != "" { match($0, /Used [0-9]+ registers/);
                                           printf "  %-22s %-6s %s;%s\n", k, a, substr($0, RSTART, RLENGTH), s; k = "" }' \
      "$OUT/cuda-kernels.txt" | sed 's/  *bytes stack frame,/ B stack,/; s/bytes spill stores, /B spill st, /; s/bytes spill loads/B spill ld/'
  objects+=("$SCRATCH/CUDABackend.o")
  # nvcc links the CUDA runtime statically (cudart_static): the executable
  # needs only the driver's libcuda.so.1, loaded at run time.
  "$NVCC" -ccbin "$CXX" "${objects[@]}" -lz -o "$OUT/ForgeRenderWorker"
fi

# --- 4. No Vulkan / GL / EGL / X11 / Wayland dependency ------------------------
deps="$(ldd "$OUT/ForgeRenderWorker" 2>/dev/null || true)"
if printf '%s\n' "$deps" | grep -Eiq 'vulkan|libGL|libEGL|libGLX|libX11|libxcb|wayland|libnvidia-gl'; then
  printf '%s\n' "$deps" >&2
  die "the worker links a graphics/window-system library; the CUDA worker must be headless compute only."
fi
say "Dynamic dependencies (no Vulkan, GL, EGL, X11 or Wayland):"
printf '%s\n' "$deps" | awk '{print "  " $1}'

# --- 5. Host self-tests ----------------------------------------------------------
"$CXX" "${CXXFLAGS[@]}" "$LINUX/tests/CUDAWorkerSelfTests.cpp" "$LINUX/src/FrameExport.cpp" "$LINUX/src/PNGWriter.cpp" \
  -lz -o "$OUT/CUDAWorkerSelfTests"
"$OUT/CUDAWorkerSelfTests"

say "Built Linux $(uname -m) CUDA worker ($REVISION, CUDA code: $ARCH_LIST): $OUT/ForgeRenderWorker"
