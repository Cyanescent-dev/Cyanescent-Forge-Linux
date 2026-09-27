#!/usr/bin/env bash
#
# Validate and benchmark the Linux CUDA ForgeRenderWorker on an NVIDIA GPU
# (written for a fresh RunPod pod; any Linux + NVIDIA driver + CUDA toolkit).
#
#   ./Tools/ForgeRenderWorker/test-cuda-renderer.sh [options]
#
# Steps (everything is logged to cuda-test-results/, then tarred):
#   0. environment: nvidia-smi, driver, CUDA toolkit, libcuda (no Vulkan needed)
#   1. build (build-linux-cuda.sh; the GPU's own architecture)
#   2. --version, --list-gpus
#   3. fidelity, Offline — High, seed 12345:
#        128x72 frames 0 and 300, 320x180 frame 150, rendered with CUDA and
#        compared with (a) the shipping Metal shader source run on the CPU
#        from the same uniforms and (b) the committed reference frames
#        (also 640x360 frame 0, reference only);
#        MAE <= 0.01 and PSNR >= 35 dB required; plus determinism (a repeat
#        render is byte-identical) and frame-state hashes (identical to the
#        committed reference)
#   4. one 1920x1080 High frame, after a warm-up, with timings
#   5. a sequence (default 30 frames of 1920x1080 High) to PNG + MP4, with
#      nvidia-smi sampled every second
#   6. --sweep only: register-cap x block-size variants at 960x540
#
# Options:
#   --quick              steps 0-4 only, fidelity at 128x72 only
#   --sweep              also run step 6
#   --sequence N         sequence length (default 30; 0 skips step 5)
#   --resolution WxH     benchmark/sequence resolution (default 1920x1080)
#   --quality NAME       benchmark/sequence quality (default high)
#   --no-build           use the existing build
#   --compare-metal DIR  also compare with Metal frames rendered on a Mac
#                        (frame_NNNNNN.png from `ForgeRenderWorker --world mandelnewton
#                        --width W --height H --seed 12345 --quality high ...`);
#                        each is matched by name and size, informational
#   --install-deps       apt-get install the user-space build/export tools that are
#                        missing (build-essential zlib1g-dev python3 ffmpeg git);
#                        never drivers or CUDA
#   --cpu-emulation      no GPU: run everything with --backend cpu-emulation at tiny
#                        sizes (for testing this script itself)
#
# Environment: FORGE_CUDA_TEST_OUTPUT (default <repo>/cuda-test-results),
#              FORGE_LINUX_BUILD (default <repo>/.build/linux-cuda-<arch>), plus
#              build-linux-cuda.sh's variables (NVCC, FORGE_CUDA_ARCHS, ...).
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LINUX="$ROOT/Tools/ForgeRenderWorker/Linux"
BUILD="${FORGE_LINUX_BUILD:-$ROOT/.build/linux-cuda-$(uname -m)}"
RESULTS="${FORGE_CUDA_TEST_OUTPUT:-$ROOT/cuda-test-results}"
WORKER="$BUILD/ForgeRenderWorker"
INSPECT="$LINUX/tools/png_inspect.py"
REFERENCE="$LINUX/reference"

QUICK=0; SWEEP=0; SEQUENCE=30; RESOLUTION=1920x1080; QUALITY=high; BUILD_STEP=1; METAL=""; EMULATION=0; INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --sweep) SWEEP=1 ;;
    --sequence) SEQUENCE="$2"; shift ;;
    --resolution) RESOLUTION="$2"; shift ;;
    --quality) QUALITY="$2"; shift ;;
    --no-build) BUILD_STEP=0 ;;
    --compare-metal) METAL="$(cd "$2" && pwd)"; shift ;;
    --cpu-emulation) EMULATION=1 ;;
    --install-deps) INSTALL=1 ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
    *) echo "test-cuda-renderer.sh: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ "$QUICK" = 1 ] && SEQUENCE=0
BACKEND=(--backend cuda)
if [ "$EMULATION" = 1 ]; then
  BACKEND=(--backend cpu-emulation)
  RESOLUTION=192x108
  [ "$SEQUENCE" -gt 3 ] && SEQUENCE=3
  SWEEP=0
fi
W="${RESOLUTION%x*}"; H="${RESOLUTION#*x}"

mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
exec > >(tee -a "$RESULTS/test-log.txt") 2>&1

FAILURES=()
fail() { echo "FAIL: $*"; FAILURES+=("$*"); }
section() { printf '\n==== %s ====\n' "$*"; }
json() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@"; }

echo "Forge CUDA renderer validation — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repository: $ROOT ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null))"
echo "Results: $RESULTS"

# --- 0. environment -------------------------------------------------------------
if [ "$INSTALL" = 1 ]; then
  section "Installing missing user-space tools"
  need=()
  command -v c++ >/dev/null 2>&1 || need+=(build-essential)
  printf '#include <zlib.h>\nint main(){return 0;}\n' | c++ -x c++ - -o /dev/null 2>/dev/null || need+=(zlib1g-dev)
  command -v python3 >/dev/null 2>&1 || need+=(python3)
  command -v ffmpeg >/dev/null 2>&1 || need+=(ffmpeg)
  command -v git >/dev/null 2>&1 || need+=(git)
  if [ "${#need[@]}" -gt 0 ]; then
    SUDO=""; [ "$(id -u)" != 0 ] && SUDO=sudo
    echo "apt-get install ${need[*]}"
    $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "${need[@]}" \
      || fail "could not install ${need[*]}"
  else
    echo "nothing missing"
  fi
fi

section "0. Environment"
uname -a
grep PRETTY_NAME /etc/os-release 2>/dev/null
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) threads; RAM $(free -g 2>/dev/null | awk '/Mem:/{print $2}') GiB"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset} NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "(The CUDA worker needs only the 'compute' capability; no Vulkan, graphics, display or X server.)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$RESULTS/nvidia-smi.txt"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm \
    --format=csv | tee "$RESULTS/nvidia-smi-query.csv"
  nvidia-smi -q > "$RESULTS/nvidia-smi-q.txt" 2>&1 || true
elif [ "$EMULATION" = 0 ]; then
  fail "nvidia-smi not found: this machine/container exposes no NVIDIA driver (start the pod with a GPU; for Docker use --gpus all)"
fi
ls -l /dev/nvidia* 2>/dev/null | head -8 || true
echo "libcuda: $(ldconfig -p 2>/dev/null | grep -m1 'libcuda.so.1' | awk '{print $NF}' || echo 'not in ldconfig cache')"
cat /proc/driver/nvidia/version 2>/dev/null | head -1 || true
NVCC_BIN="${NVCC:-$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)}"
if [ -n "$NVCC_BIN" ]; then
  "$NVCC_BIN" --version | tail -2
  toolkit="$("$NVCC_BIN" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  driver_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  if [ -n "$driver_cuda" ] && [ -n "$toolkit" ]; then
    if [ "${toolkit%%.*}" -gt "${driver_cuda%%.*}" ]; then
      fail "CUDA toolkit $toolkit is a newer major version than the driver supports ($driver_cuda): use a CUDA ${driver_cuda%%.*}.x devel image"
    elif [ "$(printf '%s\n%s\n' "$toolkit" "$driver_cuda" | sort -V | tail -1)" != "$driver_cuda" ]; then
      echo "Note: toolkit $toolkit > driver's CUDA $driver_cuda; fine through CUDA minor-version compatibility, because the build contains native SASS for this GPU."
    fi
  fi
elif [ "$EMULATION" = 0 ]; then
  echo "nvcc: not found — use a CUDA *devel* image (e.g. runpod/pytorch:*-devel-*, nvidia/cuda:*-devel-*) or set NVCC"
fi
command -v ffmpeg >/dev/null 2>&1 && ffmpeg -hide_banner -version | head -1 || echo "ffmpeg: not found (the MP4 step will be skipped; apt-get install ffmpeg)"

# --- 1. build ------------------------------------------------------------------------
section "1. Build"
if [ "$BUILD_STEP" = 1 ]; then
  BUILD_ENV=()
  [ "$EMULATION" = 1 ] && [ -z "$NVCC_BIN" ] && BUILD_ENV=(FORGE_BUILD_WITHOUT_CUDA=1)
  if ! env "${BUILD_ENV[@]}" "$ROOT/Tools/ForgeRenderWorker/build-linux-cuda.sh" "$BUILD" 2>&1 | tee "$RESULTS/build-log.txt"; then
    fail "build failed (see build-log.txt)"
  fi
  cp "$BUILD/cuda-kernels.txt" "$RESULTS/" 2>/dev/null || true
fi
if [ ! -x "$WORKER" ]; then
  fail "no worker at $WORKER"
  printf '\n%d failure(s):\n' "${#FAILURES[@]}"; printf '  %s\n' "${FAILURES[@]}"; exit 1
fi
ldd "$WORKER" > "$RESULTS/ldd.txt" 2>&1
if grep -Eiq 'vulkan|libGL|libEGL|libX11|wayland' "$RESULTS/ldd.txt"; then fail "the worker links a graphics library (ldd.txt)"; fi

# --- 2. devices ------------------------------------------------------------------------
section "2. Worker and devices"
"$WORKER" --version | tee "$RESULTS/version.txt"
if [ "$EMULATION" = 0 ]; then
  if ! "$WORKER" --list-gpus | tee "$RESULTS/list-gpus.txt"; then
    fail "--list-gpus: CUDA is not usable (see above)"
  fi
fi
"$WORKER" --capabilities > "$RESULTS/capabilities.json" 2>&1 || true

# --- 3. fidelity -----------------------------------------------------------------
section "3. Fidelity (Offline — High, seed 12345)"
FID="$RESULTS/fidelity"
rm -rf "$FID"; mkdir -p "$FID"
METRICS="$FID/metrics.jsonl"
fidelity_case() { # width height frame reference-png-or-empty [no-oracle]
  local w="$1" h="$2" f="$3" ref="$4" oracle="${5:-oracle}" dir="$FID/${1}x${2}_frame$3"
  local png="$dir/frame_$(printf %06d "$f").png"
  echo
  echo "-- ${w}x${h} frame $f"
  if ! "$WORKER" "${BACKEND[@]}" --world mandelnewton --width "$w" --height "$h" --fps 30 --first-frame "$f" --frames 1 \
       --seed 12345 --quality high --output "$dir" --dump-uniforms "$dir/uniforms" --png-threads 0 > "$dir.log" 2>&1; then
    cat "$dir.log"; fail "render ${w}x${h} frame $f"; return
  fi
  grep -E '^Frame ' "$dir.log"
  python3 "$INSPECT" stats "$png" --expect "${w}x${h}" | tail -2 || fail "frame ${w}x${h}/$f is structurally wrong"
  if [ "$oracle" = oracle ]; then
    echo "vs the shipping Metal shader source on the CPU (same uniforms):"
    FORGE_LINUX_BUILD="$BUILD" "$LINUX/fidelity/run-fidelity-check.sh" "$dir/uniforms/frame_${f}_uniforms.bin" "$png" 0.01 "$METRICS" \
      2>&1 | grep -v '^CPU reference\|^wrote' || fail "${w}x${h} frame $f differs from the CPU oracle"
  fi
  if [ -n "$ref" ] && [ -f "$ref" ]; then
    echo "vs committed reference $(basename "$ref"):"
    python3 "$INSPECT" compare "$png" "$ref" --max-mean-abs 0.01 --min-psnr 35 --json "$METRICS" \
      || fail "${w}x${h} frame $f differs from the committed reference"
  fi
}
fidelity_case 128 72 0 "$REFERENCE/mandelnewton_128x72_high_seed12345_frame000000_lavapipe-x86_64.png"
fidelity_case 128 72 300 "$REFERENCE/mandelnewton_128x72_high_seed12345_frame000300_lavapipe-x86_64.png"
if [ "$QUICK" = 0 ]; then
  fidelity_case 320 180 150 ""
  # Reference only: the CPU oracle would take minutes at this size.
  [ "$EMULATION" = 0 ] && fidelity_case 640 360 0 "$REFERENCE/mandelnewton_640x360_high_seed12345_frame000000_lavapipe-x86_64.png" no-oracle
fi

echo
echo "-- determinism: 128x72 frame 0 rendered again"
"$WORKER" "${BACKEND[@]}" --width 128 --height 72 --first-frame 0 --frames 1 --seed 12345 --quality high \
  --output "$FID/repeat" --png-threads 0 > "$FID/repeat.log" 2>&1
if cmp -s "$FID/repeat/frame_000000.png" "$FID/128x72_frame0/frame_000000.png"; then
  echo "byte-identical"
else
  fail "a repeat render of the same frame is not byte-identical"
fi

echo
echo "-- frame-state hashes (CPU side) vs reference/snapshot-hashes_128x72_frames0-300_x86_64.json"
"$WORKER" --width 128 --height 72 --fps 30 --frames 301 --seed 12345 --snapshot-hashes 0,300 \
  > "$FID/snapshot-hashes.json" 2>/dev/null
python3 - "$FID/snapshot-hashes.json" "$REFERENCE/snapshot-hashes_128x72_frames0-300_x86_64.json" <<'EOF' || fail "frame-state hashes differ from the reference"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = True
for fa, fb in zip(a["frames"], b["frames"]):
    same = fa["sections"] == fb["sections"]
    ok &= same
    print(f"frame {fa['frame']}: {'identical' if same else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
EOF

# --- 4. one full-size frame -----------------------------------------------------
section "4. One ${W}x${H} ${QUALITY} frame"
ONE="$RESULTS/single-${W}x${H}"
if "$WORKER" "${BACKEND[@]}" --width "$W" --height "$H" --first-frame 0 --frames 1 --seed 12345 --quality "$QUALITY" \
     --warmup-frames 1 --png-threads 0 --output "$ONE" | tee "$ONE.log"; then
  python3 "$INSPECT" stats "$ONE/frame_000000.png" --expect "${W}x${H}" | tail -2 || fail "${W}x${H} frame is structurally wrong"
else
  fail "${W}x${H} frame render"
fi

# --- 5. sequence -------------------------------------------------------------------
if [ "$SEQUENCE" -gt 0 ]; then
  section "5. Sequence: $SEQUENCE frames of ${W}x${H} ${QUALITY} -> PNG + MP4"
  SEQ="$RESULTS/sequence-${W}x${H}"
  rm -rf "$SEQ"
  MP4_ARGS=()
  command -v ffmpeg >/dev/null 2>&1 && MP4_ARGS=(--mp4 "$SEQ/mandelnewton_${W}x${H}.mp4")
  SAMPLER=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
      --format=csv -l 1 > "$RESULTS/nvidia-smi-during-sequence.csv" 2>/dev/null &
    SAMPLER=$!
  fi
  mkdir -p "$SEQ"
  if ! "$WORKER" "${BACKEND[@]}" --width "$W" --height "$H" --first-frame 0 --frames "$SEQUENCE" --seed 12345 \
       --quality "$QUALITY" --warmup-frames 1 --output "$SEQ" "${MP4_ARGS[@]}" | tee "$SEQ.log"; then
    fail "sequence render"
  fi
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  if [ -n "$SAMPLER" ] && [ -s "$RESULTS/nvidia-smi-during-sequence.csv" ]; then
    echo "nvidia-smi during the sequence (utilization %, SM MHz, W):"
    awk -F', ' 'NR>1 { u+=$2; c+=$3; p+=$5; n++ } END { if (n) printf "  average: %.0f %% busy, %.0f MHz, %.0f W over %d samples\n", u/n, c/n, p/n, n }' \
      "$RESULTS/nvidia-smi-during-sequence.csv"
  fi
fi

# --- 6. sweep ------------------------------------------------------------------------
if [ "$SWEEP" = 1 ]; then
  section "6. Kernel variant sweep (960x540 ${QUALITY}, frames 0-1 after a warm-up, no PNG)"
  SW="$RESULTS/sweep"; mkdir -p "$SW"
  printf '%-10s %-8s %14s %14s\n' "reg cap" "block" "accumulate ms" "frame GPU ms" | tee "$SW/sweep.txt"
  for cap in 0 128 64; do
    for block in 8x8 16x8 16x16 32x4; do
      out="$SW/cap${cap}_${block}"
      if "$WORKER" --width 960 --height 540 --first-frame 0 --frames 2 --seed 12345 --quality "$QUALITY" --no-png \
           --warmup-frames 1 --cuda-register-cap "$cap" --cuda-block "$block" --output "$out" > "$out.log" 2>&1; then
        printf '%-10s %-8s %14.1f %14.1f\n' "$cap" "$block" \
          "$(json "$out/benchmark.json" 'd["median_gpu_accumulate_ms_per_frame"]')" \
          "$(json "$out/benchmark.json" 'd["median_gpu_ms_per_frame"]')" | tee -a "$SW/sweep.txt"
      else
        printf '%-10s %-8s %14s\n' "$cap" "$block" "failed: $(tail -1 "$out.log")" | tee -a "$SW/sweep.txt"
      fi
    done
  done
  sort -k3 -n "$SW/sweep.txt" | awk 'NR>0 && $3+0>0 {print "Fastest: --cuda-register-cap " $1 " --cuda-block " $2 " (" $3 " ms accumulate)"; exit}' \
    | tee -a "$SW/sweep.txt"
fi

# --- 7. metal comparison -------------------------------------------------------------
if [ -n "$METAL" ]; then
  section "7. Compared with Metal frames from $METAL (informational)"
  for mine in "$FID"/*/frame_*.png "$RESULTS"/single-*/frame_*.png "$RESULTS"/sequence-*/frame_*.png; do
    [ -f "$mine" ] || continue
    size="$(python3 "$INSPECT" stats "$mine" --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d[\"width\"]}x{d[\"height\"]}")' 2>/dev/null)"
    for theirs in $(find "$METAL" -name "$(basename "$mine")"); do
      other="$(python3 "$INSPECT" stats "$theirs" --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d[\"width\"]}x{d[\"height\"]}")' 2>/dev/null)"
      [ "$size" = "$other" ] || continue
      python3 "$INSPECT" compare "$mine" "$theirs" --max-mean-abs 0.01 --min-psnr 35 --json "$RESULTS/metal-metrics.jsonl" || true
    done
  done
fi

# --- summary ---------------------------------------------------------------------------
section "Summary"
python3 - "$RESULTS" <<'EOF' | tee "$RESULTS/summary.txt"
import glob, json, os, sys
r = sys.argv[1]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
gpu = None
for p in sorted(glob.glob(os.path.join(r, "single-*", "benchmark.json"))) + sorted(glob.glob(os.path.join(r, "sequence-*", "benchmark.json"))):
    b = load(p)
    if b:
        gpu = b.get("renderer", {})
        break
if gpu:
    for k in ("cuda_device_name", "cuda_compute_capability", "cuda_multiprocessors", "cuda_memory_total", "nvidia_driver_version",
              "cuda_driver_supports_cuda", "cuda_runtime_version_built", "cuda_kernel_code", "cuda_accumulate_registers", "cuda_block", "backend"):
        if k in gpu: print(f"{k:32} {gpu[k]}")
print()
m = os.path.join(r, "fidelity", "metrics.jsonl")
if os.path.exists(m):
    print("Fidelity (0..1 scale):")
    print(f"  {'image':44} {'vs':34} {'MAE':>8} {'RMSE':>8} {'PSNR dB':>8} {'max':>7} {'>1/64':>7}")
    for line in open(m):
        x = json.loads(line)
        a = os.path.relpath(x["a"], r); b = os.path.basename(x["b"])
        b = "CPU oracle (shipping MSL)" if b.endswith("_cpu_reference.png") else b[:34]
        psnr = "inf" if x["psnr_db"] is None else f"{x['psnr_db']:.1f}"
        print(f"  {a[-44:]:44} {b:34} {x['mae']:8.5f} {x['rmse']:8.5f} {psnr:>8} {x['max_abs']:7.4f} {x['fraction_over_1_64']:7.2%}"
              + ("" if x["match"] else "  FAIL"))
    print()
for kind in ("single", "sequence"):
    for p in sorted(glob.glob(os.path.join(r, f"{kind}-*", "benchmark.json"))):
        b = load(p)
        if not b: continue
        a = b["average_stage_timings"]
        print(f"{kind} {b['width']}x{b['height']} {b['quality']}, {b['frames']} frame(s), {b['samples_per_pixel']} spp:")
        print(f"  accumulate kernels       {a['gpu_accumulate_kernels_seconds']*1000:9.1f} ms/frame")
        print(f"  resolve kernel           {a['gpu_resolve_kernel_seconds']*1000:9.2f} ms/frame")
        print(f"  GPU -> CPU transfer      {a['gpu_to_cpu_transfer_seconds']*1000:9.2f} ms/frame (+{a['host_copy_seconds']*1000:.2f} ms host copy)")
        print(f"  frame prep (CPU)         {(a['timeline_camera_parameter_cpu_seconds']+a['uniform_build_cpu_seconds'])*1000:9.2f} ms/frame")
        print(f"  PNG encode + write       {(a['png_encode_seconds']+a['png_write_seconds'])*1000:9.1f} ms/frame ({b['png_encoder_threads']} encoder threads)")
        print(f"  total (serial equiv.)    {a['total_frame_seconds']*1000:9.1f} ms/frame")
        print(f"  throughput               {b['effective_render_fps']:9.3f} frames/s wall ({b['effective_seconds_per_completed_frame']:.3f} s/frame); "
              f"GPU-only {b['gpu_only_fps']:.3f} frames/s; {b['megapixel_samples_per_second']:.1f} Mpixel-samples/s")
        if "mp4_encode_seconds" in b:
            print(f"  MP4 (ffmpeg, Forge settings) {b['mp4_encode_seconds']:.2f} s -> {b['mp4']}")
        print()
s = os.path.join(r, "sweep", "sweep.txt")
if os.path.exists(s):
    print(open(s).read())
EOF

tar czf "$RESULTS.tgz" --exclude='*.png' --exclude='*.mp4' -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")" 2>/dev/null \
  && echo "Results (without images): $RESULTS.tgz"
echo "Frames and logs: $RESULTS"
if [ "${#FAILURES[@]}" -gt 0 ]; then
  printf '\n%d FAILURE(S):\n' "${#FAILURES[@]}"; printf '  %s\n' "${FAILURES[@]}"
  exit 1
fi
echo
echo "ALL CHECKS PASSED"
