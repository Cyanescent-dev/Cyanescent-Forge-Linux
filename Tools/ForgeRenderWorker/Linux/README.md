# Forge Render Worker — Linux / CUDA (V0.1)

A native, headless Linux build of ForgeRenderWorker for **rented NVIDIA cloud
GPUs** (RunPod L4, A5000, A40, RTX 4090, …). It renders the Cyanescent Forge
**Mandelbrot / Newton Morph** world (`mandelnewton.world`) with CUDA to 16-bit
PNG frames (plus, optionally, an H.264 MP4 made the way Forge makes it) and a
`benchmark.json`. Frame numbering, timeline, camera, parameters, seed and
sampling are the macOS worker's.

| Where | Renderer | Status |
| --- | --- | --- |
| macOS / Apple Silicon (the app and `main.swift` worker) | **Metal** | production, unchanged |
| Linux + NVIDIA, headless cloud | **CUDA** (this directory) | V0.1 proof of concept: Mandelbrot / Newton only |
| Linux, Vulkan | retired — see `obsolete-vulkan/` | could not create a device in RunPod's headless containers |

The CUDA worker needs **no Vulkan, OpenGL, EGL, X11, Wayland, display, window
or graphics context** — only the NVIDIA driver's `libcuda.so.1`. The build
fails if the executable links any graphics or window-system library, and the
CUDA runtime is linked statically.

**Validation status.** Built and tested in a cloud VM without a GPU: nvcc
12.0 (Ubuntu 24.04) and 12.4 (the Docker image, Ubuntu 22.04) compile the
kernels for sm_75–sm_90, the host tests pass, and the
same kernel source run on the CPU (`--backend cpu-emulation`) matches the
references (see *Fidelity*). The CUDA kernels have **not yet executed on an
NVIDIA GPU**. The first RunPod run is `RUNPOD.md`, and it is one script.

## Architecture

```
                  Forge shared logic (portable C++, unchanged from the Vulkan worker)
   tempo, automation lanes, keyframed camera, world parameters, quality presets,
   absolute frame -> beats, tiles, sample batches — ported function by function from
   Swift and verified bit for bit (src/ForgeTimeline, MandelNewtonWorld, ForgeFrame)
                                     |
             ForgeUniforms — CyanescentForge/Shaders/ForgeShaderTypes.h, unmodified,
             Metal byte layout (float3 = 16 bytes), one block per dispatch
                                     |
                   src/GPUBackend.hpp (bytes in, rgba16 out; no GPU types)
                        /                                 \
          macOS: Metal (unchanged)                 Linux: cuda/CUDABackend.cu
                                                          |
                          CUDA kernels compiled from the SHIPPING Metal source:
                          ForgeMandelNewton.h, ForgeMath.h, ForgePost.h, ... via
                          cuda/ForgeShaderPort.cuh + cuda/ForgeCUDACompat.cuh
                                                          |
                   per frame: tiles x sample batches of accumulate kernels (float4
                   accumulation in VRAM) -> resolve kernel (rgba16) -> ONE copy to
                   pinned host memory
                                                          |
                   PNG encoder threads (overlap the next frame) -> ffmpeg -> MP4
```

### The key decision: compile the Metal source, don't translate it

Forge's shading modules are Metal Shading Language written in a C++-compatible
subset (that is how `Tools/CPUReference` runs them on a CPU). CUDA is C++, so
the CUDA backend compiles **the same files the Mac compiles** —
`ForgeMandelNewton.h` and the camera, lighting, material, volumetric and post
modules it includes — with no hand translation to drift (the Vulkan port had
to hand-write GLSL because GLSL cannot pass structs by reference).
`cuda/ForgeShaderPort.cuh` includes them inside `namespace forge_gpu` with the
MSL spellings mapped only while they are included:

| MSL | CUDA |
| --- | --- |
| `inline` (every shader function) | `__host__ __device__ inline` |
| `kernel void mandelnewtonRenderKernel(...)` | an ordinary function; a CUDA `__global__` wrapper calls it |
| `constant T &`, `thread`, `device` | `const T &`, nothing, nothing |
| `[[buffer(n)]]`, `[[thread_position_in_grid]]` | `[[]]` |
| `float2/3/4`, `uint2`, `saturate`, `mix`, swizzles, … | `cuda/ForgeCUDACompat.cuh` |

So coordinate mapping, iteration, the Mandelbrot → Newton morph, colour,
relief, lighting, camera rays, time dependence, precision (float
throughout; no double anywhere in the hot path), sample sequences and
evaluation order are Metal's by construction. The one piece that lives in a
`.metal` file rather than a shared header — `forgeResolveKernel` (average,
`forgePostProcess`, deterministic dither, `rgba16Unorm` store with
round-half-to-even) — is repeated expression for expression in
`forgeResolvePixel`.

Precision choices match Metal's: Forge's Metal library is compiled with
`fastMathEnabled = false`, so the CUDA build does not use `--use_fast_math`
(IEEE division and square root, no flush-to-zero, accurate `sinf`/`expf`/
`powf`). nvcc's default multiply-add contraction (FMA) is kept; with the GPU's
math functions it is one source of last-bit differences from the CPU oracle.

### Files

| File | Role |
| --- | --- |
| `src/ForgeHostTypes.hpp`, `ForgeTimeline.*`, `MandelNewtonWorld.*`, `ForgeFrame.*` | Portable frame state and uniforms (from the Vulkan worker, unchanged) |
| `src/SnapshotHash.*`, `Json.hpp`, `PNGWriter.*` | Frame-state hashes compatible with `SnapshotHash.swift`, JSON, 16-bit PNG (unchanged) |
| `src/GPUBackend.hpp` | The backend seam: uniform blocks in, rgba16 out, timings and device facts |
| `src/FrameExport.*` | Asynchronous PNG encoding; MP4 with Forge's ffmpeg arguments (`Core/MP4Export.swift`) |
| `src/main.cpp` | CLI, render loop, console timings, `benchmark.json`, `--capabilities` |
| `cuda/ForgeCUDACompat.cuh` | MSL vector types and builtins for CUDA (host + device), Metal layout |
| `cuda/ForgeShaderPort.cuh` | Includes the shipping shader modules; resolve pixel; world kernel table |
| `cuda/CUDABackend.cu` | CUDA runtime: devices, selection, buffers, kernels, streams, events, errors |
| `cuda/CPUEmulationBackend.cpp` | The same kernels on CPU threads (`--backend cpu-emulation`) |
| `cuda/CUDAUnavailable.cpp` | Linked instead of the CUDA backend in a `FORGE_BUILD_WITHOUT_CUDA=1` build |
| `cuda/CUDAArchitectures.hpp` | Which GPUs a build's SASS/PTX list can run on |
| `tools/check_cuda_layout.py` | Build step: `static_assert` every ForgeUniforms field offset in nvcc's device pass |
| `tools/png_inspect.py` | PNG validation and comparison (MAE, RMSE, PSNR, max abs, JSON) |
| `tests/CUDAWorkerSelfTests.cpp` | Host unit tests, run by every build |
| `fidelity/` | CPU oracle: the shipping MSL compiled with `ForgeCPUCompat.h`, from dumped uniforms |
| `reference/` | Committed reference frames and frame-state hashes |
| `docker/Dockerfile` | Optional container build (`validate` and slim `runtime` targets) |
| `obsolete-vulkan/` | The retired Vulkan backend, kept for reference only (not built) |

## Prerequisites

- An NVIDIA GPU of compute capability **7.0 or newer** (T4, V100, A10, A40,
  A5000, A100, L4, L40S, RTX 30/40/50, H100, …) with its driver; `nvidia-smi`
  works. In a container, only the `compute` driver capability is needed.
- The CUDA toolkit, **11.7 or newer** (`nvcc`) — any CUDA "devel" image has it
  (RunPod PyTorch images, `nvidia/cuda:*-devel-*`). A 12.x toolkit works with
  any 12.x driver: the build contains native code for the GPU it is built on.
- `build-essential zlib1g-dev python3`; `ffmpeg` for MP4 output.

The build never installs or touches the NVIDIA kernel driver.

## Build

```sh
./Tools/ForgeRenderWorker/build-linux-cuda.sh      # -> .build/linux-cuda-<arch>/ForgeRenderWorker
```

It finds `nvcc` (PATH, then `/usr/local/cuda*/bin`), picks a host compiler
that `nvcc` accepts, and builds for the GPUs `nvidia-smi` reports (on a
machine without a GPU: sm_75, sm_80, sm_86, sm_89, sm_90), always adding PTX
for the newest so later GPUs can JIT it. It prints each kernel's registers and
spills per architecture, checks every `ForgeUniforms` field offset in the
device compilation, verifies the executable has no graphics dependencies, and
runs the host self-tests. Variables: `NVCC`, `FORGE_CUDA_ARCHS="86 89"`,
`FORGE_CUDA_HOST_CXX`, `CXX`, and `FORGE_BUILD_WITHOUT_CUDA=1` (no nvcc:
CPU emulation only, for CI).

## Render

```sh
W=.build/linux-cuda-x86_64/ForgeRenderWorker

$W --list-gpus                                   # devices, driver, build, suitability
$W --width 1920 --height 1080 --frames 1 --output out            # one frame (frame 0)
$W --width 1920 --height 1080 --first-frame 300 --frames 1 --output out   # frame 300
$W --width 1920 --height 1080 --frames 120 --output out --mp4 out/mandelnewton.mp4
$W --frame-list 0,150,300 --width 3840 --height 2160 --quality extreme --output out4k
$W --width 1920 --height 1080 --frames 10 --no-png --warmup-frames 1 --output bench  # GPU only
```

| Flag | |
| --- | --- |
| `--width`, `--height`, `--fps`, `--frames`, `--first-frame`, `--frame-list`, `--seed`, `--quality standard\|high\|extreme\|quick\|fast`, `--output` | as the macOS worker (default 1920×1080, 30 FPS, 300 frames, seed 12345, Offline — High) |
| `--mp4 FILE`, `--mp4-quality standard\|high\|very-high` | after the frames, ffmpeg over the PNG sequence with Forge's arguments (H.264, `-preset slow -crf 18`, yuv420p, faststart; never overwrites) |
| `--no-png` | render and read back but write nothing (GPU benchmarking) |
| `--png-threads N` | PNG encoders overlapping the next frame's render (default half the CPU threads, 1–8; 0 = inline) |
| `--warmup-frames N` | render the first frame N times untimed first (module load, clocks) |
| `--list-gpus` | CUDA devices: compute capability, SMs, clocks, memory (total/free), PCI bus, UUID, display watchdog, compute mode, whether this build can run on it |
| `--cuda-device N`, `--cuda-device-name TEXT` | choose a device (0-based after `CUDA_VISIBLE_DEVICES`; name: unique case-insensitive substring) |
| `--cuda-block WxH` | threads per block (default 8×8 = Metal's threadgroup) |
| `--cuda-register-cap 0\|128\|64` | accumulate-kernel variant (compiler's choice, ≤128, ≤64 registers) |
| `--gpu-timeout-seconds S` | fail if a frame takes longer on the GPU |
| `--backend cpu-emulation`, `--cpu-threads N` | run the CUDA kernels on the CPU (tests; slow) |
| `--capabilities`, `--version`, `--snapshot-hashes F,…`, `--snapshot-dump DIR`, `--dump-uniforms DIR` | as the Vulkan worker |

Exit codes: 0 completed, 1 failed, 2 usage. Frames are `frame_NNNNNN.png` by
absolute frame, 16-bit RGB, sRGB-tagged. Not implemented: `--project`,
`--job`, multi-GPU, other worlds, loop closure.

**Device selection.** Default: among devices this build can run on, a device
without a display watchdog, then the most SMs, then the lowest index. The
worker never falls back to the CPU: without a usable CUDA device it exits 1
with the reason (no device, driver too old for the build, no code for the
architecture, …). To use several GPUs, run one worker per GPU with
`--cuda-device N` and disjoint `--frame-list`/`--first-frame` ranges; any
part of a range renders exactly the frames a whole-range render would.

A run prints the device (name, compute capability, SMs, clocks, memory,
driver version and the CUDA version it supports, the runtime the worker was
built with, whether the kernel runs as native SASS or JIT-compiled PTX, the
kernel's registers) and, per frame, the accumulate and resolve kernel times,
the GPU→CPU transfer, and the render-loop time; a summary adds frame
preparation, PNG encode and write, the serial-equivalent total, wall-clock
throughput and the MP4 time.

## Performance design

- All pixel work stays on the GPU. The float4 accumulation buffer never leaves
  VRAM; per frame exactly one transfer crosses PCIe: the finished rgba16
  framebuffer (16.6 MB at 1080p) into pinned host memory.
- Uniforms travel as `__grid_constant__` kernel parameters (832 bytes), so
  there is no upload and no per-thread copy; all of a frame's launches are
  queued into one stream without host synchronisation, and the host sleeps
  (blocking sync) until the copy completes.
- Forge's tile × sample-batch structure is kept (it fixes the accumulation
  order, so frames match Metal's); on a headless GPU the launches are just
  queued back to back (120 per 1080p High frame; launch cost is microseconds).
- PNG deflate runs on encoder threads while the GPU renders the next frame.
- Kernel resources (nvcc 12.0, sm_89): the accumulate kernel uses 199
  registers without spilling, which limits occupancy; `--cuda-register-cap
  128/64` trade spills for occupancy. Which wins is a measurement, not a
  guess: `test-cuda-renderer.sh --sweep`.

## Fidelity

Three checks, none needing a Mac, all run by `test-cuda-renderer.sh`:

1. **Layout** (every build): `ForgeUniforms`, `ForgeSceneParams` and
   `ForgeQuality` — 170 fields — at the host's Metal offsets, asserted at
   compile time in the CUDA device pass.
2. **Pixels**: CUDA frames vs (a) the CPU oracle — the shipping MSL compiled
   with the independent `Tools/CPUReference/ForgeCPUCompat.h`, from the exact
   uniforms the worker used (`--dump-uniforms`) — and (b) the committed
   reference frames. Required: MAE ≤ 0.01 and PSNR ≥ 35 dB (0..1 scale);
   RMSE, max |diff| and the share of pixels off by more than 1/64 are
   reported. Also: a repeat render must be byte-identical (no atomics, a fixed
   accumulation order).
3. **Frame state**: `--snapshot-hashes` (SHA-256 of the canonical frame,
   loopClosure, sceneParams and camera sections, compatible with
   `SnapshotHash.swift`) identical to `reference/`.

Measured in the cloud VM (x86_64, no GPU) with the CUDA kernel source on the
CPU (`--backend cpu-emulation`), Offline — High, seed 12345:

| Frame | vs CPU oracle (shipping MSL) | vs reference frame |
| --- | --- | --- |
| 128×72, 0 | MAE 0.00000, PSNR 126 dB | MAE 0.00072, RMSE 0.0048, PSNR 46.4 dB, max 0.26, 1.2 % > 1/64 |
| 128×72, 300 | MAE 0.00000, PSNR 125 dB | MAE 0.00107, RMSE 0.0035, PSNR 49.0 dB, max 0.11, 2.2 % > 1/64 |
| 320×180, 150 | MAE 0.00000, PSNR 126 dB | — |
| 640×360, 0 | — | MAE 0.00084, RMSE 0.0056, PSNR 45.1 dB, max 0.61, 1.4 % > 1/64 |

The reference frames were rendered by the Vulkan worker on lavapipe; that
renderer measured MAE ~0.0007 / ~0.0013 against the M3's Metal frames 0 /
300. On a GPU, expect the CUDA frames to sit at a similar distance from the
oracle and references — not byte-identical: GPU `sinf/expf/powf` and FMA
contraction differ in the last bits, and the residual concentrates at fractal
boundaries, where escape/convergence tests and the relief bisection are
chaotic. Structure, camera, colour, lighting and sky must agree; a large or
structured difference is a bug.

**Comparing with Metal directly** (on the Mac):

```sh
FORGE_WORKER_TARGET=arm64-apple-macosx14.0 ./Tools/ForgeRenderWorker/build.sh .build/ForgeRenderWorker-arm64
for f in 0 150 300; do
  .build/ForgeRenderWorker-arm64 --world mandelnewton --width 1920 --height 1080 --fps 30 \
    --first-frame $f --frames 1 --seed 12345 --quality high --output metal-frames
done
```

then on the GPU machine `test-cuda-renderer.sh --compare-metal metal-frames`
(or `python3 Tools/ForgeRenderWorker/Linux/tools/png_inspect.py compare A.png B.png`).

## Benchmarking

`test-cuda-renderer.sh` renders one 1920×1080 High frame after a warm-up, then
a 30-frame sequence to PNG + MP4 with `nvidia-smi` sampled every second, and
summarises: accumulate and resolve kernel time, GPU→CPU transfer, frame
preparation, PNG encode + write, serial-equivalent total per frame, wall-clock
frames/s, GPU-only frames/s and Mpixel-samples/s. `--sweep` measures the
register-cap × block-size variants. Everything is in `benchmark.json`
(`frame_stage_timings`, `average_stage_timings`, `renderer`, …) for
comparison with the Apple Silicon worker's `benchmark.json`.

## Adding another world

The backend is shared; a world adds its kernel and its CPU-side state:

1. **CPU side** — port the world's scene (defaults, descriptors, camera
   journey, automation) as `src/<World>World.*`, as `MandelNewtonWorld` does,
   and extend `makeWorkerProject`/`evaluateFrame` (`src/ForgeFrame.cpp`) and
   `--world`. Verify with `--snapshot-hashes` against the Mac.
2. **GPU side** — include the world's shipping module
   (`CyanescentForge/Shaders/Forge<World>.h`) in `cuda/ForgeShaderPort.cuh`;
   if it uses MSL outside the compat subset, extend `ForgeCUDACompat.cuh`
   (never edit the shared module for CUDA's sake). Add
   `struct <World>WorldKernel { static FORGE_HD void accumulate(...) }` calling
   its MSL entry point, a `WorldKernel` value (`src/GPUBackend.hpp`) and a case
   in `worldAccumulateKernel` (`CUDABackend.cu`) and `worldPixelKernel`
   (`CPUEmulationBackend.cpp`).
3. **Fidelity** — a CPU oracle for the world (as
   `fidelity/MandelNewtonCPUReference.cpp`), reference frames, and a case in
   `test-cuda-renderer.sh`.

Worlds with more than a per-pixel kernel (Physarum/MCPM agent simulation and
field textures, Duoverse volumes, loop closure) need world-owned device
buffers and extra passes between frames: give `CUDABackend` a per-world
resource object created with the framebuffers and a `prepareFrame` hook that
enqueues those passes into the same stream before the accumulation launches.
The render loop, export and timing code do not change.

## Keeping it in step

The CUDA kernels follow the shipping shader modules automatically; rebuild
and re-run `test-cuda-renderer.sh` after changing them. When
`ApollonianWorld.metal`'s `forgeResolveKernel` changes, update
`forgeResolvePixel`. When `MandelNewtonWorldScene.swift` or
`ForgeWorldSceneSupport.swift` change, update `src/MandelNewtonWorld.cpp`
(`--snapshot-hashes` against a Mac catches CPU-side drift). A
`ForgeShaderTypes.h` change is checked by the layout step.
