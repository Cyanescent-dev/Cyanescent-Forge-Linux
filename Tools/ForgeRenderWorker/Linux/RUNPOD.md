# First CUDA run on RunPod (NVIDIA L4)

Status: **the CUDA kernels have not yet run on an NVIDIA GPU.** Everything
else was built and tested in a GPU-less cloud VM (see `README.md`). This run
is meant to be validation and benchmarking, not environment debugging: one
script does all of it and leaves the results in one directory.

## 1. Start the pod

- GPU: **L4** (any compute capability ≥ 7.0 works: A5000, A40, RTX 4090, …).
- Template: a CUDA **devel** image — e.g. *RunPod PyTorch* (`runpod/pytorch:*-cuda12.*-devel-ubuntu22.04`)
  or `nvidia/cuda:12.4.1-devel-ubuntu22.04`. "devel" matters: it has `nvcc`.
  (A "runtime"/"base" image has no compiler; see *Troubleshooting*.)
- Container disk: 20 GB is plenty. No volume is needed.
- Nothing about Vulkan, graphics capabilities, X or a display is needed.

## 2. Run (web terminal or SSH)

```sh
nvidia-smi                                             # the L4, a driver, "CUDA Version: 12.x"
git clone --depth 1 -b linux-cuda-worker-v0.1 https://github.com/Cyanescent-dev/Cyanescent-Forge.git
cd Cyanescent-Forge
./Tools/ForgeRenderWorker/test-cuda-renderer.sh --install-deps --sweep
```

(If the repository is private: `git clone https://<user>:<token>@github.com/...`
with a fine-grained read-only token, or `gh repo clone`.)

`--install-deps` apt-installs only what is missing of `build-essential
zlib1g-dev python3 ffmpeg git` (never drivers or CUDA). The script then:

| Step | What | Rough time on an L4 |
| --- | --- | --- |
| 0 | environment: `nvidia-smi`, driver, toolkit vs driver CUDA version, libcuda | seconds |
| 1 | build for this GPU's architecture only (sm_89) | ~1 min |
| 2 | `--version`, `--list-gpus` | seconds |
| 3 | fidelity: 128×72 frames 0 and 300, 320×180 frame 150 vs the CPU oracle and the references, 640×360 frame 0 vs its reference; determinism; frame-state hashes | ~1 min (mostly the CPU oracle) |
| 4 | one 1920×1080 Offline — High frame after a warm-up | seconds (unmeasured) |
| 5 | 30 frames 1920×1080 High → PNG + MP4, `nvidia-smi` sampled each second | minutes at most (unmeasured) |
| 6 | `--sweep`: register cap {none, 128, 64} × block {8×8, 16×8, 16×16, 32×4} at 960×540 | ~1 min |

It ends with `ALL CHECKS PASSED` or a list of failures, and a summary like:

```
cuda_device_name                 NVIDIA L4
cuda_compute_capability          8.9
nvidia_driver_version            5xx.xx
cuda_kernel_code                 sm_89 SASS (native)
...
Fidelity (0..1 scale):
  image                         vs                            MAE     RMSE  PSNR dB     max   >1/64
  fidelity/128x72_frame0/...    CPU oracle (shipping MSL)     ...
single 1920x1080 Offline — High, 1 frame(s), 16 spp:
  accumulate kernels   ... ms/frame
  resolve kernel       ... ms/frame
  GPU -> CPU transfer  ... ms/frame
  PNG encode + write   ... ms/frame
  total (serial equiv.) ... ms/frame
  throughput           ... frames/s wall; GPU-only ... frames/s
```

Options: `--quick` (steps 0–4 only), `--sequence N`, `--resolution 3840x2160`,
`--quality extreme`, `--no-build`, `--compare-metal DIR` (Metal frames from the
Mac, see `README.md`).

## 3. Bring the results back

- `cuda-test-results/summary.txt` — the numbers above
- `cuda-test-results.tgz` — every log, `benchmark.json`, `nvidia-smi` samples
  and metrics, without images (small)
- `cuda-test-results/sequence-1920x1080/` — the frames and `mandelnewton_1920x1080.mp4`
- `cuda-test-results/single-1920x1080/frame_000000.png` — the 1080p still

Download with the RunPod file browser (Jupyter templates), `runpodctl send
cuda-test-results.tgz`, or `scp -P <port> root@<ip>:Cyanescent-Forge/cuda-test-results.tgz .`

## Manual commands

```sh
W=.build/linux-cuda-x86_64/ForgeRenderWorker
$W --list-gpus
$W --width 1920 --height 1080 --frames 1 --output one                         # frame 0
$W --width 1920 --height 1080 --frames 120 --output seq --mp4 seq/out.mp4     # 4 s of video
$W --width 3840 --height 2160 --frame-list 0,300 --quality extreme --output 4k
$W --width 1920 --height 1080 --frames 10 --no-png --warmup-frames 1 --output bench \
   --cuda-register-cap 128 --cuda-block 16x8                                  # GPU-only timing
```

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `nvcc not found` | Not a devel image. Pick one, or install the compiler: `apt-get install cuda-nvcc-12-4 cuda-cudart-dev-12-4` (NVIDIA's apt repo is preconfigured in `nvidia/cuda` images; match the driver's CUDA major version), or `NVCC=/path/to/nvcc`. |
| `CUDA toolkit X is a newer major version than the driver supports` / `cudaErrorInsufficientDriver` | Use a devel image whose CUDA major version ≤ the "CUDA Version" `nvidia-smi` prints. |
| `cudaErrorNoKernelImageForDevice` | The binary was built elsewhere for another GPU: rebuild on this machine (the script does) or set `FORGE_CUDA_ARCHS`. |
| `unsupported GNU version` during the build | The toolkit is older than the image's gcc: `apt-get install g++-11` (or the version it names) and `FORGE_CUDA_HOST_CXX=g++-11`. The script tries g++-14 … g++-9 automatically. |
| `cudaErrorNoDevice` although `nvidia-smi` works | `CUDA_VISIBLE_DEVICES` hides the GPU, or the container lacks `/dev/nvidia*`: restart the pod. |
| `cudaErrorLaunchOutOfResources` | `--cuda-block` too large for the kernel's registers: use 8×8 (default) or `--cuda-register-cap 128`. |
| A fidelity check FAILS | Do not tune thresholds; bring `cuda-test-results.tgz` and the fidelity PNGs back. |

## Other clouds / Docker

On a host with Docker and the NVIDIA Container Toolkit:

```sh
docker build -f Tools/ForgeRenderWorker/Linux/docker/Dockerfile --build-arg CUDA_ARCHS=89 -t forge-cuda-worker .
docker run --rm --gpus all -v "$PWD/out:/out" forge-cuda-worker --width 1920 --height 1080 --frames 30 --output /out
```

(RunPod pods are already containers, so there the script above is the path.)
