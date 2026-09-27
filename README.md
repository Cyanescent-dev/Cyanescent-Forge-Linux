# Cyanescent Forge Linux CUDA renderer

This repository contains the headless Linux/NVIDIA CUDA renderer, its build and validation scripts, reference frames, and the shared shader headers required to build and test it.

The published files are a Linux-only snapshot from the private upstream repository at branch `linux-cuda-worker-v0.1`, commit `f991306019309fb980978d1685a9e7336edb2438`. See [UPSTREAM.md](UPSTREAM.md) for provenance. This repository has a new Git history so the original Forge app sources and history remain private.

## Run the validation

Use an NVIDIA GPU and a CUDA devel image, then run:

```sh
nvidia-smi
git clone --depth 1 https://github.com/Cyanescent-dev/Cyanescent-Forge-Linux.git
cd Cyanescent-Forge-Linux
./Tools/ForgeRenderWorker/test-cuda-renderer.sh --install-deps --sweep
```

The script builds the worker, checks fidelity and determinism, renders a 1080p High frame and sequence, samples GPU telemetry, sweeps CUDA kernel variants, and writes `cuda-test-results/summary.txt` plus `cuda-test-results.tgz`.

The renderer is headless CUDA compute. The build does not require Vulkan, OpenGL, EGL, X11, Wayland, or a display server.
