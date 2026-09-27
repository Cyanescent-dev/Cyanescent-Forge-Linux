# Linux worker reference outputs

Produced by the first Linux worker run — the (now retired) Vulkan worker V0.1
on lavapipe, source 361bf94 — and used by the CUDA worker's
`test-cuda-renderer.sh` as fixed references (they measured MAE ~0.0007 /
~0.0013 from the M3's Metal frames 0 / 300):

| File | What |
| --- | --- |
| `mandelnewton_128x72_high_seed12345_frame000000_lavapipe-x86_64.png` | Smoke-test frame: 128×72, absolute frame 0, seed 12345, 30 fps, Offline — High |
| `mandelnewton_128x72_high_seed12345_frame000300_lavapipe-x86_64.png` | Same settings, absolute frame 300 (`--first-frame 300 --frames 1`) |
| `mandelnewton_640x360_high_seed12345_frame000000_lavapipe-x86_64.png` | 640×360, frame 0 |
| `benchmark_*_lavapipe-x86_64.json` | The runs' `benchmark.json` |
| `snapshot-hashes_128x72_frames0-300_x86_64.json` | `--snapshot-hashes 0,300` at 128×72, `--frames 301` |

Machine: Ubuntu 24.04.4 LTS, x86_64 (Intel Xeon @ 2.10 GHz, 4 vCPUs), no GPU;
Vulkan device llvmpipe (Mesa 25.2.8, LLVM 20.1.2), Vulkan 1.4.

These are 16-bit RGB PNGs. Other renderers and devices (CUDA on NVIDIA, the
CUDA kernels on the CPU, Metal on the M3) are expected to be close, not
byte-identical:

```sh
python3 Tools/ForgeRenderWorker/Linux/tools/png_inspect.py compare \
  cuda-test-results/fidelity/128x72_frame0/frame_000000.png \
  Tools/ForgeRenderWorker/Linux/reference/mandelnewton_128x72_high_seed12345_frame000000_lavapipe-x86_64.png
```

The snapshot hashes, by contrast, should be identical on every machine that
evaluates the frame state correctly (x86_64, aarch64 and the macOS worker).
