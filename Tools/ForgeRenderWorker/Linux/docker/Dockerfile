# Cyanescent Forge — Linux CUDA render worker, as a container.
#
# Build from the repository root:
#
#   docker build -f Tools/ForgeRenderWorker/Linux/docker/Dockerfile \
#       --build-arg CUDA_ARCHS="89" --build-arg REVISION="$(git rev-parse --short HEAD)" \
#       -t forge-cuda-worker .
#
# (CUDA_ARCHS: 89 = L4 / RTX 4090 / L40S, 86 = A40 / A5000 / RTX 3090,
#  80 = A100, 90 = H100, 75 = T4; several are allowed, e.g. "86 89 90". The
#  default builds all of 75 80 86 89 90 and takes a few minutes.)
#
# Run (needs the NVIDIA Container Toolkit on the host; only the `compute`
# driver capability is required — no graphics/Vulkan, no display):
#
#   docker run --rm --gpus all -v "$PWD/out:/out" forge-cuda-worker \
#       --width 1920 --height 1080 --frames 30 --output /out --mp4 /out/mandelnewton.mp4
#
#   # full validation (fidelity, benchmark), results in ./out/cuda-test-results:
#   docker run --rm --gpus all -v "$PWD/out:/out" --entrypoint /forge/Tools/ForgeRenderWorker/test-cuda-renderer.sh \
#       -e FORGE_CUDA_TEST_OUTPUT=/out/cuda-test-results forge-cuda-worker:validate --no-build
#
# Two targets: `validate` (the default build stage: toolkit, sources, tests)
# and `runtime` (the final, small image: the worker, ffmpeg and zlib only).
# The host's NVIDIA kernel driver is never installed in either: the NVIDIA
# container runtime mounts libcuda.so.1 from the host at run time.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS validate
ARG CUDA_ARCHS="75 80 86 89 90"
ARG REVISION=unknown
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev python3 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /forge
COPY CyanescentForge/Shaders CyanescentForge/Shaders
COPY Tools/CPUReference Tools/CPUReference
COPY Tools/ForgeRenderWorker Tools/ForgeRenderWorker
RUN FORGE_CUDA_ARCHS="${CUDA_ARCHS}" FORGE_WORKER_SOURCE_REVISION="${REVISION}" \
        ./Tools/ForgeRenderWorker/build-linux-cuda.sh /forge/.build/linux-cuda-x86_64 \
    && ln -s /forge/.build/linux-cuda-x86_64/ForgeRenderWorker /usr/local/bin/ForgeRenderWorker
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["/usr/local/bin/ForgeRenderWorker"]

FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU} AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        zlib1g ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=validate /forge/.build/linux-cuda-x86_64/ForgeRenderWorker /usr/local/bin/ForgeRenderWorker
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /out
ENTRYPOINT ["/usr/local/bin/ForgeRenderWorker"]
