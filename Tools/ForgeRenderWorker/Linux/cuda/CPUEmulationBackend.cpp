//
//  CPUEmulationBackend.cpp
//  Cyanescent Forge — Linux CUDA worker
//
//  The CUDA backend's kernels — the same source, cuda/ForgeShaderPort.cuh,
//  i.e. the shipping MSL modules — compiled by the host compiler and run on
//  CPU threads, with the CUDA backend's frame structure: accumulation
//  dispatches in order (tiles x sample batches), then the resolve.
//
//  It exists so the whole worker (frame state, uniforms, kernel source,
//  resolve, export, benchmark) can be exercised on machines without an
//  NVIDIA GPU, and so a CUDA frame can be compared with the same kernels
//  evaluated on the CPU. It is slow (scalar float), is never selected
//  automatically, and is labelled "cpu-emulation" everywhere it appears.
//
#include "../src/GPUBackend.hpp"

#include "ForgeShaderPort.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#ifdef FORGE_LAYOUT_ASSERTS
#include FORGE_LAYOUT_ASSERTS
#endif

namespace forge {

namespace {

double now()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

typedef void (*PixelKernel)(forge_gpu::float4 *, const forge_gpu::ForgeUniforms &, forge_gpu::uint2);

PixelKernel worldPixelKernel(WorldKernel world)
{
    switch (world) {
    case WorldKernel::MandelNewton: return &forge_gpu::MandelNewtonWorldKernel::accumulate;
    }
    return nullptr;
}

/// Runs body(row) for rows [0, rows) on `threads` threads.
template <class Body>
void parallelRows(int rows, int threads, const Body &body)
{
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (int row; (row = next.fetch_add(1)) < rows;) body(row);
    };
    std::vector<std::thread> pool;
    const int n = std::max(1, std::min(threads, rows));
    for (int i = 1; i < n; ++i) pool.emplace_back(worker);
    worker();
    for (auto &t : pool) t.join();
}

class CPUEmulationBackend final : public RenderBackend {
public:
    explicit CPUEmulationBackend(const BackendOptions &options) : options_(options)
    {
        kernel_ = worldPixelKernel(options.world);
        if (!kernel_) throw BackendError("The CPU emulation backend has no kernel for the requested world.");
        threads_ = options.cpuThreads > 0 ? options.cpuThreads
                                          : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        const size_t pixels = static_cast<size_t>(options.width) * options.height;
        accumulation_.assign(pixels, forge_gpu::float4(0.0f));
        output_.resize(pixels);
    }

    std::string backendName() const override { return "cpu-emulation"; }
    std::string deviceName() const override { return "CPU emulation of the CUDA kernels"; }
    int deviceIndex() const override { return -1; }

    std::vector<BackendFact> facts() const override
    {
        return {
            {"backend", "CPU emulation of the CUDA kernels (same kernel source; diagnostics only, not a GPU)"},
            {"cpu_threads", std::to_string(threads_)},
            {"math", "host libm float functions, -ffp-contract=off"},
        };
    }

    void renderFrame(const UniformBlocks &dispatches, const UniformBlocks &resolve, uint16_t *rgba16,
                     BackendFrameTimings &t) override
    {
        if (dispatches.blockSize != sizeof(forge_gpu::ForgeUniforms) || resolve.count != 1 ||
            resolve.blockSize != sizeof(forge_gpu::ForgeUniforms)) {
            throw BackendError("Uniform blocks do not match the emulated ForgeUniforms.");
        }
        const double start = now();
        forge_gpu::float4 *accumulation = accumulation_.data();
        for (size_t i = 0; i < dispatches.count; ++i) {
            forge_gpu::ForgeUniforms u;
            std::memcpy(&u, dispatches.block(i), sizeof(u));
            const int w = static_cast<int>(u.tileSize.x), h = static_cast<int>(u.tileSize.y);
            // One "thread" per pixel of the tile, as the CUDA grid launches
            // (the kernel itself bounds-checks against tile and image).
            parallelRows(h, threads_, [&](int y) {
                for (int x = 0; x < w; ++x) kernel_(accumulation, u, forge_gpu::uint2(x, y));
            });
        }
        const double accumulated = now();

        forge_gpu::ForgeUniforms u;
        std::memcpy(&u, resolve.block(0), sizeof(u));
        const int width = options_.width;
        parallelRows(options_.height, threads_, [&](int y) {
            for (int x = 0; x < width; ++x) {
                output_[static_cast<size_t>(y) * width + x] = forge_gpu::forgeResolvePixel(accumulation, u, x, y);
            }
        });
        const double resolved = now();

        std::memcpy(rgba16, output_.data(), output_.size() * sizeof(forge_gpu::RGBA16));
        const double copied = now();

        t.enqueueCPUSeconds = 0;
        t.accumulateGPUSeconds = accumulated - start;
        t.resolveGPUSeconds = resolved - accumulated;
        t.transferGPUSeconds = 0;
        t.hostCopySeconds = copied - resolved;
        t.waitSeconds = 0;
        t.gpuWallSeconds = resolved - start;
    }

private:
    BackendOptions options_;
    PixelKernel kernel_ = nullptr;
    int threads_ = 1;
    std::vector<forge_gpu::float4> accumulation_;
    std::vector<forge_gpu::RGBA16> output_;
};

static_assert(sizeof(forge_gpu::RGBA16) == 4 * sizeof(uint16_t), "RGBA16 must be four packed 16-bit channels");

} // namespace

std::unique_ptr<RenderBackend> makeCPUEmulationBackend(const BackendOptions &options)
{
    return std::unique_ptr<RenderBackend>(new CPUEmulationBackend(options));
}

size_t backendUniformBlockSize() { return sizeof(forge_gpu::ForgeUniforms); }

} // namespace forge
