//
//  CUDABackend.cu
//  Cyanescent Forge — Linux CUDA worker
//
//  Headless NVIDIA rendering through the CUDA runtime API: no Vulkan, no
//  OpenGL/EGL, no X11/Wayland, no display, window or graphics context. The
//  executable links the CUDA runtime statically, so on the machine it needs
//  only the NVIDIA driver's libcuda.so.1 (which the NVIDIA container runtime
//  mounts with NVIDIA_DRIVER_CAPABILITIES=compute).
//
//  Per frame (ForgeRenderer's structure, as on Metal):
//
//    for each sample batch, for each tile:   accumulate kernel (the world's
//        shipping MSL entry point), uniforms as a __grid_constant__ parameter
//    resolve kernel (forgeResolveKernel)     float4 accumulation -> rgba16
//    one cudaMemcpyAsync                     rgba16 framebuffer -> pinned host
//
//  All launches go into one stream without host synchronisation; the host
//  waits once per frame, after the copy. The accumulation buffer never
//  leaves the GPU. CUDA events bracket accumulation, resolve and transfer so
//  their GPU durations are reported separately.
//
#include "../src/GPUBackend.hpp"

#include "CUDAArchitectures.hpp"
#include "ForgeShaderPort.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef FORGE_LAYOUT_ASSERTS
#include FORGE_LAYOUT_ASSERTS
#endif

#ifndef FORGE_CUDA_ARCHS
#define FORGE_CUDA_ARCHS "unknown"
#endif

#if CUDART_VERSION < 11070
#error "The CUDA worker needs CUDA 11.7 or newer (__grid_constant__)."
#endif

namespace forge_gpu {

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// The accumulate kernel in three register budgets. Which is fastest depends
// on the GPU (occupancy against spills), so all three are compiled and
// --cuda-register-cap picks one; test-cuda-renderer.sh --sweep measures them.
//   R0    compiler's choice (no bound)
//   R128  __launch_bounds__(256, 2): at most 128 registers per thread
//   R64   __launch_bounds__(256, 4): at most 64 registers per thread
#define FORGE_ACCUMULATE_BODY                                                           \
    {                                                                                   \
        World::accumulate(accumulation, u,                                              \
                          uint2(blockIdx.x * blockDim.x + threadIdx.x,                  \
                                blockIdx.y * blockDim.y + threadIdx.y));                \
    }

template <class World>
__global__ void forgeAccumulateR0(float4 *accumulation, const __grid_constant__ ForgeUniforms u)
FORGE_ACCUMULATE_BODY

template <class World>
__global__ void __launch_bounds__(256, 2) forgeAccumulateR128(float4 *accumulation, const __grid_constant__ ForgeUniforms u)
FORGE_ACCUMULATE_BODY

template <class World>
__global__ void __launch_bounds__(256, 4) forgeAccumulateR64(float4 *accumulation, const __grid_constant__ ForgeUniforms u)
FORGE_ACCUMULATE_BODY

#undef FORGE_ACCUMULATE_BODY

__global__ void forgeResolve(const float4 *accumulation, RGBA16 *output, const __grid_constant__ ForgeUniforms u)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int width = uint(u.imageSize.x);
    if (x >= width || y >= uint(u.imageSize.y)) { return; }
    output[y * width + x] = forgeResolvePixel(accumulation, u, x, y);
}

typedef void (*AccumulateKernel)(float4 *, const ForgeUniforms);

template <class World>
AccumulateKernel accumulateKernel(int registerCap)
{
    switch (registerCap) {
    case 128: return forgeAccumulateR128<World>;
    case 64: return forgeAccumulateR64<World>;
    default: return forgeAccumulateR0<World>;
    }
}

AccumulateKernel worldAccumulateKernel(forge::WorldKernel world, int registerCap)
{
    switch (world) {
    case forge::WorldKernel::MandelNewton: return accumulateKernel<MandelNewtonWorldKernel>(registerCap);
    }
    return nullptr;
}

} // namespace forge_gpu

namespace forge {

namespace {

double now()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string hint(cudaError_t e)
{
    switch (e) {
    case cudaErrorNoDevice:
        return " No CUDA device is visible: check nvidia-smi, CUDA_VISIBLE_DEVICES, and that the container "
               "was started with GPU access (--gpus all / NVIDIA_VISIBLE_DEVICES).";
    case cudaErrorInsufficientDriver:
        return " The NVIDIA driver is older than this build's CUDA runtime " + cudaVersionString(CUDART_VERSION) +
               ": rebuild with an older CUDA toolkit (<= the 'CUDA Version' nvidia-smi prints) or update the driver.";
    case cudaErrorNoKernelImageForDevice:
        return " This binary has no code for the GPU's architecture (compiled: " FORGE_CUDA_ARCHS
               "). Rebuild on the GPU machine (build-linux-cuda.sh detects it) or set FORGE_CUDA_ARCHS.";
    case cudaErrorLaunchOutOfResources:
        return " Too many threads per block for this kernel's registers: use a smaller --cuda-block or "
               "--cuda-register-cap 128.";
    case cudaErrorLaunchTimeout:
        return " The display watchdog killed the kernel: render on a GPU without a display attached.";
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
        return " A kernel faulted; the CUDA context is unusable. Check dmesg for Xid errors.";
    case cudaErrorMemoryAllocation:
        return " Out of GPU memory: another process may hold VRAM (nvidia-smi).";
    case cudaErrorDevicesUnavailable:
        return " The GPU is in exclusive/prohibited compute mode and busy (nvidia-smi -q -d COMPUTE).";
    default:
        return "";
    }
}

[[noreturn]] void fail(cudaError_t e, const char *what, const char *file, int line)
{
    std::ostringstream s;
    s << "CUDA error " << cudaGetErrorName(e) << " (" << static_cast<int>(e) << ") in " << what << " at " << file
      << ":" << line << ": " << cudaGetErrorString(e) << "." << hint(e);
    throw BackendError(s.str());
}

#define FORGE_CUDA(call)                                                     \
    do {                                                                     \
        const cudaError_t forge_cuda_error_ = (call);                        \
        if (forge_cuda_error_ != cudaSuccess) {                              \
            fail(forge_cuda_error_, #call, __FILE__, __LINE__);              \
        }                                                                    \
    } while (0)

std::string lower(std::string s)
{
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string uuidString(const cudaUUID_t &uuid)
{
    static const char *digits = "0123456789abcdef";
    std::string s = "GPU-";
    for (int i = 0; i < 16; ++i) {
        const unsigned char b = static_cast<unsigned char>(uuid.bytes[i]);
        s += digits[b >> 4];
        s += digits[b & 15];
        if (i == 3 || i == 5 || i == 7 || i == 9) s += '-';
    }
    return s;
}

const char *computeModeName(int mode)
{
    switch (mode) {
    case cudaComputeModeDefault: return "Default";
    case cudaComputeModeProhibited: return "Prohibited";
    case cudaComputeModeExclusiveProcess: return "ExclusiveProcess";
    default: return "Other";
    }
}

/// The NVIDIA kernel module's version (the driver version nvidia-smi shows),
/// when /proc exposes it; empty otherwise.
std::string nvidiaKernelModuleVersion()
{
    std::ifstream f("/proc/driver/nvidia/version");
    std::string line;
    if (!std::getline(f, line)) return "";
    const auto module = line.find("Kernel Module");
    std::istringstream words(module == std::string::npos ? line : line.substr(module + 13));
    std::string version;
    words >> version;
    return version;
}

std::string megabytes(unsigned long long bytes)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

CUDADeviceInfo describe(int index)
{
    cudaDeviceProp p{};
    FORGE_CUDA(cudaGetDeviceProperties(&p, index));
    CUDADeviceInfo d;
    d.index = index;
    d.name = p.name;
    d.computeMajor = p.major;
    d.computeMinor = p.minor;
    d.multiprocessors = p.multiProcessorCount;
    d.totalMemoryBytes = p.totalGlobalMem;
    // Clock attributes (cudaDevAttrClockRate etc.) are queried separately:
    // the cudaDeviceProp fields are deprecated in CUDA 12.x.
    cudaDeviceGetAttribute(&d.clockKHz, cudaDevAttrClockRate, index);
    cudaDeviceGetAttribute(&d.memoryClockKHz, cudaDevAttrMemoryClockRate, index);
    cudaDeviceGetAttribute(&d.memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, index);
    d.maxThreadsPerMultiprocessor = p.maxThreadsPerMultiProcessor;
    d.registersPerMultiprocessor = p.regsPerMultiprocessor;
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), index) == cudaSuccess) d.pciBusID = bus;
    d.uuid = uuidString(p.uuid);
    d.kernelExecTimeout = p.kernelExecTimeoutEnabled != 0;
    d.integrated = p.integrated != 0;
    d.computeMode = p.computeMode;
    d.computeModeName = computeModeName(p.computeMode);
    // Runnable: this build carries SASS the device can execute (same major
    // version, minor <= the device's) or PTX it can JIT (version <= the
    // device's). The selected device is verified again with the real kernel.
    if (p.major < 7) {
        d.unsupportedReason = "compute capability " + std::to_string(p.major) + "." + std::to_string(p.minor) +
                              " is below the minimum 7.0";
    } else if (p.computeMode == cudaComputeModeProhibited) {
        d.unsupportedReason = "compute mode Prohibited";
    } else if (!cudaArchitecturesCover(FORGE_CUDA_ARCHS, p.major, p.minor)) {
        d.unsupportedReason = "no code for sm_" + std::to_string(p.major * 10 + p.minor) + " in this build (" +
                              FORGE_CUDA_ARCHS + "); rebuild on this machine";
    }
    d.supported = d.unsupportedReason.empty();
    return d;
}

struct Selection {
    int index = -1;
    std::string mode;
};

Selection selectDevice(const std::vector<CUDADeviceInfo> &devices, const BackendOptions &o)
{
    if (devices.empty()) throw BackendError("No CUDA devices.");
    Selection s;
    if (o.cudaDevice >= 0) {
        if (o.cudaDevice >= static_cast<int>(devices.size())) {
            throw BackendError("--cuda-device " + std::to_string(o.cudaDevice) + ": only " +
                               std::to_string(devices.size()) + " CUDA device(s) are visible (0-based; "
                               "CUDA_VISIBLE_DEVICES renumbers them).");
        }
        s.index = o.cudaDevice;
        s.mode = "--cuda-device";
    } else if (!o.cudaDeviceName.empty()) {
        const std::string wanted = lower(o.cudaDeviceName);
        std::vector<int> matches;
        for (const auto &d : devices) {
            if (lower(d.name).find(wanted) != std::string::npos) matches.push_back(d.index);
        }
        if (matches.size() > 1) {
            std::vector<int> exact;
            for (int i : matches) {
                if (lower(devices[i].name) == wanted) exact.push_back(i);
            }
            if (exact.size() == 1) matches = exact;
        }
        if (matches.empty()) throw BackendError("--cuda-device-name '" + o.cudaDeviceName + "' matches no CUDA device.");
        if (matches.size() > 1) {
            throw BackendError("--cuda-device-name '" + o.cudaDeviceName + "' matches " +
                               std::to_string(matches.size()) + " devices; use --cuda-device N.");
        }
        s.index = matches.front();
        s.mode = "--cuda-device-name";
    } else {
        // Headless devices first (no display watchdog), then the most SMs,
        // then the lowest index.
        for (const auto &d : devices) {
            if (!d.supported) continue;
            if (s.index < 0) { s.index = d.index; continue; }
            const auto &best = devices[s.index];
            if ((best.kernelExecTimeout && !d.kernelExecTimeout) ||
                (best.kernelExecTimeout == d.kernelExecTimeout && d.multiprocessors > best.multiprocessors)) {
                s.index = d.index;
            }
        }
        if (s.index < 0) {
            std::string why;
            for (const auto &d : devices) why += "\n  [" + std::to_string(d.index) + "] " + d.name + ": " + d.unsupportedReason;
            throw BackendError("No usable CUDA device:" + why);
        }
        s.mode = devices.size() == 1 ? "only device" : "automatic (headless, most SMs)";
    }
    const auto &chosen = devices[s.index];
    if (!chosen.supported) {
        throw BackendError("CUDA device " + std::to_string(chosen.index) + " (" + chosen.name +
                           ") cannot run this worker: " + chosen.unsupportedReason + ".");
    }
    return s;
}

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------
class CUDABackend final : public RenderBackend {
public:
    explicit CUDABackend(const BackendOptions &options) : options_(options)
    {
        const CUDAEnvironment env = cudaEnvironment();
        if (!env.error.empty()) throw BackendError(env.error);
        driverVersion_ = env.driverVersion;
        const Selection selection = selectDevice(env.devices, options);
        info_ = env.devices[selection.index];
        selectionMode_ = selection.mode;

        // Blocking sync: the host thread sleeps while a frame renders, which
        // leaves the CPU cores to the PNG/video export threads.
        FORGE_CUDA(cudaSetDevice(info_.index));
        const cudaError_t flags = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        if (flags != cudaSuccess && flags != cudaErrorSetOnActiveProcess) FORGE_CUDA(flags);
        cudaGetLastError();
        FORGE_CUDA(cudaFree(nullptr)); // create the primary context now, so setup cost is not frame 0's

        accumulate_ = forge_gpu::worldAccumulateKernel(options.world, options.registerCap);
        if (!accumulate_) throw BackendError("The CUDA backend has no kernel for the requested world.");
        FORGE_CUDA(cudaFuncGetAttributes(&accumulateAttributes_, reinterpret_cast<const void *>(accumulate_)));
        FORGE_CUDA(cudaFuncGetAttributes(&resolveAttributes_, reinterpret_cast<const void *>(forge_gpu::forgeResolve)));

        const int threads = options.blockWidth * options.blockHeight;
        if (options.blockWidth <= 0 || options.blockHeight <= 0 || threads > accumulateAttributes_.maxThreadsPerBlock ||
            threads > resolveAttributes_.maxThreadsPerBlock) {
            throw BackendError("--cuda-block " + std::to_string(options.blockWidth) + "x" +
                               std::to_string(options.blockHeight) + " (" + std::to_string(threads) +
                               " threads) exceeds this kernel's limit of " +
                               std::to_string(accumulateAttributes_.maxThreadsPerBlock) + " threads per block (" +
                               std::to_string(accumulateAttributes_.numRegs) + " registers per thread).");
        }

        const size_t pixels = static_cast<size_t>(options.width) * options.height;
        FORGE_CUDA(cudaMalloc(&accumulation_, pixels * sizeof(forge_gpu::float4)));
        FORGE_CUDA(cudaMalloc(&output_, pixels * sizeof(forge_gpu::RGBA16)));
        FORGE_CUDA(cudaMallocHost(&staging_, pixels * sizeof(forge_gpu::RGBA16)));
        FORGE_CUDA(cudaMemset(accumulation_, 0, pixels * sizeof(forge_gpu::float4)));
        FORGE_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        for (auto &e : events_) FORGE_CUDA(cudaEventCreateWithFlags(&e, cudaEventBlockingSync));
        FORGE_CUDA(cudaDeviceSynchronize());
        size_t freeBytes = 0, totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) info_.freeMemoryBytes = freeBytes;
        allocatedBytes_ = pixels * (sizeof(forge_gpu::float4) + sizeof(forge_gpu::RGBA16));
    }

    ~CUDABackend() override
    {
        // After a timeout the kernels may still be running, and cudaFree
        // would wait for them: leave everything to process exit, which tears
        // the context down. (After a fault the calls just return errors.)
        if (unhealthy_) return;
        for (auto &e : events_) {
            if (e) cudaEventDestroy(e);
        }
        if (stream_) cudaStreamDestroy(stream_);
        if (staging_) cudaFreeHost(staging_);
        if (output_) cudaFree(output_);
        if (accumulation_) cudaFree(accumulation_);
    }

    std::string backendName() const override { return "cuda"; }
    std::string deviceName() const override { return info_.name; }
    int deviceIndex() const override { return info_.index; }

    std::vector<BackendFact> facts() const override
    {
        std::vector<BackendFact> f;
        auto add = [&](const std::string &k, const std::string &v) { f.push_back({k, v}); };
        add("backend", "CUDA runtime API, headless compute (no Vulkan, OpenGL/EGL, X11/Wayland or display)");
        add("cuda_device_index", std::to_string(info_.index));
        add("cuda_device_name", info_.name);
        add("cuda_device_selection", selectionMode_);
        add("cuda_compute_capability", std::to_string(info_.computeMajor) + "." + std::to_string(info_.computeMinor));
        add("cuda_multiprocessors", std::to_string(info_.multiprocessors));
        add("cuda_sm_clock_mhz", std::to_string(info_.clockKHz / 1000));
        add("cuda_memory_total", megabytes(info_.totalMemoryBytes));
        add("cuda_memory_free_after_allocation", megabytes(info_.freeMemoryBytes));
        add("cuda_memory_clock_mhz", std::to_string(info_.memoryClockKHz / 1000));
        add("cuda_memory_bus_width_bits", std::to_string(info_.memoryBusWidthBits));
        add("cuda_pci_bus_id", info_.pciBusID);
        add("cuda_device_uuid", info_.uuid);
        add("cuda_display_watchdog", info_.kernelExecTimeout ? "yes" : "no");
        add("cuda_compute_mode", info_.computeModeName);
        add("cuda_driver_supports_cuda", cudaVersionString(driverVersion_));
        const std::string module = nvidiaKernelModuleVersion();
        add("nvidia_driver_version", module.empty() ? "unknown (/proc/driver/nvidia/version not readable)" : module);
        add("cuda_runtime_version_built", cudaVersionString(CUDART_VERSION) + " (static cudart)");
        add("cuda_compiled_architectures", FORGE_CUDA_ARCHS);
        const int binary = accumulateAttributes_.binaryVersion;
        const int device = info_.computeMajor * 10 + info_.computeMinor;
        add("cuda_kernel_code", "sm_" + std::to_string(binary) +
                                    (binary == device ? " SASS (native)" : " (PTX " + std::to_string(accumulateAttributes_.ptxVersion) +
                                                                                  ", JIT-compiled by the driver)"));
        add("cuda_accumulate_register_cap", options_.registerCap ? std::to_string(options_.registerCap) : "none (compiler)");
        add("cuda_accumulate_registers", std::to_string(accumulateAttributes_.numRegs));
        add("cuda_accumulate_local_bytes", std::to_string(accumulateAttributes_.localSizeBytes));
        add("cuda_accumulate_max_threads_per_block", std::to_string(accumulateAttributes_.maxThreadsPerBlock));
        add("cuda_resolve_registers", std::to_string(resolveAttributes_.numRegs));
        add("cuda_block", std::to_string(options_.blockWidth) + "x" + std::to_string(options_.blockHeight));
        add("cuda_framebuffers", megabytes(allocatedBytes_) + " device (float4 accumulation + rgba16 output), " +
                                     megabytes(static_cast<unsigned long long>(options_.width) * options_.height * 8) +
                                     " pinned host staging");
        add("math", "IEEE float (no --use_fast_math; Metal renders with fastMathEnabled = false)");
        return f;
    }

    void renderFrame(const UniformBlocks &dispatches, const UniformBlocks &resolve, uint16_t *rgba16,
                     BackendFrameTimings &t) override
    {
        if (unhealthy_) throw BackendError("The CUDA context is unusable after an earlier failure.");
        checkBlocks(dispatches);
        checkBlocks(resolve);
        if (resolve.count != 1) throw BackendError("The resolve pass takes exactly one uniform block.");
        if (dispatches.count > options_.maxDispatchesPerFrame && options_.maxDispatchesPerFrame > 0) {
            throw BackendError("More dispatches than the frame plan allows.");
        }

        const double start = now();
        const dim3 block(options_.blockWidth, options_.blockHeight);
        forge_gpu::ForgeUniforms u;
        FORGE_CUDA(cudaEventRecord(events_[0], stream_));
        for (size_t i = 0; i < dispatches.count; ++i) {
            std::memcpy(&u, dispatches.block(i), sizeof(u));
            const unsigned int w = static_cast<unsigned int>(u.tileSize.x);
            const unsigned int h = static_cast<unsigned int>(u.tileSize.y);
            if (w == 0 || h == 0) continue;
            const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            accumulate_<<<grid, block, 0, stream_>>>(static_cast<forge_gpu::float4 *>(accumulation_), u);
        }
        FORGE_CUDA(cudaGetLastError());
        FORGE_CUDA(cudaEventRecord(events_[1], stream_));

        std::memcpy(&u, resolve.block(0), sizeof(u));
        const dim3 grid((options_.width + block.x - 1) / block.x, (options_.height + block.y - 1) / block.y);
        forge_gpu::forgeResolve<<<grid, block, 0, stream_>>>(static_cast<const forge_gpu::float4 *>(accumulation_),
                                                             static_cast<forge_gpu::RGBA16 *>(output_), u);
        FORGE_CUDA(cudaGetLastError());
        FORGE_CUDA(cudaEventRecord(events_[2], stream_));

        const size_t bytes = static_cast<size_t>(options_.width) * options_.height * sizeof(forge_gpu::RGBA16);
        FORGE_CUDA(cudaMemcpyAsync(staging_, output_, bytes, cudaMemcpyDeviceToHost, stream_));
        FORGE_CUDA(cudaEventRecord(events_[3], stream_));
        const double enqueued = now();
        t.enqueueCPUSeconds = enqueued - start;

        wait(events_[3]);
        const double done = now();
        t.waitSeconds = done - enqueued;
        t.gpuWallSeconds = done - start;

        float ms = 0;
        FORGE_CUDA(cudaEventElapsedTime(&ms, events_[0], events_[1]));
        t.accumulateGPUSeconds = ms / 1000.0;
        FORGE_CUDA(cudaEventElapsedTime(&ms, events_[1], events_[2]));
        t.resolveGPUSeconds = ms / 1000.0;
        FORGE_CUDA(cudaEventElapsedTime(&ms, events_[2], events_[3]));
        t.transferGPUSeconds = ms / 1000.0;

        const double copyStart = now();
        std::memcpy(rgba16, staging_, bytes);
        t.hostCopySeconds = now() - copyStart;
    }

private:
    void checkBlocks(const UniformBlocks &b) const
    {
        if (b.blockSize != sizeof(forge_gpu::ForgeUniforms) || b.stride < b.blockSize) {
            throw BackendError("Uniform block size " + std::to_string(b.blockSize) + " does not match the CUDA "
                               "ForgeUniforms (" + std::to_string(sizeof(forge_gpu::ForgeUniforms)) + " bytes).");
        }
    }

    void wait(cudaEvent_t event)
    {
        cudaError_t e;
        if (options_.gpuTimeoutSeconds <= 0) {
            e = cudaEventSynchronize(event);
        } else {
            const double deadline = now() + options_.gpuTimeoutSeconds;
            while ((e = cudaEventQuery(event)) == cudaErrorNotReady) {
                if (now() > deadline) {
                    unhealthy_ = true;
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "The GPU did not finish the frame within --gpu-timeout-seconds %.0f.",
                                  options_.gpuTimeoutSeconds);
                    throw BackendError(buf);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        if (e != cudaSuccess) {
            unhealthy_ = true;
            fail(e, "frame completion", __FILE__, __LINE__);
        }
    }

    BackendOptions options_;
    CUDADeviceInfo info_;
    std::string selectionMode_;
    int driverVersion_ = 0;
    forge_gpu::AccumulateKernel accumulate_ = nullptr;
    cudaFuncAttributes accumulateAttributes_{};
    cudaFuncAttributes resolveAttributes_{};
    void *accumulation_ = nullptr;
    void *output_ = nullptr;
    void *staging_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t events_[4] = {nullptr, nullptr, nullptr, nullptr};
    unsigned long long allocatedBytes_ = 0;
    bool unhealthy_ = false;
};

} // namespace

bool cudaCompiledIn() { return true; }

std::string cudaVersionString(int version)
{
    if (version <= 0) return "none";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

CUDAEnvironment cudaEnvironment()
{
    CUDAEnvironment env;
    env.compiledIn = true;
    env.runtimeVersion = CUDART_VERSION;
    env.compiledArchitectures = FORGE_CUDA_ARCHS;
    cudaDriverGetVersion(&env.driverVersion);
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        cudaGetLastError();
        if (env.driverVersion == 0) {
            // CUDA reports a missing driver as "insufficient"; say what it is.
            env.error = "CUDA is not available: no NVIDIA driver (libcuda.so.1) is visible to this process. "
                        "Check nvidia-smi; in a container, start it with GPU access (--gpus all, or a GPU pod).";
        } else {
            env.error = std::string("CUDA is not available: ") + cudaGetErrorName(e) + ": " + cudaGetErrorString(e) + "." + hint(e);
        }
        return env;
    }
    try {
        for (int i = 0; i < count; ++i) env.devices.push_back(describe(i));
    } catch (const BackendError &error) {
        env.error = error.what();
    }
    return env;
}

std::unique_ptr<RenderBackend> makeCUDABackend(const BackendOptions &options)
{
    return std::unique_ptr<RenderBackend>(new CUDABackend(options));
}

/// Free device memory, which needs a context per device: only for listings.
void cudaQueryFreeMemory(std::vector<CUDADeviceInfo> &devices)
{
    for (auto &d : devices) {
        size_t freeBytes = 0, totalBytes = 0;
        if (cudaSetDevice(d.index) == cudaSuccess && cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
            d.freeMemoryBytes = freeBytes;
        }
        cudaGetLastError();
    }
}

} // namespace forge
