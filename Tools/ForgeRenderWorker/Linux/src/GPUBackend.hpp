//
//  GPUBackend.hpp
//  Cyanescent Forge — Linux worker
//
//  The seam between the portable worker (timeline, camera, parameters,
//  uniforms, export — plain C++, built with the host compiler) and a render
//  backend (CUDA, or the CPU emulation of the same kernels).
//
//  It deliberately uses no Forge vector types and no CUDA types: the host's
//  float2/float3/float4 (ForgeHostTypes.hpp) and CUDA's vector_types.h both
//  claim those names, so backend implementations never include the host
//  types. Uniforms cross this seam as the bytes of the host's ForgeUniforms,
//  which has the Metal layout; each backend compiles the same
//  ForgeShaderTypes.h with Metal-layout vectors and checks, at build time,
//  that every field lands at the same offset (tools/check_cuda_layout.py).
//
//  Render model (ForgeRenderer's): per frame, a list of accumulation
//  dispatches — tiles x sample batches, in submission order, each with its
//  own ForgeUniforms (tileOrigin/tileSize, sampleOffset, accumulate flag) —
//  then one full-image resolve into 16-bit RGBA, then one device-to-host
//  copy. Nothing else crosses between CPU and GPU during a frame.
//
#ifndef FORGE_GPU_BACKEND_HPP
#define FORGE_GPU_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forge {

/// A backend failure (CUDA error, no device, ...). The message is complete
/// and printable; the worker exits 1.
class BackendError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// World kernels a backend can run (forge_gpu::ForgeWorldKernel).
enum class WorldKernel : int {
    MandelNewton = 1,
};

/// Uniform blocks as the host laid them out: `count` blocks of `blockSize`
/// bytes, `stride` bytes apart.
struct UniformBlocks {
    const unsigned char *bytes = nullptr;
    size_t blockSize = 0;
    size_t stride = 0;
    size_t count = 0;

    const unsigned char *block(size_t i) const { return bytes + i * stride; }
};

struct BackendOptions {
    WorldKernel world = WorldKernel::MandelNewton;
    int width = 0;
    int height = 0;
    /// Largest tile (ForgeQuality tile size), for launch-size checks.
    int tileSize = 0;
    /// Upper bound on accumulation dispatches per frame (tiles x batches).
    size_t maxDispatchesPerFrame = 0;

    // CUDA
    int cudaDevice = -1;          // -1: select automatically
    std::string cudaDeviceName;   // case-insensitive substring; empty: unused
    int blockWidth = 8;           // threads per block: blockWidth x blockHeight
    int blockHeight = 8;          //   (Metal's threadgroup is 8x8)
    int registerCap = 0;          // accumulate-kernel variant: 0 (compiler), 128, 64
    double gpuTimeoutSeconds = 0; // 0: wait forever

    // CPU emulation
    int cpuThreads = 0; // 0: all hardware threads
};

/// Per-frame timings, in seconds. GPU durations come from device events
/// (CUDA) or host clocks (CPU emulation). Uniforms travel as kernel
/// parameters (__grid_constant__), so there is no separate upload.
struct BackendFrameTimings {
    double enqueueCPUSeconds = 0;     // host time launching the frame's work
    double accumulateGPUSeconds = 0;  // all accumulation dispatches (GPU-timed)
    double resolveGPUSeconds = 0;     // resolve pass (GPU-timed)
    double transferGPUSeconds = 0;    // framebuffer device -> host (GPU-timed)
    double hostCopySeconds = 0;       // pinned staging -> caller's buffer
    double waitSeconds = 0;           // host blocked waiting for the GPU
    double gpuWallSeconds = 0;        // first enqueue to completion, host clock
};

/// One labelled fact about the device/backend, for logs and benchmark.json.
struct BackendFact {
    std::string key;
    std::string value;
};

class RenderBackend {
public:
    virtual ~RenderBackend() = default;

    /// "cuda" or "cpu-emulation".
    virtual std::string backendName() const = 0;
    virtual std::string deviceName() const = 0;
    virtual int deviceIndex() const = 0;
    /// Ordered facts: device, driver/runtime, build (see each backend).
    virtual std::vector<BackendFact> facts() const = 0;

    /// Renders one frame: `dispatches` accumulate in order, `resolve` (one
    /// block) resolves the full image; `rgba16` receives width*height*4
    /// values, rows top to bottom.
    virtual void renderFrame(const UniformBlocks &dispatches, const UniformBlocks &resolve,
                             uint16_t *rgba16, BackendFrameTimings &timings) = 0;
};

// --- CUDA (CUDABackend.cu; absent when built without nvcc) --------------------

struct CUDADeviceInfo {
    int index = 0;
    std::string name;
    int computeMajor = 0, computeMinor = 0;
    int multiprocessors = 0;
    unsigned long long totalMemoryBytes = 0;
    unsigned long long freeMemoryBytes = 0; // 0 when the device could not be queried
    int clockKHz = 0;
    int memoryClockKHz = 0;
    int memoryBusWidthBits = 0;
    int maxThreadsPerMultiprocessor = 0;
    int registersPerMultiprocessor = 0;
    std::string pciBusID;
    std::string uuid;
    bool kernelExecTimeout = false; // a display watchdog applies to this device
    bool integrated = false;
    int computeMode = 0;            // cudaComputeMode
    std::string computeModeName;
    bool supported = false;         // meets this build's minimum (sm_70+ and a compiled arch)
    std::string unsupportedReason;
};

struct CUDAEnvironment {
    bool compiledIn = false;       // built with nvcc
    int runtimeVersion = 0;        // CUDART_VERSION of the build (e.g. 12040)
    int driverVersion = 0;         // cudaDriverGetVersion: highest CUDA the driver supports (0: no driver)
    std::string compiledArchitectures; // e.g. "sm_75 sm_80 sm_86 sm_89 sm_90 compute_90"
    std::string error;             // why devices could not be enumerated, if so
    std::vector<CUDADeviceInfo> devices;
};

bool cudaCompiledIn();
/// Never throws: failures are reported in CUDAEnvironment::error.
CUDAEnvironment cudaEnvironment();
std::string cudaVersionString(int version); // 12040 -> "12.4"
/// Fills CUDADeviceInfo::freeMemoryBytes (creates a context per device: listings only).
void cudaQueryFreeMemory(std::vector<CUDADeviceInfo> &devices);
/// Selects and initialises a device, allocates the framebuffers. Throws BackendError.
std::unique_ptr<RenderBackend> makeCUDABackend(const BackendOptions &options);

// --- CPU emulation (CPUEmulationBackend.cpp) ----------------------------------
//
// The same kernel source the CUDA backend compiles (the shipping MSL through
// cuda/ForgeShaderPort.cuh), executed on CPU threads with the same tile x
// batch x resolve structure. For tests and diagnostics without an NVIDIA GPU;
// never selected automatically.
std::unique_ptr<RenderBackend> makeCPUEmulationBackend(const BackendOptions &options);

/// sizeof(forge_gpu::ForgeUniforms) as the backends compiled it.
size_t backendUniformBlockSize();

} // namespace forge

#endif // FORGE_GPU_BACKEND_HPP
