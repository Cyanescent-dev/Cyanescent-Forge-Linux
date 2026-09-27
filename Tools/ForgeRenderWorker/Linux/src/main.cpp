//
//  main.cpp — ForgeRenderWorker for Linux (CUDA), V0.1
//  Cyanescent Forge
//
//  The Linux counterpart of Tools/ForgeRenderWorker/main.swift for headless
//  NVIDIA cloud GPUs. Same CLI concepts, frame numbering and output layout as
//  the macOS worker; the Mandelbrot / Newton Morph world rendered by the CUDA
//  backend (cuda/CUDABackend.cu) from the shipping Metal shader source.
//
//  Frame N is evaluated as render index N - base of the project's range,
//  exactly as the macOS worker and Forge's OfflineRenderer do, so any part
//  of a range reproduces the frames a whole-range render would.
//
//  The portable pieces are shared with the (retired) Vulkan worker:
//  ForgeTimeline, MandelNewtonWorld, ForgeFrame, SnapshotHash, PNGWriter.
//
//  Not in V0.1: --project, --job (remote jobs, status.json, resume),
//  --dual-gpu / multi-GPU, other worlds, loop closure, field export.
//
#include "ForgeFrame.hpp"
#include "ForgeTimeline.hpp"
#include "FrameExport.hpp"
#include "GPUBackend.hpp"
#include "Json.hpp"
#include "MandelNewtonWorld.hpp"
#include "PNGWriter.hpp"
#include "SnapshotHash.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifndef FORGE_WORKER_SOURCE_REVISION
#define FORGE_WORKER_SOURCE_REVISION "unknown"
#endif

using namespace forge;

namespace {

const char *kWorkerVersion = "linux-cuda-0.1";

// --- errors -------------------------------------------------------------------
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct HelpRequested {};

double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// --- system information ---------------------------------------------------------
std::string buildArchitecture() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
}

std::string machineArchitecture() {
    struct utsname u {};
    return uname(&u) == 0 ? u.machine : "unknown";
}

std::string osDescription() {
    std::string pretty;
    std::ifstream release("/etc/os-release");
    std::string line;
    while (std::getline(release, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            pretty = line.substr(12);
            if (pretty.size() >= 2 && pretty.front() == '"') pretty = pretty.substr(1, pretty.size() - 2);
        }
    }
    struct utsname u {};
    if (uname(&u) == 0) {
        pretty += (pretty.empty() ? "" : "; ") + std::string(u.sysname) + " " + u.release;
    }
    return pretty.empty() ? "Linux" : pretty;
}

std::string cpuModel() {
    std::ifstream info("/proc/cpuinfo");
    std::string line, implementer, part;
    while (std::getline(info, line)) {
        auto value = [&]() {
            auto colon = line.find(':');
            std::string v = colon == std::string::npos ? "" : line.substr(colon + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            return v;
        };
        if (line.rfind("model name", 0) == 0) return value();
        if (line.rfind("CPU implementer", 0) == 0 && implementer.empty()) implementer = value();
        if (line.rfind("CPU part", 0) == 0 && part.empty()) part = value();
    }
    // aarch64 /proc/cpuinfo has no model name; report the MIDR fields.
    if (!implementer.empty()) return "ARM CPU (implementer " + implementer + ", part " + part + ")";
    return "Unknown";
}

int hardwareThreads() { return std::max(1, static_cast<int>(std::thread::hardware_concurrency())); }

// --- options --------------------------------------------------------------------
enum class Backend { CUDA, CPUEmulation };

struct Options {
    std::string world = MandelNewtonWorld::cliName;
    int width = 1920;
    int height = 1080;
    double fps = 30;
    int frames = 300;
    int firstFrame = 0;
    std::vector<int> frameList;
    uint32_t seed = 12345;
    std::string output = "RenderWorkerOutput";
    QualityPreset quality = QualityPreset::OfflineHigh;

    Backend backend = Backend::CUDA;
    bool listDevices = false;
    int cudaDevice = -1;
    std::string cudaDeviceName;
    int blockWidth = 8, blockHeight = 8;
    int registerCap = 0;
    double gpuTimeoutSeconds = 0;
    int cpuThreads = 0;

    bool png = true;
    int pngThreads = -1; // -1: automatic
    std::string mp4;
    MP4Quality mp4Quality = MP4Quality::High;
    int warmupFrames = 0;

    bool capabilities = false;
    bool version = false;
    std::vector<int> snapshotHashFrames;
    std::string snapshotDump;
    std::string dumpUniforms;
};

const char *kUsage = R"(Forge Render Worker (Linux, CUDA) — V0.1: Mandelbrot / Newton Morph

Usage:
  ForgeRenderWorker [--world mandelnewton] [options]

Frames:
  --width N            Output width (default 1920)
  --height N           Output height (default 1080)
  --fps N              Frame rate (default 30)
  --frames N           Sequential frame count (default 300)
  --first-frame N      Absolute first frame (default 0)
  --frame-list A,B,…   Render exactly these absolute frames
  --seed N             Deterministic project seed (default 12345)
  --quality NAME       standard|high|extreme|quick|fast (default high = Offline — High)
  --output DIR         PNG and benchmark.json destination (default ./RenderWorkerOutput)

Output:
  --no-png             Render and read back, but write no PNGs (GPU benchmarking)
  --png-threads N      PNG encoder threads overlapping the next frame's render
                       (default: half the CPU threads, 1..8; 0 = encode inline)
  --mp4 FILE           After the frames, encode FILE from the PNG sequence with
                       Forge's ffmpeg settings (H.264, yuv420p); needs contiguous frames
  --mp4-quality Q      standard|high|very-high (default high: -preset slow -crf 18)
  --warmup-frames N    Render the first frame N extra times, untimed and unsaved, first
  --benchmark          Accepted for clarity; benchmark.json is always written

Renderer:
  --backend NAME       cuda (default) | cpu-emulation (the CUDA kernels run on the CPU;
                       for tests without an NVIDIA GPU — slow)
  --list-gpus          List CUDA devices, driver and build details (alias --list-cuda-devices)
  --cuda-device N      Use CUDA device N (0-based, after CUDA_VISIBLE_DEVICES)
  --cuda-device-name S Use the one CUDA device whose name contains S (case-insensitive)
                       (default: headless devices first, then the most SMs)
  --cuda-block WxH     Threads per block (default 8x8, Metal's threadgroup size)
  --cuda-register-cap N  Accumulate-kernel variant: 0 (compiler's choice, default), 128, 64
  --gpu-timeout-seconds S  Fail if a frame takes longer than S on the GPU (default 0 = no limit)
  --cpu-threads N      Threads for --backend cpu-emulation (default: all)

Information:
  --capabilities       Print worker, device and world capabilities as JSON
  --version            Print the worker version

Diagnostics:
  --snapshot-hashes F,…  Print SHA-256 hashes of the renderer-facing frame state
                         (frame, loopClosure, sceneParams, camera sections) for
                         absolute frames F,… and exit; compare with the macOS
                         worker's --snapshot-hashes. No GPU work.
  --snapshot-dump DIR    With --snapshot-hashes, also write each section's bytes
  --dump-uniforms DIR    Also write each frame's full-image uniforms (canonical
                         words) for fidelity/MandelNewtonCPUReference

Not implemented on Linux: --project, --job, --dual-gpu, other worlds.
Exit codes: 0 completed, 1 failed, 2 usage.
)";

std::vector<int> parseFrameList(const std::string &text, const char *flag) {
    std::vector<int> list;
    std::stringstream s(text);
    std::string item;
    while (std::getline(s, item, ',')) {
        char *end = nullptr;
        errno = 0;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || errno != 0 || v < 0 || v > INT_MAX) {
            throw UsageError(std::string(flag) + " takes non-negative frame numbers, e.g. 0,300,600");
        }
        list.push_back(static_cast<int>(v));
    }
    if (list.empty()) throw UsageError(std::string(flag) + " takes frame numbers, e.g. 0,300");
    return list;
}

long parseInteger(const std::string &text, const char *flag, long lo, long hi) {
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (text.empty() || *end != '\0' || errno != 0 || v < lo || v > hi) {
        throw UsageError(std::string(flag) + " must be an integer between " + std::to_string(lo) + " and " +
                         std::to_string(hi) + ".");
    }
    return v;
}

double parseSeconds(const std::string &text, const char *flag) {
    char *end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (text.empty() || *end != '\0' || !std::isfinite(v) || v < 0) {
        throw UsageError(std::string(flag) + " takes a non-negative number of seconds.");
    }
    return v;
}

Options parseOptions(int argc, char **argv) {
    Options o;
    bool firstFrameGiven = false;
    std::vector<std::string> args(argv + 1, argv + argc);
    size_t i = 0;
    auto take = [&](const std::string &flag) -> std::string {
        if (i >= args.size()) throw UsageError("Missing value for " + flag + ".");
        return args[i++];
    };
    while (i < args.size()) {
        const std::string flag = args[i++];
        if (flag == "--world") {
            o.world = take(flag);
        } else if (flag == "--width") {
            o.width = static_cast<int>(parseInteger(take(flag), "--width", 1, 16384));
        } else if (flag == "--height") {
            o.height = static_cast<int>(parseInteger(take(flag), "--height", 1, 16384));
        } else if (flag == "--fps") {
            const std::string v = take(flag);
            char *end = nullptr;
            o.fps = std::strtod(v.c_str(), &end);
            if (v.empty() || *end != '\0' || !std::isfinite(o.fps) || o.fps <= 0 || o.fps > 1000) {
                throw UsageError("FPS must be greater than 0 and at most 1000.");
            }
        } else if (flag == "--frames") {
            o.frames = static_cast<int>(parseInteger(take(flag), "--frames", 1, 1000000));
        } else if (flag == "--first-frame") {
            o.firstFrame = static_cast<int>(parseInteger(take(flag), "--first-frame", 0, INT_MAX / 2));
            firstFrameGiven = true;
        } else if (flag == "--frame-list") {
            o.frameList = parseFrameList(take(flag), "--frame-list");
            std::sort(o.frameList.begin(), o.frameList.end());
            if (std::adjacent_find(o.frameList.begin(), o.frameList.end()) != o.frameList.end()) {
                throw UsageError("--frame-list takes distinct absolute frame numbers.");
            }
        } else if (flag == "--seed") {
            o.seed = static_cast<uint32_t>(parseInteger(take(flag), "--seed", 0, 4294967295L));
        } else if (flag == "--output") {
            o.output = take(flag);
        } else if (flag == "--quality") {
            const std::string name = take(flag);
            auto q = qualityFromCLIName(name);
            if (!q) throw UsageError("Quality must be standard, high, extreme, quick, or fast.");
            o.quality = *q;
        } else if (flag == "--backend") {
            const std::string name = take(flag);
            if (name == "cuda") o.backend = Backend::CUDA;
            else if (name == "cpu-emulation" || name == "cpu") o.backend = Backend::CPUEmulation;
            else throw UsageError("--backend must be cuda or cpu-emulation.");
        } else if (flag == "--list-gpus" || flag == "--list-cuda-devices") {
            o.listDevices = true;
        } else if (flag == "--cuda-device" || flag == "--gpu-index") {
            o.cudaDevice = static_cast<int>(parseInteger(take(flag), flag.c_str(), 0, 1024));
        } else if (flag == "--cuda-device-name") {
            o.cudaDeviceName = take(flag);
            if (o.cudaDeviceName.empty()) throw UsageError("--cuda-device-name needs a non-empty name.");
        } else if (flag == "--cuda-block") {
            const std::string v = take(flag);
            const auto x = v.find('x');
            if (x == std::string::npos) throw UsageError("--cuda-block takes WxH, e.g. 8x8 or 16x8.");
            o.blockWidth = static_cast<int>(parseInteger(v.substr(0, x), "--cuda-block width", 1, 1024));
            o.blockHeight = static_cast<int>(parseInteger(v.substr(x + 1), "--cuda-block height", 1, 1024));
            if (o.blockWidth * o.blockHeight > 1024 || (o.blockWidth * o.blockHeight) % 32 != 0) {
                throw UsageError("--cuda-block must have a multiple of 32 threads, at most 1024 (e.g. 8x8, 16x8, 16x16).");
            }
        } else if (flag == "--cuda-register-cap") {
            o.registerCap = static_cast<int>(parseInteger(take(flag), "--cuda-register-cap", 0, 255));
            if (o.registerCap != 0 && o.registerCap != 128 && o.registerCap != 64) {
                throw UsageError("--cuda-register-cap must be 0, 128 or 64 (the compiled variants).");
            }
        } else if (flag == "--gpu-timeout-seconds") {
            o.gpuTimeoutSeconds = parseSeconds(take(flag), "--gpu-timeout-seconds");
        } else if (flag == "--cpu-threads") {
            o.cpuThreads = static_cast<int>(parseInteger(take(flag), "--cpu-threads", 1, 4096));
        } else if (flag == "--no-png") {
            o.png = false;
        } else if (flag == "--png-threads") {
            o.pngThreads = static_cast<int>(parseInteger(take(flag), "--png-threads", 0, 64));
        } else if (flag == "--mp4") {
            o.mp4 = take(flag);
        } else if (flag == "--mp4-quality") {
            const std::string q = take(flag);
            if (q == "standard") o.mp4Quality = MP4Quality::Standard;
            else if (q == "high") o.mp4Quality = MP4Quality::High;
            else if (q == "very-high" || q == "veryHigh") o.mp4Quality = MP4Quality::VeryHigh;
            else throw UsageError("--mp4-quality must be standard, high or very-high.");
        } else if (flag == "--warmup-frames") {
            o.warmupFrames = static_cast<int>(parseInteger(take(flag), "--warmup-frames", 0, 100));
        } else if (flag == "--benchmark") {
            // Every run writes timing data; accepted for the documented invocation.
        } else if (flag == "--capabilities") {
            o.capabilities = true;
        } else if (flag == "--version") {
            o.version = true;
        } else if (flag == "--snapshot-hashes") {
            o.snapshotHashFrames = parseFrameList(take(flag), "--snapshot-hashes");
        } else if (flag == "--snapshot-dump") {
            o.snapshotDump = take(flag);
        } else if (flag == "--dump-uniforms") {
            o.dumpUniforms = take(flag);
        } else if (flag == "--help" || flag == "-h") {
            throw HelpRequested{};
        } else if (flag.rfind("--vulkan", 0) == 0 || flag == "--list-vulkan-devices" ||
                   flag == "--require-discrete-gpu" || flag == "--shader-dir" || flag == "--pipeline-cache-dir" ||
                   flag == "--no-pipeline-cache" || flag == "--debug-force-noncoherent") {
            throw UsageError(flag + " belongs to the retired Vulkan worker; this worker renders with CUDA "
                             "(see --list-gpus, --cuda-device).");
        } else if (flag == "--project" || flag == "--job" || flag == "--dual-gpu" || flag == "--gpu" ||
                   flag == "--display-tile-size") {
            throw UsageError(flag + " is not implemented by the Linux worker yet.");
        } else {
            throw UsageError("Unknown option: " + flag);
        }
    }
    if (o.world != MandelNewtonWorld::cliName && o.world != MandelNewtonWorld::identifier) {
        throw UsageError("--world must be mandelnewton; the Linux CUDA V0.1 worker renders only Mandelbrot / Newton.");
    }
    if (firstFrameGiven && !o.frameList.empty()) {
        throw UsageError("Use either --first-frame/--frames or --frame-list, not both.");
    }
    if (o.cudaDevice >= 0 && !o.cudaDeviceName.empty()) {
        throw UsageError("Use either --cuda-device or --cuda-device-name, not both.");
    }
    if (!o.mp4.empty()) {
        if (!o.png) throw UsageError("--mp4 encodes the PNG sequence; it cannot be combined with --no-png.");
        if (!o.frameList.empty()) {
            for (size_t k = 1; k < o.frameList.size(); ++k) {
                if (o.frameList[k] != o.frameList[k - 1] + 1) {
                    throw UsageError("--mp4 needs contiguous frames; --frame-list has gaps.");
                }
            }
        }
    }
    return o;
}

// --- files ----------------------------------------------------------------------
void makeDirectories(const std::string &path) {
    std::string partial;
    std::stringstream s(path);
    std::string part;
    if (!path.empty() && path[0] == '/') partial = "/";
    while (std::getline(s, part, '/')) {
        if (part.empty()) continue;
        partial += part + "/";
        if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
            throw std::runtime_error("Cannot create directory " + partial + ": " + std::strerror(errno));
        }
    }
}

bool fileExists(const std::string &path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

// --- plan -----------------------------------------------------------------------
struct RenderPlan {
    WorkerProject project;
    int base = 0;
    std::vector<int> absoluteFrames;
    std::string output;

    int firstFrame() const { return absoluteFrames.empty() ? base : absoluteFrames.front(); }
};

RenderPlan makePlan(const Options &o) {
    RenderPlan plan;
    if (!o.frameList.empty()) {
        plan.absoluteFrames = o.frameList;
    } else {
        for (int f = 0; f < o.frames; ++f) plan.absoluteFrames.push_back(o.firstFrame + f);
    }
    plan.project = makeWorkerProject(o.width, o.height, o.fps, o.seed, o.quality, plan.absoluteFrames.back());
    plan.base = 0;
    plan.output = o.output;
    const int end = plan.base + plan.project.renderFrameCount();
    if (plan.absoluteFrames.front() < plan.base || plan.absoluteFrames.back() >= end) {
        throw UsageError("Frames must lie inside the project's render range " + std::to_string(plan.base) + "…" +
                         std::to_string(end - 1) + ".");
    }
    return plan;
}

// --- frame state ------------------------------------------------------------------
struct FrameState {
    int frame;
    FrameRequest request;
    EvaluatedFrame evaluated;
    double timelineSeconds;
};

FrameState makeFrameState(const RenderPlan &plan, int absoluteFrame) {
    const double start = now();
    FrameState s;
    s.frame = absoluteFrame;
    s.request = frameRequest(plan.project, absoluteFrame);
    s.evaluated = evaluateFrame(plan.project, s.request);
    s.timelineSeconds = now() - start;
    return s;
}

/// One frame's uniforms in ForgeRenderer's dispatch order: every tile of
/// sample batch 0, then batch 1, ...; plus the full-image resolve uniforms.
struct FrameUniforms {
    std::vector<ForgeUniforms> dispatches;
    ForgeUniforms resolve;
};

FrameUniforms makeFrameUniforms(const RenderPlan &plan, const FrameState &state, const std::vector<Tile> &frameTiles,
                                const std::vector<SampleBatch> &batches) {
    FrameUniforms f;
    f.dispatches.reserve(frameTiles.size() * batches.size());
    for (size_t b = 0; b < batches.size(); ++b) {
        for (const Tile &tile : frameTiles) {
            f.dispatches.push_back(makeUniforms(plan.project, state.request, tile, batches[b].offset, batches[b].count,
                                                b > 0, state.evaluated));
        }
    }
    const Tile full{0, 0, plan.project.width, plan.project.height};
    f.resolve = makeUniforms(plan.project, state.request, full, 0, state.request.quality.samplesPerPixel, true,
                             state.evaluated);
    return f;
}

UniformBlocks blocks(const ForgeUniforms *u, size_t count) {
    UniformBlocks b;
    b.bytes = reinterpret_cast<const unsigned char *>(u);
    b.blockSize = sizeof(ForgeUniforms);
    b.stride = sizeof(ForgeUniforms);
    b.count = count;
    return b;
}

int printSnapshotHashes(const Options &o) {
    const RenderPlan plan = makePlan([&] {
        Options copy = o;
        // Hash frames lie inside the range the render would have.
        const int last = *std::max_element(o.snapshotHashFrames.begin(), o.snapshotHashFrames.end());
        if (copy.frameList.empty() && copy.firstFrame + copy.frames - 1 < last) copy.frames = last - copy.firstFrame + 1;
        return copy;
    }());
    const int end = plan.base + plan.project.renderFrameCount();
    if (!o.snapshotDump.empty()) makeDirectories(o.snapshotDump);
    Json rows = Json::array();
    std::vector<int> frames = o.snapshotHashFrames;
    std::sort(frames.begin(), frames.end());
    for (int frame : frames) {
        if (frame < plan.base || frame >= end) {
            throw UsageError("Frame " + std::to_string(frame) + " is outside the project's render range.");
        }
        const FrameState s = makeFrameState(plan, frame);
        const SnapshotSections h = snapshotHashes(frame, plan.project, s.request, s.evaluated);
        Json row = Json::object();
        row["frame"] = frame;
        row["snapshotSHA256"] = nullptr;
        Json sections = Json::object();
        for (const auto &kv : h.hashes) sections[kv.first] = kv.second;
        row["sections"] = sections;
        Json omitted = Json::array();
        omitted.push("project");
        row["sectionsNotComputed"] = omitted;
        rows.push(row);
        if (!o.snapshotDump.empty()) {
            for (const auto &kv : h.bytes) {
                writeFileAtomically(o.snapshotDump + "/frame_" + std::to_string(frame) + "_" + kv.first + ".bin",
                                    kv.second);
            }
        }
        std::fprintf(stderr, "frame %d sceneParams SHA256: %s\n", frame, h.hashes[2].second.c_str());
    }
    Json doc = Json::object();
    doc["workerVersion"] = kWorkerVersion;
    doc["architecture"] = buildArchitecture();
    doc["world"] = MandelNewtonWorld::identifier;
    doc["sourceRevision"] = FORGE_WORKER_SOURCE_REVISION;
    doc["os"] = osDescription();
    doc["hashing"] = "SHA-256 of a canonical little-endian field-by-field serialization (SnapshotHash.swift / SnapshotHash.cpp); "
                     "compare per section — the project section is not computed on Linux";
    doc["frames"] = rows;
    std::printf("%s\n", doc.dump().c_str());
    return 0;
}

// --- devices --------------------------------------------------------------------
std::string mib(unsigned long long bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

Json cudaDeviceJson(const CUDADeviceInfo &d) {
    Json j = Json::object();
    // The macOS worker's device keys first (a Forge controller reads these).
    j["index"] = d.index;
    j["name"] = d.name;
    j["registryID"] = nullptr;
    j["headless"] = !d.kernelExecTimeout;
    j["lowPower"] = d.integrated;
    j["computeCapability"] = std::to_string(d.computeMajor) + "." + std::to_string(d.computeMinor);
    j["multiprocessors"] = d.multiprocessors;
    j["totalMemoryBytes"] = d.totalMemoryBytes;
    if (d.freeMemoryBytes) j["freeMemoryBytes"] = d.freeMemoryBytes;
    j["smClockMHz"] = d.clockKHz / 1000;
    j["memoryClockMHz"] = d.memoryClockKHz / 1000;
    j["memoryBusWidthBits"] = d.memoryBusWidthBits;
    j["pciBusID"] = d.pciBusID;
    j["uuid"] = d.uuid;
    j["displayWatchdog"] = d.kernelExecTimeout;
    j["computeMode"] = d.computeModeName;
    j["supported"] = d.supported;
    if (!d.supported) j["unsupportedReason"] = d.unsupportedReason;
    return j;
}

void printCUDAEnvironmentHeader(const CUDAEnvironment &env) {
    std::printf("CUDA build: %s runtime (static), compiled for: %s\n",
                env.compiledIn ? cudaVersionString(env.runtimeVersion).c_str() : "not compiled in",
                env.compiledIn ? env.compiledArchitectures.c_str() : "-");
    std::printf("NVIDIA driver supports CUDA: %s\n", cudaVersionString(env.driverVersion).c_str());
    std::fflush(stdout);
}

int listDevices(const Options &o) {
    CUDAEnvironment env = cudaEnvironment();
    printCUDAEnvironmentHeader(env);
    if (!env.error.empty()) {
        std::printf("\n%s\n", env.error.c_str());
        return 1;
    }
    cudaQueryFreeMemory(env.devices);
    std::printf("CUDA devices (%zu):\n", env.devices.size());
    for (const auto &d : env.devices) {
        std::printf("\n[%d] %s\n", d.index, d.name.c_str());
        std::printf("    compute capability %d.%d, %d SMs at %d MHz, %d max threads/SM, %d registers/SM\n",
                    d.computeMajor, d.computeMinor, d.multiprocessors, d.clockKHz / 1000,
                    d.maxThreadsPerMultiprocessor, d.registersPerMultiprocessor);
        std::printf("    memory %s total, %s free, %d-bit at %d MHz\n", mib(d.totalMemoryBytes).c_str(),
                    d.freeMemoryBytes ? mib(d.freeMemoryBytes).c_str() : "?", d.memoryBusWidthBits,
                    d.memoryClockKHz / 1000);
        std::printf("    PCI %s   %s\n", d.pciBusID.c_str(), d.uuid.c_str());
        std::printf("    display watchdog: %s   compute mode: %s%s\n", d.kernelExecTimeout ? "YES" : "no",
                    d.computeModeName.c_str(), d.integrated ? "   integrated" : "");
        std::printf("    %s\n", d.supported ? "usable by this build" : ("NOT usable: " + d.unsupportedReason).c_str());
    }
    (void)o;
    return 0;
}

// --- benchmark --------------------------------------------------------------------
struct StageSample {
    int frame = 0;
    double timelineCameraCPU = 0;
    double uniformBuild = 0;
    double enqueueCPU = 0;
    double gpuAccumulate = 0;
    double gpuResolve = 0;
    double gpuToCPUTransfer = 0;
    double hostCopy = 0;
    double gpuWait = 0;
    double gpuWall = 0;
    double pngQueueWait = 0; // render loop blocked on a full PNG queue
    double pngEncode = 0;
    double pngWrite = 0;
    double renderLoop = 0;   // what this frame cost the render loop
    double total = 0;        // render loop + PNG encode + write, as if serial
};

const std::vector<std::pair<const char *, double StageSample::*>> &stageFields() {
    static const std::vector<std::pair<const char *, double StageSample::*>> fields = {
        {"timeline_camera_parameter_cpu", &StageSample::timelineCameraCPU},
        {"uniform_build_cpu", &StageSample::uniformBuild},
        {"kernel_launch_cpu", &StageSample::enqueueCPU},
        {"gpu_accumulate_kernels", &StageSample::gpuAccumulate},
        {"gpu_resolve_kernel", &StageSample::gpuResolve},
        {"gpu_to_cpu_transfer", &StageSample::gpuToCPUTransfer},
        {"host_copy", &StageSample::hostCopy},
        {"gpu_wait", &StageSample::gpuWait},
        {"gpu_wall", &StageSample::gpuWall},
        {"png_queue_wait", &StageSample::pngQueueWait},
        {"png_encode", &StageSample::pngEncode},
        {"png_write", &StageSample::pngWrite},
        {"render_loop", &StageSample::renderLoop},
    };
    return fields;
}

Json stageDocument(const StageSample &s) {
    Json j = Json::object();
    j["frame"] = s.frame;
    for (const auto &f : stageFields()) j[std::string(f.first) + "_seconds"] = s.*(f.second);
    j["total_frame_seconds"] = s.total;
    return j;
}

std::unique_ptr<RenderBackend> makeBackend(const Options &o, const RenderPlan &plan, int tileSize,
                                           size_t dispatchesPerFrame) {
    BackendOptions b;
    b.world = WorldKernel::MandelNewton;
    b.width = plan.project.width;
    b.height = plan.project.height;
    b.tileSize = tileSize;
    b.maxDispatchesPerFrame = dispatchesPerFrame;
    b.cudaDevice = o.cudaDevice;
    b.cudaDeviceName = o.cudaDeviceName;
    b.blockWidth = o.blockWidth;
    b.blockHeight = o.blockHeight;
    b.registerCap = o.registerCap;
    b.gpuTimeoutSeconds = o.gpuTimeoutSeconds;
    b.cpuThreads = o.cpuThreads;
    return o.backend == Backend::CUDA ? makeCUDABackend(b) : makeCPUEmulationBackend(b);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 == 0 ? (v[n / 2 - 1] + v[n / 2]) / 2 : v[n / 2];
}

int render(const Options &o) {
    const RenderPlan plan = makePlan(o);
    makeDirectories(plan.output);
    if (!o.mp4.empty() && fileExists(o.mp4)) {
        throw UsageError("--mp4 " + o.mp4 + " already exists (Forge never overwrites a video); remove it or choose another name.");
    }
    if (backendUniformBlockSize() != sizeof(ForgeUniforms)) {
        throw std::runtime_error("ForgeUniforms is " + std::to_string(sizeof(ForgeUniforms)) + " bytes on the host but " +
                                 std::to_string(backendUniformBlockSize()) + " in the kernels: layout mismatch.");
    }

    const FrameRequest probe = frameRequest(plan.project, plan.firstFrame());
    const std::vector<Tile> frameTiles = tiles(plan.project.width, plan.project.height, probe.quality.tileSize);
    const std::vector<SampleBatch> batches = sampleBatches(probe.quality);

    const double setupStart = now();
    CUDAEnvironment env;
    if (o.backend == Backend::CUDA) {
        env = cudaEnvironment();
        printCUDAEnvironmentHeader(env);
    }
    std::unique_ptr<RenderBackend> backend =
        makeBackend(o, plan, probe.quality.tileSize, frameTiles.size() * batches.size());
    const double setupSeconds = now() - setupStart;
    const std::vector<BackendFact> facts = backend->facts();

    std::printf("\n==================================================================\n");
    std::printf("Forge Render Worker %s — renderer: %s\n", kWorkerVersion,
                backend->backendName() == "cuda" ? "CUDA (headless NVIDIA compute)" : "CPU EMULATION (not a GPU)");
    for (const auto &f : facts) std::printf("  %-38s %s\n", f.key.c_str(), f.value.c_str());
    std::printf("World: %s (%s), shipping Metal shader source compiled for this backend\n",
                MandelNewtonWorld::displayName, MandelNewtonWorld::identifier);
    std::printf("Host: %s, %d threads; %s\n", cpuModel().c_str(), hardwareThreads(), osDescription().c_str());
    std::printf("Resolution: %dx%d\n", plan.project.width, plan.project.height);
    std::printf("Frames: %zu from %d at %g FPS; seed %u; project range starts at frame %d\n",
                plan.absoluteFrames.size(), plan.firstFrame(), plan.project.frameRate, plan.project.seed, plan.base);
    std::printf("Quality: %s (%d spp in %zu batches); loop closure: Off\n", qualityDisplayName(plan.project.quality),
                probe.quality.samplesPerPixel, batches.size());
    std::printf("Tiles: %d px (%zu tiles x %zu batches = %zu kernel launches per frame + 1 resolve)\n",
                probe.quality.tileSize, frameTiles.size(), batches.size(), frameTiles.size() * batches.size());
    std::printf("Backend setup: %.3f s\n", setupSeconds);
    std::printf("==================================================================\n");
    std::fflush(stdout);

    if (!o.dumpUniforms.empty()) makeDirectories(o.dumpUniforms);
    const size_t pixelValues = static_cast<size_t>(plan.project.width) * plan.project.height * 4;

    // Warm-up: first-launch costs (module load, PTX JIT, clocks ramping)
    // stay out of the measured frames.
    double warmupSeconds = 0;
    for (int w = 0; w < o.warmupFrames; ++w) {
        const double start = now();
        const FrameState state = makeFrameState(plan, plan.firstFrame());
        const FrameUniforms u = makeFrameUniforms(plan, state, frameTiles, batches);
        std::vector<uint16_t> pixels(pixelValues);
        BackendFrameTimings t;
        backend->renderFrame(blocks(u.dispatches.data(), u.dispatches.size()), blocks(&u.resolve, 1), pixels.data(), t);
        warmupSeconds += now() - start;
        std::printf("Warm-up %d/%d: %.3f s (GPU %.3f s)\n", w + 1, o.warmupFrames, now() - start, t.gpuWallSeconds);
        std::fflush(stdout);
    }

    const int pngThreads = !o.png ? 0 : (o.pngThreads >= 0 ? o.pngThreads : std::max(1, std::min(8, hardwareThreads() / 2)));
    std::unique_ptr<PNGExporter> exporter;
    if (o.png) {
        exporter.reset(new PNGExporter(plan.output, plan.project.filenamePrefix, plan.project.width, plan.project.height,
                                       pngThreads, static_cast<size_t>(std::max(2, pngThreads * 2))));
    }

    std::vector<StageSample> stages;
    const double wallStart = now();
    for (int frame : plan.absoluteFrames) {
        const double frameStart = now();
        StageSample stage;
        stage.frame = frame;
        const FrameState state = makeFrameState(plan, frame);
        stage.timelineCameraCPU = state.timelineSeconds;

        const double uniformStart = now();
        const FrameUniforms u = makeFrameUniforms(plan, state, frameTiles, batches);
        stage.uniformBuild = now() - uniformStart;

        if (!o.dumpUniforms.empty()) {
            const std::vector<uint32_t> words = canonicalUniformWords(u.resolve);
            std::vector<uint8_t> bytes;
            for (uint32_t w : words) {
                for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(w >> (i * 8)));
            }
            writeFileAtomically(o.dumpUniforms + "/frame_" + std::to_string(frame) + "_uniforms.bin", bytes);
        }

        std::vector<uint16_t> pixels(pixelValues);
        BackendFrameTimings t;
        backend->renderFrame(blocks(u.dispatches.data(), u.dispatches.size()), blocks(&u.resolve, 1), pixels.data(), t);
        stage.enqueueCPU = t.enqueueCPUSeconds;
        stage.gpuAccumulate = t.accumulateGPUSeconds;
        stage.gpuResolve = t.resolveGPUSeconds;
        stage.gpuToCPUTransfer = t.transferGPUSeconds;
        stage.hostCopy = t.hostCopySeconds;
        stage.gpuWait = t.waitSeconds;
        stage.gpuWall = t.gpuWallSeconds;

        if (exporter) stage.pngQueueWait = exporter->submit(frame, std::move(pixels));
        stage.renderLoop = now() - frameStart;
        stages.push_back(stage);
        std::printf("Frame %d: accumulate %.1f ms, resolve %.2f ms, GPU->CPU %.2f ms (+%.2f ms copy), "
                    "render loop %.1f ms%s\n",
                    frame, stage.gpuAccumulate * 1000, stage.gpuResolve * 1000, stage.gpuToCPUTransfer * 1000,
                    stage.hostCopy * 1000, stage.renderLoop * 1000,
                    !exporter ? " (no PNG)" : (pngThreads > 0 ? " (PNG queued)" : " (PNG written)"));
        std::fflush(stdout);
    }
    const double renderedWall = now() - wallStart;
    if (exporter) exporter->finish();
    const double totalWall = now() - wallStart;

    // PNG timings joined by frame.
    double pngEncodeTotal = 0, pngWriteTotal = 0;
    if (exporter) {
        for (const auto &p : exporter->timings()) {
            for (auto &s : stages) {
                if (s.frame == p.frame) {
                    s.pngEncode = p.encodeSeconds;
                    s.pngWrite = p.writeSeconds;
                }
            }
            pngEncodeTotal += p.encodeSeconds;
            pngWriteTotal += p.writeSeconds;
        }
    }
    for (auto &s : stages) {
        // Inline encoding is already inside the render loop's time.
        s.total = s.renderLoop + (pngThreads > 0 ? s.pngEncode + s.pngWrite : 0);
    }

    double mp4Seconds = 0;
    if (!o.mp4.empty()) {
        const double start = now();
        std::printf("Encoding %s with ffmpeg (Forge's H.264 settings)...\n", o.mp4.c_str());
        std::fflush(stdout);
        runFFmpeg(mp4Arguments(plan.output + "/" + plan.project.filenamePrefix + "_%06d.png", plan.firstFrame(),
                               static_cast<int>(plan.absoluteFrames.size()), plan.project.frameRate, o.mp4Quality,
                               o.mp4));
        mp4Seconds = now() - start;
    }
    const double finishedWall = now() - wallStart;

    // benchmark.json — the macOS worker's keys where they mean the same thing.
    const size_t n = stages.size();
    std::vector<double> totals, loops, gpu, accum;
    for (const auto &s : stages) {
        totals.push_back(s.total);
        loops.push_back(s.renderLoop);
        gpu.push_back(s.gpuAccumulate + s.gpuResolve + s.gpuToCPUTransfer);
        accum.push_back(s.gpuAccumulate);
    }
    auto sum = [](const std::vector<double> &v) {
        double s = 0;
        for (double x : v) s += x;
        return s;
    };
    const double averageFrame = n ? sum(totals) / n : 0;

    Json doc = Json::object();
    doc["world"] = MandelNewtonWorld::identifier;
    doc["world_display_name"] = MandelNewtonWorld::displayName;
    Json frameList = Json::array();
    for (int f : plan.absoluteFrames) frameList.push(f);
    doc["frame_list"] = frameList;
    doc["architecture"] = machineArchitecture();
    doc["build_architecture"] = buildArchitecture();
    doc["os"] = osDescription();
    doc["platform"] = "linux";
    doc["backend"] = backend->backendName();
    doc["gpu"] = backend->deviceName();
    doc["selected_gpu_name"] = backend->deviceName();
    doc["selected_gpu_index"] = backend->deviceIndex();
    Json factDoc = Json::object();
    for (const auto &f : facts) factDoc[f.key] = f.value;
    doc["renderer"] = factDoc;
    if (o.backend == Backend::CUDA) {
        Json all = Json::array();
        for (const auto &d : env.devices) all.push(cudaDeviceJson(d));
        doc["all_cuda_devices"] = all;
        doc["cuda_driver_supports"] = cudaVersionString(env.driverVersion);
        doc["cuda_runtime_built"] = cudaVersionString(env.runtimeVersion);
    }
    doc["cpu_model"] = cpuModel();
    doc["logical_cores"] = static_cast<long>(sysconf(_SC_NPROCESSORS_ONLN));
    doc["ram_bytes"] = static_cast<long long>(sysconf(_SC_PHYS_PAGES)) * static_cast<long long>(sysconf(_SC_PAGESIZE));
    doc["width"] = plan.project.width;
    doc["height"] = plan.project.height;
    doc["frames"] = static_cast<int>(n);
    doc["first_frame"] = plan.firstFrame();
    doc["fps"] = plan.project.frameRate;
    doc["seed"] = plan.project.seed;
    doc["quality"] = qualityDisplayName(plan.project.quality);
    doc["samples_per_pixel"] = probe.quality.samplesPerPixel;
    doc["sample_batches_per_frame"] = static_cast<int>(batches.size());
    doc["tiles_per_frame"] = static_cast<int>(frameTiles.size());
    doc["tile_size"] = probe.quality.tileSize;
    doc["loop_closure"] = "Off";
    doc["job_id"] = nullptr;
    doc["render_mode"] = "single-gpu";
    doc["worker_version"] = kWorkerVersion;
    doc["source_revision"] = FORGE_WORKER_SOURCE_REVISION;
    doc["png_output"] = o.png;
    doc["png_encoder_threads"] = pngThreads;
    doc["warmup_frames"] = o.warmupFrames;
    doc["warmup_seconds"] = warmupSeconds;
    doc["timing_note"] =
        "gpu_accumulate_kernels, gpu_resolve_kernel and gpu_to_cpu_transfer are CUDA event times on the GPU "
        "(host clocks for cpu-emulation). render_loop is what a frame cost the render loop (frame state, "
        "uniforms, launch, GPU wait, host copy, and PNG back-pressure); PNG encode/write run on encoder threads "
        "overlapping the next frame when png_encoder_threads > 0. total_frame_seconds = render_loop + png_encode "
        "+ png_write (serial equivalent). effective_render_fps uses wall time including the final PNG flush.";
    doc["setup_seconds_excluding_frames"] = setupSeconds;
    doc["total_seconds"] = sum(totals);
    doc["render_loop_wall_seconds"] = renderedWall;
    doc["total_wall_seconds"] = totalWall;
    doc["average_ms_per_frame"] = averageFrame * 1000;
    doc["median_ms_per_frame"] = median(totals) * 1000;
    doc["median_render_loop_ms_per_frame"] = median(loops) * 1000;
    doc["median_gpu_ms_per_frame"] = median(gpu) * 1000;
    doc["median_gpu_accumulate_ms_per_frame"] = median(accum) * 1000;
    doc["min_ms_per_frame"] = (n ? *std::min_element(totals.begin(), totals.end()) : 0) * 1000;
    doc["max_ms_per_frame"] = (n ? *std::max_element(totals.begin(), totals.end()) : 0) * 1000;
    doc["effective_render_fps"] = totalWall > 0 ? n / totalWall : 0.0;
    doc["effective_seconds_per_completed_frame"] = totalWall / std::max<size_t>(n, 1);
    doc["gpu_only_fps"] = sum(gpu) > 0 ? n / sum(gpu) : 0.0;
    doc["megapixel_samples_per_second"] =
        sum(accum) > 0 ? static_cast<double>(plan.project.width) * plan.project.height * probe.quality.samplesPerPixel * n /
                             sum(accum) / 1e6
                       : 0.0;
    doc["frames_rendered_this_run"] = static_cast<int>(n);
    doc["frames_resumed"] = 0;
    doc["png_encoding_seconds"] = pngEncodeTotal;
    doc["disk_write_seconds"] = pngWriteTotal;
    doc["output_format"] = o.png ? "PNG 16-bit RGB, sRGB" : "none (--no-png)";
    if (!o.mp4.empty()) {
        doc["mp4"] = o.mp4;
        doc["mp4_encode_seconds"] = mp4Seconds;
    }
    doc["total_wall_seconds_including_mp4"] = finishedWall;
    Json stageRows = Json::array();
    for (const auto &s : stages) stageRows.push(stageDocument(s));
    doc["frame_stage_timings"] = stageRows;
    Json averages = Json::object();
    for (const auto &f : stageFields()) {
        double total = 0;
        for (const auto &s : stages) total += s.*(f.second);
        averages[std::string(f.first) + "_seconds"] = n ? total / n : 0.0;
    }
    averages["total_frame_seconds"] = averageFrame;
    doc["average_stage_timings"] = averages;

    const std::string text = doc.dump() + "\n";
    const std::string benchmarkPath = plan.output + "/benchmark.json";
    writeFileAtomically(benchmarkPath, std::vector<uint8_t>(text.begin(), text.end()));

    auto averageOf = [&](double StageSample::*field) {
        double total = 0;
        for (const auto &s : stages) total += s.*field;
        return n ? total / n * 1000 : 0.0;
    };
    std::printf("\nSummary (%s on %s, %dx%d, %s):\n", backend->backendName().c_str(), backend->deviceName().c_str(),
                plan.project.width, plan.project.height, qualityDisplayName(plan.project.quality));
    std::printf("  average per frame: accumulate %.1f ms, resolve %.2f ms, GPU->CPU %.2f ms, host copy %.2f ms\n",
                averageOf(&StageSample::gpuAccumulate), averageOf(&StageSample::gpuResolve),
                averageOf(&StageSample::gpuToCPUTransfer), averageOf(&StageSample::hostCopy));
    std::printf("                     frame prep %.2f ms, render loop %.1f ms, PNG encode %.1f ms, PNG write %.1f ms\n",
                averageOf(&StageSample::timelineCameraCPU) + averageOf(&StageSample::uniformBuild),
                averageOf(&StageSample::renderLoop), averageOf(&StageSample::pngEncode), averageOf(&StageSample::pngWrite));
    std::printf("  total per frame (serial equivalent): %.1f ms average, %.1f ms median\n", averageFrame * 1000,
                median(totals) * 1000);
    std::printf("  wall: %.2f s for %zu frames (%.2f s rendering, %.2f s final PNG flush) = %.3f frames/s, %.3f s/frame\n",
                totalWall, n, renderedWall, totalWall - renderedWall, totalWall > 0 ? n / totalWall : 0.0,
                totalWall / std::max<size_t>(n, 1));
    if (!o.mp4.empty()) std::printf("  MP4: %s (%.2f s ffmpeg)\n", o.mp4.c_str(), mp4Seconds);
    std::printf("Output: %s   Benchmark: %s\n", o.png ? plan.output.c_str() : "(no PNG)", benchmarkPath.c_str());
    return 0;
}

int capabilities(const Options &o) {
    Json doc = Json::object();
    doc["workerVersion"] = kWorkerVersion;
    // No --job support yet: protocol 0 and no job schemas, so a Forge
    // controller never sends this worker a remote job.
    doc["protocolVersion"] = 0;
    doc["jobSchemaVersions"] = Json::array();
    doc["architecture"] = buildArchitecture();
    doc["os"] = osDescription();
    doc["platform"] = "linux";
    doc["backend"] = "cuda";
    const CUDAEnvironment env = cudaEnvironment();
    doc["cudaCompiledIn"] = env.compiledIn;
    doc["cudaRuntimeBuilt"] = cudaVersionString(env.runtimeVersion);
    doc["cudaDriverSupports"] = cudaVersionString(env.driverVersion);
    doc["cudaCompiledArchitectures"] = env.compiledArchitectures;
    Json devices = Json::array();
    int defaultIndex = -1;
    for (const auto &d : env.devices) {
        devices.push(cudaDeviceJson(d));
        if (d.supported && defaultIndex < 0) defaultIndex = d.index;
    }
    if (!env.error.empty()) doc["cudaError"] = env.error;
    doc["devices"] = devices;
    doc["defaultDeviceIndex"] = defaultIndex >= 0 ? Json(defaultIndex) : Json(nullptr);
    doc["dualGPUAvailable"] = false;
    Json worlds = Json::array();
    worlds.push(MandelNewtonWorld::identifier); // only worlds this worker genuinely renders
    doc["worlds"] = worlds;
    doc["sourceRevision"] = FORGE_WORKER_SOURCE_REVISION;
    std::printf("%s\n", doc.dump().c_str());
    (void)o;
    return 0;
}

} // namespace

// Exit codes as on macOS: 0 completed, 1 failed, 2 usage.
int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.version) {
            std::printf("ForgeRenderWorker %s (CUDA %s, protocol 0, %s, source %s)\n", kWorkerVersion,
                        cudaCompiledIn() ? cudaVersionString(cudaEnvironment().runtimeVersion).c_str() : "not compiled in",
                        buildArchitecture().c_str(), FORGE_WORKER_SOURCE_REVISION);
            return 0;
        }
        if (options.capabilities) return capabilities(options);
        if (!options.snapshotHashFrames.empty()) return printSnapshotHashes(options);
        if (options.listDevices) return listDevices(options);
        return render(options);
    } catch (const HelpRequested &) {
        std::printf("%s", kUsage);
        return 0;
    } catch (const UsageError &e) {
        std::fprintf(stderr, "ForgeRenderWorker: %s\n\n%s", e.what(), kUsage);
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ForgeRenderWorker: %s\n", e.what());
        return 1;
    }
}
