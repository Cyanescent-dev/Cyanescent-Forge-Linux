//
//  MandelNewtonCPUReference.cpp
//  Cyanescent Forge — fidelity oracle for the Linux worker (validation only)
//
//  Compiles the SHIPPING Metal shader source — CyanescentForge/Shaders/
//  ForgeMandelNewton.h and ForgePost.h, unmodified — as C++ against the CPU
//  implementation of the MSL builtins in Tools/CPUReference/ForgeCPUCompat.h,
//  and renders one frame from the exact uniforms the Linux worker hands its
//  GPU backend (written with `ForgeRenderWorker --dump-uniforms DIR`).
//
//  Comparing its PNG with the worker's (tools/png_inspect.py compare) checks
//  the CUDA render of the Metal world without a Mac. (It was written for the
//  retired Vulkan port's hand-translated GLSL; the CUDA backend compiles the
//  shipping MSL itself, so here it checks the CUDA compat layer, the
//  resolve port and the GPU's arithmetic.) It is not a renderer for production use: it is
//  single-precision scalar C++ and slow.
//
//  Build/run: Tools/ForgeRenderWorker/Linux/fidelity/run-fidelity-check.sh
//
// Standard headers first: ForgeCPUCompat.h #defines `constant`, which must
// not reach library headers.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../src/PNGWriter.hpp"

#include "../../../CPUReference/ForgeCPUCompat.h"

// The pieces of MSL the kernel entry point in ForgeMandelNewton.h spells that
// ForgeCPUCompat.h leaves out (it compiles the pure-maths modules only).
using std::abs;
using std::isinf;
using std::isnan;
using std::log2;
struct uint2 {
    unsigned int x, y;
    uint2(unsigned int a, unsigned int b) : x(a), y(b) {}
};
#define thread
#define device
#define kernel static inline
#pragma GCC diagnostic ignored "-Wattributes"

#include "../../../../CyanescentForge/Shaders/ForgeShaderTypes.h"
#include "../../../../CyanescentForge/Shaders/ForgeMath.h"
#include "../../../../CyanescentForge/Shaders/ForgeMandelNewton.h"
#include "../../../../CyanescentForge/Shaders/ForgePost.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s frame_N_uniforms.bin out.png\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // The dump holds ForgeUniforms as canonical little-endian words with the
    // 3-vector padding lanes removed — exactly this harness's packed layout.
    if (bytes.size() != sizeof(ForgeUniforms)) {
        std::fprintf(stderr, "uniform dump is %zu bytes; the CPU ForgeUniforms is %zu\n", bytes.size(),
                     sizeof(ForgeUniforms));
        return 1;
    }
    ForgeUniforms u;
    for (size_t i = 0; i < bytes.size(); i += 4) {
        const uint32_t word = uint32_t(bytes[i]) | uint32_t(bytes[i + 1]) << 8 | uint32_t(bytes[i + 2]) << 16 |
                              uint32_t(bytes[i + 3]) << 24;
        std::memcpy(reinterpret_cast<unsigned char *>(&u) + i, &word, 4);
    }
    const int width = int(u.imageSize.x), height = int(u.imageSize.y);
    std::fprintf(stderr, "CPU reference: %dx%d, frame %u, seed %u, %d samples, mnBalance %.6f, camera (%.4f, %.4f, %.4f)\n",
                 width, height, u.frameIndex, u.seed, u.quality.samplesPerPixel, u.scene.mnBalance,
                 u.cameraPosition.x, u.cameraPosition.y, u.cameraPosition.z);

    // mandelnewtonRenderKernel over the whole image, all samples in one pass.
    std::vector<float4> accumulation(size_t(width) * height, float4(0.0f));
    u.tileOrigin = float2(0.0f, 0.0f);
    u.tileSize = float2(float(width), float(height));
    u.accumulate = 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) mandelnewtonRenderKernel(accumulation.data(), u, uint2(x, y));
    }

    // forgeResolveKernel (ApollonianWorld.metal), then the rgba16Unorm store.
    std::vector<uint16_t> rgba16(size_t(width) * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float4 accumulated = accumulation[size_t(y) * width + x];
            const float3 sum = accumulated.xyz;
            const float3 hdr = accumulated.w > 0.0f ? sum / accumulated.w : float3(0.0f);
            const float2 uv = float2((float(x) + 0.5f) / u.imageSize.x, (float(y) + 0.5f) / u.imageSize.y);
            const unsigned int ditherFrame = u.scene.mcpmWorldExtent > 0.0f ? 0u : u.frameIndex;
            const unsigned int ditherSeed = forgeHashCombine(forgeHashCombine(u.seed, ditherFrame),
                                                             unsigned(x) * 2654435761u ^ unsigned(y) * 40503u);
            const float dither = forgeRandom(ditherSeed);
            const float3 color = u.scene.mcpmDebugMode > 0.5f ? saturate(hdr)
                                                              : forgePostProcess(hdr, uv, u.scene.exposure, dither);
            const float c[4] = {color.x, color.y, color.z, 1.0f};
            for (int k = 0; k < 4; ++k) {
                rgba16[(size_t(y) * width + x) * 4 + k] =
                    uint16_t(std::nearbyint(std::fmin(std::fmax(c[k], 0.0f), 1.0f) * 65535.0f));
            }
        }
    }
    forge::writeFileAtomically(argv[2], forge::encodePNG16(rgba16, width, height));
    std::fprintf(stderr, "wrote %s\n", argv[2]);
    return 0;
}
