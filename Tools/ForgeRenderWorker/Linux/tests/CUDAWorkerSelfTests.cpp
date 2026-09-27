//
//  CUDAWorkerSelfTests.cpp
//  Cyanescent Forge — Linux CUDA worker
//
//  Host unit tests, run by build-linux-cuda.sh on every build (no GPU needed):
//
//    * the MSL compat layer the kernels are compiled against (layout,
//      swizzles, builtins the shipping modules rely on)
//    * the resolve's Metal rgba16Unorm store (clamp, x65535, round half to even)
//    * which devices a build's architecture list can run on
//    * the MP4 arguments are Forge's (CyanescentForge/Core/MP4Export.swift)
//    * the asynchronous PNG exporter
//
//  Rendering correctness is checked end to end by test-cuda-renderer.sh
//  (reference frames and the CPU oracle).
//
#include "../cuda/CUDAArchitectures.hpp"
#include "../cuda/ForgeShaderPort.cuh"
#include "../src/FrameExport.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

int checks = 0, failures = 0;

void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

bool near(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) <= eps; }

void compatLayer() {
    using namespace forge_gpu;
    check(sizeof(float3) == 16 && alignof(float3) == 16, "float3 is 16 bytes, 16-aligned (MSL)");
    check(sizeof(float2) == 8 && sizeof(float4) == 16, "float2/float4 sizes (MSL)");
    const float3 v(1.0f, 2.0f, 3.0f);
    const float2 xz = v.xz;
    check(xz.x == 1.0f && xz.y == 3.0f, "float3.xz swizzle");
    const float4 w(1.0f, 2.0f, 3.0f, 4.0f);
    const float3 xyz = w.xyz;
    check(xyz.x == 1.0f && xyz.z == 3.0f, "float4.xyz swizzle");
    check(near(length(normalize(float3(3.0f, 4.0f, 0.0f))), 1.0f), "normalize length 1");
    const float3 zero = normalize(float3(0.0f));
    check(zero.x == 0.0f && zero.y == 0.0f && zero.z == 0.0f, "normalize(0) stays 0 (ForgeCPUCompat semantics)");
    check(near(mix(2.0f, 4.0f, 0.25f), 2.5f), "mix");
    check(near(smoothstep(0.0f, 1.0f, 0.5f), 0.5f) && smoothstep(0.0f, 1.0f, -1.0f) == 0.0f, "smoothstep");
    check(clamp(2.0f, 0.0f, 1.0f) == 1.0f && saturate(-1.0f) == 0.0f, "clamp/saturate");
    check(near(fract(-0.25f), 0.75f), "fract of a negative");
    check(min(3, 5) == 3 && max(3u, 5u) == 5u && min(1.0f, -1.0f) == -1.0f, "min/max overloads");
    const float3 c = cross(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
    check(c.z == 1.0f && c.x == 0.0f, "cross");
    check(forge_gpu::isnan(std::nanf("")) && forge_gpu::isinf(INFINITY) && !forge_gpu::isnan(1.0f), "isnan/isinf");
    check(forge_gpu::abs(-2) == 2 && forge_gpu::abs(-2.5f) == 2.5f, "abs int/float");
    // Shipping ForgeMath.h through the port: deterministic hash and Halton.
    check(forgeHashCombine(1u, 2u) == forgeHashCombine(1u, 2u), "forgeHashCombine deterministic");
    const float r = forgeRandom(12345u);
    check(r >= 0.0f && r < 1.0f, "forgeRandom in [0,1)");
}

void unorm16() {
    using forge_gpu::forgeUnorm16;
    check(forgeUnorm16(0.0f) == 0 && forgeUnorm16(1.0f) == 65535, "unorm16 endpoints");
    check(forgeUnorm16(-1.0f) == 0 && forgeUnorm16(2.0f) == 65535, "unorm16 clamps");
    check(forgeUnorm16(0.5f) == 32768, "unorm16 32767.5 rounds half to even -> 32768");
    check(forgeUnorm16(0.5f / 65535.0f) == 0, "unorm16 0.5 rounds half to even -> 0");
    check(forgeUnorm16(1.5f / 65535.0f) == 2, "unorm16 1.5 rounds half to even -> 2");
    check(forgeUnorm16(std::nanf("")) == 0, "unorm16 NaN -> 0");

    // Resolve of an empty accumulator is black (hdr 0), with alpha 1.
    forge_gpu::ForgeUniforms u{};
    u.imageSize = forge_gpu::float2(1.0f, 1.0f);
    u.scene.exposure = 1.0f;
    const forge_gpu::float4 empty(0.0f);
    const forge_gpu::RGBA16 p = forge_gpu::forgeResolvePixel(&empty, u, 0, 0);
    check(p.a == 65535, "resolve alpha is 1");
    check(p.r <= 129 && p.g <= 129 && p.b <= 129, "resolve of zero radiance is black plus at most half an 8-bit dither step");
}

void architectures() {
    using forge::cudaArchitecturesCover;
    const std::string portable = "sm_75 sm_80 sm_86 sm_89 sm_90 compute_90";
    check(cudaArchitecturesCover(portable, 8, 9), "L4/4090 (8.9) covered by sm_89");
    check(cudaArchitecturesCover(portable, 8, 6), "A40/A5000/3090 (8.6) covered");
    check(cudaArchitecturesCover(portable, 7, 5), "T4 (7.5) covered");
    check(cudaArchitecturesCover(portable, 8, 7), "Orin (8.7) runs sm_86 SASS");
    check(cudaArchitecturesCover(portable, 12, 0), "Blackwell (12.0) JITs compute_90 PTX");
    check(!cudaArchitecturesCover(portable, 7, 0), "V100 (7.0) not covered by this list");
    check(!cudaArchitecturesCover("sm_89 compute_89", 8, 6), "sm_89 SASS does not run on 8.6");
    check(!cudaArchitecturesCover("sm_89 compute_89", 8, 0), "compute_89 PTX does not JIT for 8.0");
    check(cudaArchitecturesCover("sm_89 compute_89", 9, 0), "compute_89 PTX JITs for 9.0");
    check(!cudaArchitecturesCover("none", 8, 9), "no code list covers nothing");
}

void mp4() {
    const std::vector<std::string> args =
        forge::mp4Arguments("out/frame_%06d.png", 12, 300, 30.0, forge::MP4Quality::High, "out.mp4");
    const std::vector<std::string> expected = {
        "-hide_banner", "-nostats", "-v", "error", "-n", "-framerate", "30", "-start_number", "12",
        "-i", "out/frame_%06d.png", "-frames:v", "300", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "out.mp4"};
    check(args == expected, "MP4 arguments match ForgeMP4Encoding (High)");
    const auto veryHigh = forge::mp4Arguments("p", 0, 1, 29.97, forge::MP4Quality::VeryHigh, "o");
    check(veryHigh[6] == "29.97" && veryHigh[16] == "slower" && veryHigh[18] == "15", "MP4 Very High = slower/15");
    const auto standard = forge::mp4Arguments("p", 0, 1, 24, forge::MP4Quality::Standard, "o");
    check(standard[16] == "medium" && standard[18] == "20", "MP4 Standard = medium/20");
}

void pngExporter() {
    char dir[] = "/tmp/forge-selftest-XXXXXX";
    if (!mkdtemp(dir)) {
        check(false, "mkdtemp");
        return;
    }
    const int w = 16, h = 8;
    {
        forge::PNGExporter exporter(dir, "frame", w, h, 2, 3);
        for (int f = 0; f < 6; ++f) {
            std::vector<uint16_t> px(static_cast<size_t>(w) * h * 4, static_cast<uint16_t>(f * 1000));
            exporter.submit(100 + f, std::move(px));
        }
        exporter.finish();
        const auto timings = exporter.timings();
        check(timings.size() == 6, "exporter timed every frame");
        bool ordered = true, files = true;
        for (size_t i = 0; i < timings.size(); ++i) {
            ordered = ordered && timings[i].frame == 100 + static_cast<int>(i);
            struct stat st {};
            files = files && stat(exporter.path(100 + static_cast<int>(i)).c_str(), &st) == 0 && st.st_size > 0;
        }
        check(ordered, "exporter timings in submission order");
        check(files, "exporter wrote every PNG");
        check(exporter.path(7) == std::string(dir) + "/frame_000007.png", "frame file naming");
        for (int f = 0; f < 6; ++f) unlink(exporter.path(100 + f).c_str());
    }
    {
        forge::PNGExporter inlineExporter(dir, "frame", w, h, 0, 1);
        inlineExporter.submit(1, std::vector<uint16_t>(static_cast<size_t>(w) * h * 4, 0));
        inlineExporter.finish();
        check(inlineExporter.timings().size() == 1, "inline exporter");
        unlink(inlineExporter.path(1).c_str());
    }
    rmdir(dir);
}

} // namespace

int main() {
    compatLayer();
    unorm16();
    architectures();
    mp4();
    pngExporter();
    std::printf("CUDAWorkerSelfTests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
