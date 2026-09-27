//
//  ForgeShaderPort.cuh
//  Cyanescent Forge — Linux CUDA worker
//
//  Compiles the SHIPPING Metal shader modules — CyanescentForge/Shaders/*.h,
//  unmodified — as CUDA C++ (and as host C++ for the CPU emulation backend).
//
//  The modules are written in the C++-compatible subset of MSL that
//  Tools/CPUReference relies on. The MSL spellings that remain are mapped
//  here, only while the modules are being included:
//
//    inline            -> __host__ __device__ inline (every shader function)
//    kernel            -> static inline (the MSL entry points become ordinary
//                         functions; the CUDA __global__ kernels call them)
//    constant T &      -> const T & (read-only; the uniforms themselves live
//                         in the kernel's __grid_constant__ parameter)
//    thread / device   -> nothing (address spaces are implicit in CUDA)
//    [[buffer(n)]], [[thread_position_in_grid]] -> [[]]
//
//  So the CUDA backend renders with the same functions, constants and
//  evaluation order as Metal — there is no hand translation to drift (the
//  Vulkan port needed one because GLSL cannot pass structs by reference).
//
//  Everything is inside namespace forge_gpu, which keeps the MSL-named types
//  (float3, uint2, ForgeUniforms, ...) away from CUDA's and the host's.
//
#ifndef FORGE_SHADER_PORT_CUH
#define FORGE_SHADER_PORT_CUH

#include "ForgeCUDACompat.cuh"

#define FORGE_CPU_REFERENCE 1 // ForgeShaderTypes.h: host-supplied float2/3/4

// `device` must become nothing, but CUDA's own __device__ macro expands to
// __location__(device), which would then lose its attribute. So while the
// modules are included, CUDA's __host__/__device__ macros are set aside and
// the execution-space attributes are spelled directly.
#pragma push_macro("inline")
#pragma push_macro("__host__")
#pragma push_macro("__device__")
#undef __host__
#undef __device__
#define constant const
#define thread
#define device
#define kernel static inline
#define buffer(index)
#define thread_position_in_grid
#if defined(__CUDACC__)
#define inline __attribute__((__host__)) __attribute__((__device__)) inline
#endif
#if defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace forge_gpu {
#include "../../../../CyanescentForge/Shaders/ForgeShaderTypes.h"
#include "../../../../CyanescentForge/Shaders/ForgeMath.h"
#include "../../../../CyanescentForge/Shaders/ForgePost.h"
// Worlds. Each world's module brings in the shared camera, lighting,
// material and volumetric modules it uses.
#include "../../../../CyanescentForge/Shaders/ForgeMandelNewton.h"
} // namespace forge_gpu

#if defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC diagnostic pop
#endif
#undef constant
#undef thread
#undef device
#undef kernel
#undef buffer
#undef thread_position_in_grid
#pragma pop_macro("__device__")
#pragma pop_macro("__host__")
#pragma pop_macro("inline")

namespace forge_gpu {

// ---------------------------------------------------------------------------
// forgeResolveKernel (CyanescentForge/Shaders/ApollonianWorld.metal) for one
// pixel. That kernel lives in the .metal file, not in a shared header, so its
// body is repeated here expression for expression. It averages the linear
// HDR accumulation, applies Forge's display transform with the same
// deterministic dither, and stores what Metal's rgba16Unorm texture write
// stores: clamp to [0, 1], scale by 65535, round to nearest even.
// ---------------------------------------------------------------------------
struct alignas(8) RGBA16 {
    unsigned short r, g, b, a;
};

FORGE_HD unsigned short forgeUnorm16(float x)
{
    return (unsigned short)::rintf(clamp(x, 0.0f, 1.0f) * 65535.0f);
}

FORGE_HD RGBA16 forgeResolvePixel(const float4 *accumulation, const ForgeUniforms &uniforms,
                                  unsigned int px, unsigned int py)
{
    float4 accumulated = accumulation[py * uint(uniforms.imageSize.x) + px];
    float3 hdr = accumulated.w > 0.0f ? float3(accumulated.xyz) / accumulated.w : float3(0.0f);

    float2 uv = (float2(float(px), float(py)) + 0.5f) / uniforms.imageSize;

    uint ditherFrame = uniforms.scene.mcpmWorldExtent > 0.0f ? 0u : uniforms.frameIndex;
    uint ditherSeed = forgeHashCombine(forgeHashCombine(uniforms.seed, ditherFrame),
                                       px * 2654435761u ^ py * 40503u);
    float dither = forgeRandom(ditherSeed);

    float3 color = uniforms.scene.mcpmDebugMode > 0.5f
        ? saturate(hdr) : forgePostProcess(hdr, uv, uniforms.scene.exposure, dither);

    RGBA16 out;
    out.r = forgeUnorm16(color.x);
    out.g = forgeUnorm16(color.y);
    out.b = forgeUnorm16(color.z);
    out.a = forgeUnorm16(1.0f);
    return out;
}

// ---------------------------------------------------------------------------
// Worlds.
//
// A world is a type with an `accumulate` function: one invocation of that
// world's MSL accumulation kernel for thread `tid` of a tile dispatch (it adds
// mean radiance x samples, and the sample weight in .w, to the float4
// accumulation buffer). The CUDA __global__ kernel and the CPU emulation are
// templates over this type, so adding a world is: include its shipping module
// above, add a struct here, add an enum value and a case in
// forgeWorldKernelName / the two backends' dispatch switches.
// ---------------------------------------------------------------------------
enum class ForgeWorldKernel : int {
    MandelNewton = 1,
};

struct MandelNewtonWorldKernel {
    static FORGE_HD void accumulate(float4 *accumulation, const ForgeUniforms &u, uint2 tid)
    {
        // The shipping MSL entry point, unmodified.
        mandelnewtonRenderKernel(accumulation, u, tid);
    }
};

} // namespace forge_gpu

#endif // FORGE_SHADER_PORT_CUH
