//
//  ForgeHostTypes.hpp
//  Cyanescent Forge — Linux worker
//
//  Brings in the one shared CPU/GPU struct definition,
//  CyanescentForge/Shaders/ForgeShaderTypes.h, unmodified. That header has a
//  branch for non-Apple C++ compilers (FORGE_CPU_REFERENCE) which expects the
//  host to supply float2/float3/float4. The ones here have exactly the size
//  and alignment of simd_float2/3/4 and MSL float2/3/4 (a 3-vector is 16
//  bytes, 16-byte aligned), so ForgeUniforms has the byte layout Metal
//  receives on macOS — and that the Vulkan shaders declare (checked at build
//  time by tools/check_layout.py).
//
//  Nothing here depends on the host's architecture beyond IEEE-754 floats and
//  the standard C++ alignment rules, which x86_64 and aarch64 share.
//
#ifndef FORGE_HOST_TYPES_HPP
#define FORGE_HOST_TYPES_HPP

#include <cstddef>
#include <cstdint>

struct alignas(8) float2 {
    float x, y;
};

struct alignas(16) float3 {
    float x, y, z;
};

struct alignas(16) float4 {
    float x, y, z, w;
};

static_assert(sizeof(float2) == 8, "float2 must match simd_float2");
static_assert(sizeof(float3) == 16 && alignof(float3) == 16, "float3 must match simd_float3 / MSL float3");
static_assert(sizeof(float4) == 16, "float4 must match simd_float4");

#ifndef FORGE_CPU_REFERENCE
#define FORGE_CPU_REFERENCE 1
#endif
#include "../../../../CyanescentForge/Shaders/ForgeShaderTypes.h"

inline float3 forgeFloat3(float x, float y, float z) { return float3{x, y, z}; }

#endif // FORGE_HOST_TYPES_HPP
