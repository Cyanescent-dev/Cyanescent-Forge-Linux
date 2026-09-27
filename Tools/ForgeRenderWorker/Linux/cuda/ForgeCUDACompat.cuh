//
//  ForgeCUDACompat.cuh
//  Cyanescent Forge — Linux CUDA worker
//
//  The Metal Shading Language vector types and builtins that the shared
//  shader modules (CyanescentForge/Shaders/*.h) use, as CUDA C++ that
//  compiles for the device (nvcc) and for the host (nvcc's host pass, or
//  g++ for the CPU emulation backend).
//
//  It is the CUDA sibling of Tools/CPUReference/ForgeCPUCompat.h, which
//  proved the modules are written in a C++-compatible subset of MSL. Two
//  differences matter:
//
//    * Layout. float3 is 16 bytes and 16-byte aligned, float2 8/8, float4
//      16/16 — exactly MSL, simd_float3 and src/ForgeHostTypes.hpp — so
//      ForgeUniforms has the byte layout Metal receives on macOS, and the
//      host's struct is copied to the GPU as is. (ForgeCPUCompat.h packs
//      float3 in 12 bytes.) The build checks every field's offset against
//      the host's (tools/check_cuda_layout.py).
//
//    * Precision. Single-precision float functions only (sinf, expf, powf,
//      ...): no double promotion anywhere, which on consumer NVIDIA GPUs
//      would be ~64x slower and would not match Metal. Metal compiles Forge
//      with fastMathEnabled = false (MetalContext.swift), so the CUDA build
//      does not use --use_fast_math either: IEEE division and square root,
//      no flush-to-zero, and CUDA's accurate libdevice transcendentals.
//
//  Everything lives in namespace forge_gpu so these types never meet CUDA's
//  own ::float2/::float3/::float4 (vector_types.h), which have no
//  constructors and a 12-byte float3.
//
#ifndef FORGE_CUDA_COMPAT_CUH
#define FORGE_CUDA_COMPAT_CUH

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define FORGE_HD __host__ __device__ __forceinline__
#else
#define FORGE_HD inline
#endif

namespace forge_gpu {

struct vec2; struct vec3; struct vec4;

// ---------------------------------------------------------------------------
// Swizzle proxies (read-only), as in ForgeCPUCompat.h. MSL exposes swizzles
// as data members (p.xz); the proxies overlay the component array and
// convert to a vector. Only the swizzles the shared modules use exist, so a
// module that drifts outside the portable subset fails to compile here.
// ---------------------------------------------------------------------------
template <int N, int A, int B> struct Sw2 { float d[N]; FORGE_HD operator vec2() const; };
template <int N, int A, int B, int C> struct Sw3 { float d[N]; FORGE_HD operator vec3() const; };

struct alignas(8) vec2 {
    union {
        float d[2];
        struct { float x, y; };
    };
    FORGE_HD vec2() : x(0.0f), y(0.0f) {}
    FORGE_HD vec2(float s) : x(s), y(s) {}
    FORGE_HD vec2(float a, float b) : x(a), y(b) {}
    FORGE_HD float  operator[](int i) const { return d[i]; }
    FORGE_HD float& operator[](int i)       { return d[i]; }
};

// 16 bytes, 16-byte aligned: the fourth lane is padding, as in MSL float3.
struct alignas(16) vec3 {
    union {
        float d[3];
        struct { float x, y, z; };
        Sw2<3,0,1> xy;
        Sw2<3,0,2> xz;
        Sw2<3,1,2> yz;
        Sw2<3,2,0> zx;
        Sw3<3,1,2,0> yzx;
        Sw3<3,2,0,1> zxy;
        Sw3<3,0,2,1> xzy;
    };
    FORGE_HD vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    FORGE_HD vec3(float s) : x(s), y(s), z(s) {}
    FORGE_HD vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    FORGE_HD vec3(vec2 a, float c) : x(a.x), y(a.y), z(c) {}
    FORGE_HD vec3(float a, vec2 b) : x(a), y(b.x), z(b.y) {}
    FORGE_HD float  operator[](int i) const { return d[i]; }
    FORGE_HD float& operator[](int i)       { return d[i]; }
};

struct alignas(16) vec4 {
    union {
        float d[4];
        struct { float x, y, z, w; };
        Sw2<4,0,1> xy;
        Sw2<4,2,3> zw;
        Sw3<4,0,1,2> xyz;
    };
    FORGE_HD vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    FORGE_HD vec4(float s) : x(s), y(s), z(s), w(s) {}
    FORGE_HD vec4(float a, float b, float c, float e) : x(a), y(b), z(c), w(e) {}
    FORGE_HD vec4(vec3 a, float e) : x(a.x), y(a.y), z(a.z), w(e) {}
    FORGE_HD float  operator[](int i) const { return d[i]; }
    FORGE_HD float& operator[](int i)       { return d[i]; }
};

static_assert(sizeof(vec2) == 8 && alignof(vec2) == 8, "float2 must match MSL / simd_float2");
static_assert(sizeof(vec3) == 16 && alignof(vec3) == 16, "float3 must match MSL / simd_float3");
static_assert(sizeof(vec4) == 16 && alignof(vec4) == 16, "float4 must match MSL / simd_float4");

template <int N, int A, int B> FORGE_HD Sw2<N,A,B>::operator vec2() const { return vec2(d[A], d[B]); }
template <int N, int A, int B, int C> FORGE_HD Sw3<N,A,B,C>::operator vec3() const { return vec3(d[A], d[B], d[C]); }

// MSL uint2 as the kernel entry points spell it (thread_position_in_grid).
struct uint2 {
    unsigned int x, y;
    FORGE_HD uint2(unsigned int a, unsigned int b) : x(a), y(b) {}
};
typedef unsigned int uint;

// --- arithmetic -------------------------------------------------------------
#define FORGE_VEC2_OP(OP) \
    FORGE_HD vec2 operator OP (vec2 a, vec2 b){ return vec2(a.x OP b.x, a.y OP b.y); } \
    FORGE_HD vec2 operator OP (vec2 a, float b){ return vec2(a.x OP b, a.y OP b); } \
    FORGE_HD vec2 operator OP (float a, vec2 b){ return vec2(a OP b.x, a OP b.y); }
#define FORGE_VEC3_OP(OP) \
    FORGE_HD vec3 operator OP (vec3 a, vec3 b){ return vec3(a.x OP b.x, a.y OP b.y, a.z OP b.z); } \
    FORGE_HD vec3 operator OP (vec3 a, float b){ return vec3(a.x OP b, a.y OP b, a.z OP b); } \
    FORGE_HD vec3 operator OP (float a, vec3 b){ return vec3(a OP b.x, a OP b.y, a OP b.z); }
#define FORGE_VEC4_OP(OP) \
    FORGE_HD vec4 operator OP (vec4 a, vec4 b){ return vec4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w); } \
    FORGE_HD vec4 operator OP (vec4 a, float b){ return vec4(a.x OP b, a.y OP b, a.z OP b, a.w OP b); } \
    FORGE_HD vec4 operator OP (float a, vec4 b){ return vec4(a OP b.x, a OP b.y, a OP b.z, a OP b.w); }

FORGE_VEC2_OP(+) FORGE_VEC2_OP(-) FORGE_VEC2_OP(*) FORGE_VEC2_OP(/)
FORGE_VEC3_OP(+) FORGE_VEC3_OP(-) FORGE_VEC3_OP(*) FORGE_VEC3_OP(/)
FORGE_VEC4_OP(+) FORGE_VEC4_OP(-) FORGE_VEC4_OP(*) FORGE_VEC4_OP(/)
#undef FORGE_VEC2_OP
#undef FORGE_VEC3_OP
#undef FORGE_VEC4_OP

FORGE_HD vec2 operator-(vec2 a){ return vec2(-a.x, -a.y); }
FORGE_HD vec3 operator-(vec3 a){ return vec3(-a.x, -a.y, -a.z); }
FORGE_HD vec4 operator-(vec4 a){ return vec4(-a.x, -a.y, -a.z, -a.w); }

#define FORGE_ASSIGN(T, OP) \
    FORGE_HD T& operator OP##= (T& a, T b){ a = a OP b; return a; } \
    FORGE_HD T& operator OP##= (T& a, float b){ a = a OP b; return a; }
FORGE_ASSIGN(vec2,+) FORGE_ASSIGN(vec2,-) FORGE_ASSIGN(vec2,*) FORGE_ASSIGN(vec2,/)
FORGE_ASSIGN(vec3,+) FORGE_ASSIGN(vec3,-) FORGE_ASSIGN(vec3,*) FORGE_ASSIGN(vec3,/)
FORGE_ASSIGN(vec4,+) FORGE_ASSIGN(vec4,-) FORGE_ASSIGN(vec4,*) FORGE_ASSIGN(vec4,/)
#undef FORGE_ASSIGN

// --- scalar builtins (float only; the names hide the global double versions) --
FORGE_HD float fabs(float a) { return ::fabsf(a); }
FORGE_HD float abs(float a) { return ::fabsf(a); }
FORGE_HD int abs(int a) { return a < 0 ? -a : a; }
FORGE_HD float floor(float a) { return ::floorf(a); }
FORGE_HD float sqrt(float a) { return ::sqrtf(a); }
FORGE_HD float pow(float a, float b) { return ::powf(a, b); }
FORGE_HD float exp(float a) { return ::expf(a); }
FORGE_HD float exp2(float a) { return ::exp2f(a); }
FORGE_HD float log(float a) { return ::logf(a); }
FORGE_HD float log2(float a) { return ::log2f(a); }
FORGE_HD float sin(float a) { return ::sinf(a); }
FORGE_HD float cos(float a) { return ::cosf(a); }
FORGE_HD float tan(float a) { return ::tanf(a); }
FORGE_HD float atan2(float y, float x) { return ::atan2f(y, x); }
FORGE_HD float precise_divide(float a, float b) { return a / b; }
#if defined(__CUDA_ARCH__)
FORGE_HD bool isnan(float a) { return ::isnan(a); }
FORGE_HD bool isinf(float a) { return ::isinf(a); }
FORGE_HD float rsqrt(float a) { return ::rsqrtf(a); }
#else
FORGE_HD bool isnan(float a) { return __builtin_isnan(a); }
FORGE_HD bool isinf(float a) { return __builtin_isinf(a); }
FORGE_HD float rsqrt(float a) { return 1.0f / ::sqrtf(a); }
#endif

FORGE_HD float min(float a, float b){ return ::fminf(a, b); }
FORGE_HD float max(float a, float b){ return ::fmaxf(a, b); }
FORGE_HD int min(int a, int b){ return a < b ? a : b; }
FORGE_HD int max(int a, int b){ return a > b ? a : b; }
FORGE_HD unsigned int min(unsigned int a, unsigned int b){ return a < b ? a : b; }
FORGE_HD unsigned int max(unsigned int a, unsigned int b){ return a > b ? a : b; }

FORGE_HD float clamp(float v, float lo, float hi){ return min(max(v, lo), hi); }
FORGE_HD float saturate(float v){ return clamp(v, 0.0f, 1.0f); }
FORGE_HD float fract(float a){ return a - ::floorf(a); }
FORGE_HD float mix(float a, float b, float t){ return a + (b - a) * t; }
FORGE_HD float step(float edge, float x){ return x < edge ? 0.0f : 1.0f; }
FORGE_HD float smoothstep(float e0, float e1, float x){
    float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}
FORGE_HD float sign(float a){ return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f); }

// --- vector builtins ----------------------------------------------------------
FORGE_HD vec2 abs(vec2 a){ return vec2(fabs(a.x), fabs(a.y)); }
FORGE_HD vec3 abs(vec3 a){ return vec3(fabs(a.x), fabs(a.y), fabs(a.z)); }
FORGE_HD vec4 abs(vec4 a){ return vec4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w)); }

FORGE_HD vec2 min(vec2 a, vec2 b){ return vec2(min(a.x,b.x), min(a.y,b.y)); }
FORGE_HD vec3 min(vec3 a, vec3 b){ return vec3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
FORGE_HD vec4 min(vec4 a, vec4 b){ return vec4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)); }
FORGE_HD vec2 max(vec2 a, vec2 b){ return vec2(max(a.x,b.x), max(a.y,b.y)); }
FORGE_HD vec3 max(vec3 a, vec3 b){ return vec3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
FORGE_HD vec4 max(vec4 a, vec4 b){ return vec4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)); }
FORGE_HD vec3 min(vec3 a, float b){ return min(a, vec3(b)); }
FORGE_HD vec3 max(vec3 a, float b){ return max(a, vec3(b)); }

FORGE_HD vec2 clamp(vec2 v, float lo, float hi){ return min(max(v, vec2(lo)), vec2(hi)); }
FORGE_HD vec3 clamp(vec3 v, float lo, float hi){ return min(max(v, vec3(lo)), vec3(hi)); }
FORGE_HD vec4 clamp(vec4 v, float lo, float hi){ return min(max(v, vec4(lo)), vec4(hi)); }
FORGE_HD vec3 clamp(vec3 v, vec3 lo, vec3 hi){ return min(max(v, lo), hi); }

FORGE_HD vec2 saturate(vec2 v){ return clamp(v, 0.0f, 1.0f); }
FORGE_HD vec3 saturate(vec3 v){ return clamp(v, 0.0f, 1.0f); }
FORGE_HD vec4 saturate(vec4 v){ return clamp(v, 0.0f, 1.0f); }

FORGE_HD vec2 floor(vec2 a){ return vec2(floor(a.x), floor(a.y)); }
FORGE_HD vec3 floor(vec3 a){ return vec3(floor(a.x), floor(a.y), floor(a.z)); }
FORGE_HD vec2 fract(vec2 a){ return a - floor(a); }
FORGE_HD vec3 fract(vec3 a){ return a - floor(a); }

FORGE_HD float dot(vec2 a, vec2 b){ return a.x*b.x + a.y*b.y; }
FORGE_HD float dot(vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
FORGE_HD float dot(vec4 a, vec4 b){ return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

FORGE_HD float length(vec2 a){ return sqrt(dot(a,a)); }
FORGE_HD float length(vec3 a){ return sqrt(dot(a,a)); }
FORGE_HD float length(vec4 a){ return sqrt(dot(a,a)); }

// As ForgeCPUCompat.h (the oracle that matched the M3 closely): a zero
// vector stays zero rather than becoming NaN.
FORGE_HD vec2 normalize(vec2 a){ float l = length(a); return l > 0.0f ? a / l : a; }
FORGE_HD vec3 normalize(vec3 a){ float l = length(a); return l > 0.0f ? a / l : a; }

FORGE_HD vec3 cross(vec3 a, vec3 b){
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

FORGE_HD vec2 mix(vec2 a, vec2 b, float t){ return a + (b - a) * t; }
FORGE_HD vec3 mix(vec3 a, vec3 b, float t){ return a + (b - a) * t; }
FORGE_HD vec4 mix(vec4 a, vec4 b, float t){ return a + (b - a) * t; }
FORGE_HD vec3 mix(vec3 a, vec3 b, vec3 t){ return a + (b - a) * t; }

FORGE_HD vec3 step(vec3 edge, vec3 x){ return vec3(step(edge.x,x.x), step(edge.y,x.y), step(edge.z,x.z)); }
FORGE_HD vec3 smoothstep(float e0, float e1, vec3 x){
    return vec3(smoothstep(e0,e1,x.x), smoothstep(e0,e1,x.y), smoothstep(e0,e1,x.z));
}

FORGE_HD vec3 exp(vec3 a){ return vec3(exp(a.x), exp(a.y), exp(a.z)); }
FORGE_HD vec3 pow(vec3 a, float b){ return vec3(pow(a.x,b), pow(a.y,b), pow(a.z,b)); }
FORGE_HD vec3 pow(vec3 a, vec3 b){ return vec3(pow(a.x,b.x), pow(a.y,b.y), pow(a.z,b.z)); }
FORGE_HD vec3 sqrt(vec3 a){ return vec3(sqrt(a.x), sqrt(a.y), sqrt(a.z)); }

FORGE_HD vec2 sin(vec2 a){ return vec2(sin(a.x), sin(a.y)); }
FORGE_HD vec3 sin(vec3 a){ return vec3(sin(a.x), sin(a.y), sin(a.z)); }
FORGE_HD vec2 cos(vec2 a){ return vec2(cos(a.x), cos(a.y)); }
FORGE_HD vec3 cos(vec3 a){ return vec3(cos(a.x), cos(a.y), cos(a.z)); }

// The shared modules spell the types the way MSL does.
typedef vec2 float2;
typedef vec3 float3;
typedef vec4 float4;

} // namespace forge_gpu

#endif // FORGE_CUDA_COMPAT_CUH
