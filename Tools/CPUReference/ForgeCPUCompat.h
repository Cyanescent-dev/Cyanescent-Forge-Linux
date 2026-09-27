//
//  ForgeCPUCompat.h
//  Cyanescent Forge — CPU reference harness
//
//  Purpose
//  -------
//  The shading modules under CyanescentForge/Shaders/ are written in Metal
//  Shading Language, but deliberately restricted to a subset that is *also*
//  valid C++17. This header supplies the vector types and builtin functions
//  that MSL provides intrinsically, so the exact same source files can be
//  compiled by g++/clang on any machine and executed on the CPU.
//
//  That is what lets us verify the Apollonian distance estimator, the camera
//  path and the lighting model numerically and visually without a Mac.
//
//  Rules the shared modules follow so this works:
//    * only float2/float3/float4 vectors (no half, no packed_, no matrices
//      beyond the small helpers defined here)
//    * swizzles limited to the set implemented below
//    * no address-space qualifiers (device/constant/threadgroup) in the
//      pure-maths modules; those live only in the .metal kernel file
//
#ifndef ForgeCPUCompat_h
#define ForgeCPUCompat_h

#include <cmath>
#include <cstdint>
#include <algorithm>

#define FORGE_CPU_REFERENCE 1

// MSL qualifiers that have no meaning in the portable shader modules.
// `thread` is deliberately not defined: these modules do not use it, and a
// macro with that name would corrupt `std::thread` in C++ headers included by
// the harness after this file.
#define constant const

namespace forge_compat {

struct vec2; struct vec3; struct vec4;

// ---------------------------------------------------------------------------
// Swizzle proxies.
//
// MSL exposes swizzles as data members (p.xy). To keep the shared source
// identical we store the components in a raw array and overlay named proxy
// members that convert to the appropriate vector type. Only the swizzles the
// shaders actually use are provided; anything else is a compile error here,
// which is the desired behaviour (it tells us the module drifted out of the
// portable subset).
// ---------------------------------------------------------------------------
template <int N, int A, int B> struct Sw2 { float d[N]; inline operator vec2() const; };
template <int N, int A, int B, int C> struct Sw3 { float d[N]; inline operator vec3() const; };

struct vec2 {
    union {
        float d[2];
        struct { float x, y; };
    };
    vec2() { x = 0; y = 0; }
    vec2(float s) { x = s; y = s; }
    vec2(float a, float b) { x = a; y = b; }
    float  operator[](int i) const { return d[i]; }
    float& operator[](int i)       { return d[i]; }
};

struct vec3 {
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
    vec3() { x = 0; y = 0; z = 0; }
    vec3(float s) { x = s; y = s; z = s; }
    vec3(float a, float b, float c) { x = a; y = b; z = c; }
    vec3(vec2 a, float c) { x = a.x; y = a.y; z = c; }
    vec3(float a, vec2 b) { x = a; y = b.x; z = b.y; }
    float  operator[](int i) const { return d[i]; }
    float& operator[](int i)       { return d[i]; }
};

struct vec4 {
    union {
        float d[4];
        struct { float x, y, z, w; };
        Sw2<4,0,1> xy;
        Sw2<4,2,3> zw;
        Sw3<4,0,1,2> xyz;
    };
    vec4() { x = 0; y = 0; z = 0; w = 0; }
    vec4(float s) { x = s; y = s; z = s; w = s; }
    vec4(float a, float b, float c, float e) { x = a; y = b; z = c; w = e; }
    vec4(vec3 a, float e) { x = a.x; y = a.y; z = a.z; w = e; }
    float  operator[](int i) const { return d[i]; }
    float& operator[](int i)       { return d[i]; }
};

template <int N, int A, int B> inline Sw2<N,A,B>::operator vec2() const { return vec2(d[A], d[B]); }
template <int N, int A, int B, int C> inline Sw3<N,A,B,C>::operator vec3() const { return vec3(d[A], d[B], d[C]); }

// --- arithmetic -------------------------------------------------------------
#define FORGE_VEC2_OP(OP) \
    inline vec2 operator OP (vec2 a, vec2 b){ return vec2(a.x OP b.x, a.y OP b.y); } \
    inline vec2 operator OP (vec2 a, float b){ return vec2(a.x OP b, a.y OP b); } \
    inline vec2 operator OP (float a, vec2 b){ return vec2(a OP b.x, a OP b.y); }
#define FORGE_VEC3_OP(OP) \
    inline vec3 operator OP (vec3 a, vec3 b){ return vec3(a.x OP b.x, a.y OP b.y, a.z OP b.z); } \
    inline vec3 operator OP (vec3 a, float b){ return vec3(a.x OP b, a.y OP b, a.z OP b); } \
    inline vec3 operator OP (float a, vec3 b){ return vec3(a OP b.x, a OP b.y, a OP b.z); }
#define FORGE_VEC4_OP(OP) \
    inline vec4 operator OP (vec4 a, vec4 b){ return vec4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w); } \
    inline vec4 operator OP (vec4 a, float b){ return vec4(a.x OP b, a.y OP b, a.z OP b, a.w OP b); } \
    inline vec4 operator OP (float a, vec4 b){ return vec4(a OP b.x, a OP b.y, a OP b.z, a OP b.w); }

FORGE_VEC2_OP(+) FORGE_VEC2_OP(-) FORGE_VEC2_OP(*) FORGE_VEC2_OP(/)
FORGE_VEC3_OP(+) FORGE_VEC3_OP(-) FORGE_VEC3_OP(*) FORGE_VEC3_OP(/)
FORGE_VEC4_OP(+) FORGE_VEC4_OP(-) FORGE_VEC4_OP(*) FORGE_VEC4_OP(/)

inline vec2 operator-(vec2 a){ return vec2(-a.x, -a.y); }
inline vec3 operator-(vec3 a){ return vec3(-a.x, -a.y, -a.z); }
inline vec4 operator-(vec4 a){ return vec4(-a.x, -a.y, -a.z, -a.w); }

#define FORGE_ASSIGN(T, OP) \
    inline T& operator OP##= (T& a, T b){ a = a OP b; return a; } \
    inline T& operator OP##= (T& a, float b){ a = a OP b; return a; }
FORGE_ASSIGN(vec2,+) FORGE_ASSIGN(vec2,-) FORGE_ASSIGN(vec2,*) FORGE_ASSIGN(vec2,/)
FORGE_ASSIGN(vec3,+) FORGE_ASSIGN(vec3,-) FORGE_ASSIGN(vec3,*) FORGE_ASSIGN(vec3,/)
FORGE_ASSIGN(vec4,+) FORGE_ASSIGN(vec4,-) FORGE_ASSIGN(vec4,*) FORGE_ASSIGN(vec4,/)

// --- builtins ---------------------------------------------------------------
using std::fabs; using std::floor; using std::sqrt; using std::pow;
using std::exp; using std::log; using std::sin; using std::cos; using std::tan;

inline vec2 abs(vec2 a){ return vec2(std::fabs(a.x), std::fabs(a.y)); }
inline vec3 abs(vec3 a){ return vec3(std::fabs(a.x), std::fabs(a.y), std::fabs(a.z)); }
inline vec4 abs(vec4 a){ return vec4(std::fabs(a.x), std::fabs(a.y), std::fabs(a.z), std::fabs(a.w)); }

inline float min(float a, float b){ return a < b ? a : b; }
inline unsigned int min(unsigned int a, unsigned int b){ return a < b ? a : b; }
inline unsigned int max(unsigned int a, unsigned int b){ return a > b ? a : b; }
inline int min(int a, int b){ return a < b ? a : b; }
inline int max(int a, int b){ return a > b ? a : b; }
inline float max(float a, float b){ return a > b ? a : b; }
inline vec2 min(vec2 a, vec2 b){ return vec2(min(a.x,b.x), min(a.y,b.y)); }
inline vec3 min(vec3 a, vec3 b){ return vec3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
inline vec4 min(vec4 a, vec4 b){ return vec4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)); }
inline vec2 max(vec2 a, vec2 b){ return vec2(max(a.x,b.x), max(a.y,b.y)); }
inline vec3 max(vec3 a, vec3 b){ return vec3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
inline vec4 max(vec4 a, vec4 b){ return vec4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)); }
inline vec3 min(vec3 a, float b){ return min(a, vec3(b)); }
inline vec3 max(vec3 a, float b){ return max(a, vec3(b)); }

inline float clamp(float v, float lo, float hi){ return min(max(v, lo), hi); }
inline vec2 clamp(vec2 v, float lo, float hi){ return min(max(v, vec2(lo)), vec2(hi)); }
inline vec3 clamp(vec3 v, float lo, float hi){ return min(max(v, vec3(lo)), vec3(hi)); }
inline vec4 clamp(vec4 v, float lo, float hi){ return min(max(v, vec4(lo)), vec4(hi)); }
inline vec3 clamp(vec3 v, vec3 lo, vec3 hi){ return min(max(v, lo), hi); }

inline float saturate(float v){ return clamp(v, 0.0f, 1.0f); }
inline vec2 saturate(vec2 v){ return clamp(v, 0.0f, 1.0f); }
inline vec3 saturate(vec3 v){ return clamp(v, 0.0f, 1.0f); }
inline vec4 saturate(vec4 v){ return clamp(v, 0.0f, 1.0f); }

inline vec2 floor(vec2 a){ return vec2(std::floor(a.x), std::floor(a.y)); }
inline vec3 floor(vec3 a){ return vec3(std::floor(a.x), std::floor(a.y), std::floor(a.z)); }

inline float fract(float a){ return a - std::floor(a); }
inline vec2 fract(vec2 a){ return a - floor(a); }
inline vec3 fract(vec3 a){ return a - floor(a); }

inline float dot(vec2 a, vec2 b){ return a.x*b.x + a.y*b.y; }
inline float dot(vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float dot(vec4 a, vec4 b){ return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

inline float length(vec2 a){ return std::sqrt(dot(a,a)); }
inline float length(vec3 a){ return std::sqrt(dot(a,a)); }
inline float length(vec4 a){ return std::sqrt(dot(a,a)); }

inline vec2 normalize(vec2 a){ float l = length(a); return l > 0.0f ? a / l : a; }
inline vec3 normalize(vec3 a){ float l = length(a); return l > 0.0f ? a / l : a; }

inline vec3 cross(vec3 a, vec3 b){
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline float mix(float a, float b, float t){ return a + (b - a) * t; }
inline vec2 mix(vec2 a, vec2 b, float t){ return a + (b - a) * t; }
inline vec3 mix(vec3 a, vec3 b, float t){ return a + (b - a) * t; }
inline vec4 mix(vec4 a, vec4 b, float t){ return a + (b - a) * t; }
inline vec3 mix(vec3 a, vec3 b, vec3 t){ return a + (b - a) * t; }

inline float step(float edge, float x){ return x < edge ? 0.0f : 1.0f; }
inline vec3 step(vec3 edge, vec3 x){ return vec3(step(edge.x,x.x), step(edge.y,x.y), step(edge.z,x.z)); }

inline float smoothstep(float e0, float e1, float x){
    float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}
inline vec3 smoothstep(float e0, float e1, vec3 x){
    return vec3(smoothstep(e0,e1,x.x), smoothstep(e0,e1,x.y), smoothstep(e0,e1,x.z));
}

inline float sign(float a){ return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f); }
inline float rsqrt(float a){ return 1.0f / std::sqrt(a); }
inline float exp2(float a){ return std::exp2(a); }

inline vec3 exp(vec3 a){ return vec3(std::exp(a.x), std::exp(a.y), std::exp(a.z)); }
inline vec3 pow(vec3 a, float b){ return vec3(std::pow(a.x,b), std::pow(a.y,b), std::pow(a.z,b)); }
inline vec3 pow(vec3 a, vec3 b){ return vec3(std::pow(a.x,b.x), std::pow(a.y,b.y), std::pow(a.z,b.z)); }
inline vec3 sqrt(vec3 a){ return vec3(std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z)); }

inline vec2 sin(vec2 a){ return vec2(std::sin(a.x), std::sin(a.y)); }
inline vec3 sin(vec3 a){ return vec3(std::sin(a.x), std::sin(a.y), std::sin(a.z)); }
inline vec3 cos(vec3 a){ return vec3(std::cos(a.x), std::cos(a.y), std::cos(a.z)); }
inline vec2 cos(vec2 a){ return vec2(std::cos(a.x), std::cos(a.y)); }

inline float atan2(float y, float x){ return std::atan2(y, x); }
inline float precise_divide(float a, float b){ return a / b; }

} // namespace forge_compat

using namespace forge_compat;

// The shared modules spell the types the way MSL does.
typedef forge_compat::vec2 float2;
typedef forge_compat::vec3 float3;
typedef forge_compat::vec4 float4;

#endif /* ForgeCPUCompat_h */
