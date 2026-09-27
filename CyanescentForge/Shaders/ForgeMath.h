//
//  ForgeMath.h
//  Cyanescent Forge — shared shading module
//
//  Small numerical helpers and deterministic hashing. Everything random in
//  Forge comes from these functions, seeded from the project seed and the
//  frame index, so a render is reproducible bit-for-bit. Nothing here may read
//  wall-clock time.
//
#ifndef ForgeMath_h
#define ForgeMath_h

#include "ForgeShaderTypes.h"

#define FORGE_PI      3.14159265358979323846f
#define FORGE_TAU     6.28318530717958647692f

// --- deterministic hashing --------------------------------------------------
//
// PCG-style integer hash. Cheap, well distributed, and identical on CPU and
// GPU because it is pure 32-bit integer arithmetic.
inline unsigned int forgeHashU32(unsigned int v) {
    v ^= v >> 17;
    v *= 0xed5ad4bbu;
    v ^= v >> 11;
    v *= 0xac4c1b51u;
    v ^= v >> 15;
    v *= 0x31848babu;
    v ^= v >> 14;
    return v;
}

inline unsigned int forgeHashCombine(unsigned int a, unsigned int b) {
    return forgeHashU32(a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2)));
}

// Uniform float in [0,1).
inline float forgeRandom(unsigned int state) {
    return float(forgeHashU32(state) & 0x00ffffffu) / 16777216.0f;
}

// --- low discrepancy sampling ----------------------------------------------
//
// Van der Corput / Hammersley pairs give far smoother convergence than plain
// white noise for the small sample counts a preview uses, and remain unbiased
// at the large counts an offline render uses.
inline float forgeRadicalInverse(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

// Van der Corput in base 3, for the second Halton dimension.
inline float forgeRadicalInverseBase3(unsigned int i) {
    float f = 1.0f;
    float r = 0.0f;
    // 21 iterations exhausts a 32-bit index in base 3; the bound also keeps the
    // loop provably terminating for the shader compiler.
    for (int k = 0; k < 21 && i > 0u; ++k) {
        f = f / 3.0f;
        r = r + f * float(i % 3u);
        i = i / 3u;
    }
    return r;
}

// Sample `index` of a Halton (2,3) sequence, Cranley-Patterson rotated by a
// per-pixel hash so neighbouring pixels do not share a pattern.
//
// Halton rather than Hammersley on purpose. Hammersley's first dimension is
// index/count, which needs the sample count up front — and the progressive
// preview does not have one. Passing the running count pins the jitter to a
// single sub-pixel position (so accumulating never antialiases); passing a
// large fixed count clusters the first samples together instead. Halton is well
// distributed at *every* prefix length, so the same sequence serves a one-sample
// preview, a preview that has been converging for a minute, and a 36-sample
// offline frame.
inline float2 forgeHalton(unsigned int index, unsigned int rotation) {
    float u = fract(forgeRadicalInverse(index) + forgeRandom(rotation));
    float v = fract(forgeRadicalInverseBase3(index) + forgeRandom(rotation ^ 0x68bc21ebu));
    return float2(u, v);
}

// Uniform point on a disc, from a unit square sample. Used for the lens.
inline float2 forgeSampleDisc(float2 uv) {
    float r = sqrt(uv.x);
    float a = FORGE_TAU * uv.y;
    return float2(r * cos(a), r * sin(a));
}

// --- rotations --------------------------------------------------------------
inline float3 forgeRotateY(float3 p, float a) {
    float c = cos(a), s = sin(a);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

inline float3 forgeRotateX(float3 p, float a) {
    float c = cos(a), s = sin(a);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

inline float3 forgeRotateZ(float3 p, float a) {
    float c = cos(a), s = sin(a);
    return float3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

// --- misc -------------------------------------------------------------------
inline float forgeSmoothMin(float a, float b, float k) {
    float h = saturate(0.5f + 0.5f * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0f - h);
}

inline float forgeLuminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// IQ's cosine gradient palette. Coherent by construction: any input produces a
// colour that belongs to the same family, which is why the world stays
// psychedelic without becoming a rainbow mess.
inline float3 forgePalette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(FORGE_TAU * (c * t + d));
}

#endif /* ForgeMath_h */
