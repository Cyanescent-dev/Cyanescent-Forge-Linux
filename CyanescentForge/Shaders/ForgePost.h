//
//  ForgePost.h
//  Cyanescent Forge — shared shading module
//
//  Display transform. Applied once, after all samples have been accumulated,
//  so that averaging happens in linear light where it is correct.
//
#ifndef ForgePost_h
#define ForgePost_h

#include "ForgeShaderTypes.h"

#include "ForgeMath.h"

// ACES filmic approximation (Narkowicz). Keeps the very bright cores of the
// emissive cavities from clipping to flat white and rolls the highlights the
// way film does, which matters when the intended output is projection.
inline float3 forgeToneMapACES(float3 x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return saturate((x * (a * x + float3(b))) / (x * (c * x + float3(d)) + float3(e)));
}

inline float3 forgeLinearToSRGB(float3 c)
{
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(max(c, float3(0.0f)), 1.0f / 2.4f) - 0.055f;
    return mix(lo, hi, step(float3(0.0031308f), c));
}

// Full display transform for one pixel.
//   `uv` is 0..1 across the image, used for vignette and lateral aberration.
//   `dither` is a deterministic value in [0,1) that breaks up the banding a
//   smooth gradient would otherwise show in an 8-bit file.
inline float3 forgePostProcess(float3 hdr, float2 uv, float exposure, float dither)
{
    float3 c = hdr * exposure;

    // Very slight lateral chromatic shift towards the corners. Enough to read
    // as a lens, not enough to be a look.
    float2 centred = uv - float2(0.5f);
    float r2 = dot(centred, centred);
    c.x = c.x * (1.0f + 0.030f * r2);
    c.z = c.z * (1.0f - 0.030f * r2);

    // Vignette.
    c = c * (1.0f - 0.30f * r2 * r2 - 0.18f * r2);

    c = forgeToneMapACES(c);
    c = forgeLinearToSRGB(c);

    // Ordered-ish dither at roughly one 8-bit step.
    c = c + float3((dither - 0.5f) * (1.0f / 255.0f));

    return saturate(c);
}

#endif /* ForgePost_h */
