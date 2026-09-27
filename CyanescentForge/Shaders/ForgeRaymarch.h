//
//  ForgeRaymarch.h
//  Cyanescent Forge — shared shading module
//
//  Sphere tracing against the Apollonian field, plus the two visibility
//  queries (soft shadows, ambient occlusion) that need their own marches.
//
//  The step budget and the surface epsilon are both quality-driven. Preview
//  stops early and accepts a fuzzy silhouette; offline marches until the
//  detail is resolved to sub-pixel accuracy. Neither changes the geometry.
//
#ifndef ForgeRaymarch_h
#define ForgeRaymarch_h

#include "ForgeShaderTypes.h"

#include "ForgeApollonian.h"

struct ForgeMarchResult {
    int    hit;         // 1 on surface, 0 on escape
    float  t;           // distance travelled
    float3 position;
    ForgeFieldSample field;
    float  glow;        // light bled out of nearby geometry along the ray
    float  steps01;     // fraction of the step budget consumed
};

inline ForgeMarchResult forgeMarch(float3 origin,
                                   float3 direction,
                                   constant ForgeSceneParams &sp,
                                   constant ForgeQuality &q,
                                   float pixelRadius,
                                   float tStart)
{
    ForgeMarchResult r;
    r.hit = 0;
    r.t = tStart;
    r.position = origin;
    r.glow = 0.0f;
    r.steps01 = 0.0f;

    float t = tStart;
    int i = 0;
    ForgeFieldSample f;

    for (i = 0; i < q.raymarchSteps; ++i) {
        float3 p = origin + direction * t;
        f = forgeApollonianWorld(p, sp, q.fractalIterations, q.macroLayerIterations);
        float d = f.distance;

        // Adaptive epsilon: resolve detail down to the size of a pixel at this
        // distance, and no further. detailScale below 1 sharpens the surface at
        // real cost, which is the offline renderer's main lever.
        float eps = max(t * pixelRadius * q.detailScale, 1e-6f);

        if (d < eps) {
            r.hit = 1;
            break;
        }
        if (t > q.maxDistance) {
            break;
        }

        // 0.92 relaxation: the estimator is a lower bound, but the fold makes
        // it locally optimistic near cell boundaries. Backing off slightly
        // costs steps and removes the banded surface artefacts that otherwise
        // appear exactly where the recursion is most interesting.
        float stepLength = max(d * 0.92f, eps * 0.5f);

        // Proximity glow, integrated properly along the ray: light leaking out
        // of the structure the ray is squeezing past. This is what makes the
        // narrow recursive corridors read as luminous rather than merely dark.
        r.glow += exp(-max(d, 0.0f) * 9.0f) * stepLength;

        t += stepLength;
    }

    // Normal quality configurations always execute the loop. Retain defined
    // behaviour for a zero-step diagnostic configuration without paying for a
    // discarded field evaluation on every ordinary ray.
    if (q.raymarchSteps <= 0) {
        f = forgeApollonianWorld(origin, sp, q.fractalIterations, q.macroLayerIterations);
    }

    r.t = t;
    r.position = origin + direction * t;
    r.field = f;
    r.steps01 = float(i) / float(max(q.raymarchSteps, 1));
    return r;
}

// IQ-style soft shadow: the penumbra comes free from the distance estimate,
// because how close the shadow ray passed to the occluder is exactly the
// distance field value along it.
inline float forgeSoftShadow(float3 origin,
                             float3 lightDirection,
                             constant ForgeSceneParams &sp,
                             constant ForgeQuality &q,
                             float tMin,
                             float tMax)
{
    float res = 1.0f;
    float t = tMin;
    float k = max(q.shadowSoftness, 1.0f);

    for (int i = 0; i < q.shadowSteps; ++i) {
        float h = forgeApollonianDistance(origin + lightDirection * t, sp,
                                          q.fractalIterations, q.macroLayerIterations);
        res = min(res, k * h / t);
        t += clamp(h, 0.0015f, 0.09f);
        if (res < 0.002f || t > tMax) {
            break;
        }
    }
    return saturate(res);
}

// Cone-ish ambient occlusion by probing the field along the normal. More taps
// reach further and produce the deep contact darkening in the crevices that
// makes the recursion legible.
inline float forgeAmbientOcclusion(float3 position,
                                   float3 normal,
                                   constant ForgeSceneParams &sp,
                                   constant ForgeQuality &q)
{
    float occlusion = 0.0f;
    float scale = 1.0f;
    int n = max(q.aoSamples, 1);

    for (int i = 0; i < n; ++i) {
        float h = 0.004f + 0.16f * float(i) / float(n);
        float d = forgeApollonianDistance(position + normal * h, sp,
                                          q.fractalIterations, q.macroLayerIterations);
        occlusion += (h - d) * scale;
        scale *= 0.88f;
    }
    return saturate(1.0f - 2.4f * occlusion);
}

#endif /* ForgeRaymarch_h */
