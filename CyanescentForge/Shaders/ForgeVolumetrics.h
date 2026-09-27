//
//  ForgeVolumetrics.h
//  Cyanescent Forge — shared shading module
//
//  Participating medium along the primary ray.
//
//  This is the single most expensive thing in the renderer and the clearest
//  demonstration of why Forge exists: every step of the fog march evaluates the
//  distance field (to thicken the medium where it pools in the crevices) and
//  fires a short shadow ray (to carve god rays out of the structure). At
//  offline sample counts it produces genuine atmospheric depth; at preview
//  counts it degrades gracefully to a coloured haze.
//
#ifndef ForgeVolumetrics_h
#define ForgeVolumetrics_h

#include "ForgeShaderTypes.h"

#include "ForgeMath.h"
#include "ForgeRaymarch.h"

// Henyey-Greenstein phase function. g > 0 pushes scattering forward, which is
// what makes the air glow around the key light.
inline float forgeHenyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float d = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / max(4.0f * FORGE_PI * d * sqrt(max(d, 1e-4f)), 1e-4f);
}

struct ForgeVolumeResult {
    float3 inscatter;      // radiance added along the ray
    float3 transmittance;  // what survives of whatever is behind it
};

inline ForgeVolumeResult forgeMarchVolume(float3 origin,
                                          float3 direction,
                                          float tEnd,
                                          constant ForgeSceneParams &sp,
                                          constant ForgeQuality &q,
                                          unsigned int rngSeed)
{
    ForgeVolumeResult result;
    result.inscatter = float3(0.0f);
    result.transmittance = float3(1.0f);

    int steps = q.volumetricSamples;
    if (steps <= 0 || sp.fogDensity <= 1e-5f) {
        // Still apply plain analytic extinction so distance reads correctly.
        float ext = 1.0f - exp(-sp.fogDensity * tEnd);
        result.transmittance = float3(1.0f - ext);
        result.inscatter = sp.fogColor * (sp.ambientIntensity * 1.7f) * ext;
        return result;
    }

    float far = min(tEnd, q.maxDistance);
    float stepLength = far / float(steps);

    float3 lightDir = normalize(sp.keyLightDirection);
    float phase = forgeHenyeyGreenstein(dot(direction, lightDir), 0.55f);

    // Extinction is tinted: blue survives further than red, so distant
    // structures cool off. Cheap, and it is most of the sense of vast scale.
    float3 extinctionTint = float3(1.25f, 1.05f, 0.85f);

    for (int i = 0; i < steps; ++i) {
        // Deterministic stratified jitter. Seeded from the pixel and frame, so
        // the same frame always produces the same fog, but neighbouring pixels
        // do not band.
        float jitter = forgeRandom(forgeHashCombine(rngSeed, (unsigned int)(i + 1)));
        float t = (float(i) + jitter) * stepLength;
        float3 p = origin + direction * t;

        // Fog pools around the geometry. One distance evaluation per step: the
        // expensive part, and the reason the crevices fill with light.
        float d = forgeApollonianDistance(p, sp, q.fractalIterations, q.macroLayerIterations);
        float proximity = exp(-max(d, 0.0f) * 6.0f);
        float density = sp.fogDensity * (1.0f + sp.fogHeightFalloff * proximity);

        float3 stepExtinction = exp(-density * extinctionTint * stepLength);

        // Ambient scattering. This term is what the far distance fades *to*:
        // set it too low and long sight-lines go black instead of hazy, which
        // destroys the aerial perspective that makes the world read as vast.
        float3 lit = sp.fogColor * (sp.ambientIntensity * 1.7f);

        if (sp.godRayStrength > 1e-4f && q.shadowSteps > 0) {
            // Short shadow ray towards the key light. Truncated hard: shafts
            // only need local occlusion to read convincingly.
            float shadow = forgeSoftShadow(p, lightDir, sp, q, 0.02f, 3.0f);
            lit = lit + sp.keyLightColor * sp.keyLightIntensity * shadow * phase * sp.godRayStrength;
        }

        // Energy-conserving integration of the segment. (1 - stepExtinction) is
        // already the fraction of light absorbed over this step, so density must
        // NOT be applied a second time; the scattering albedo is what decides
        // how much of the absorbed energy comes back out as inscatter.
        const float scatteringAlbedo = 0.82f;
        float3 segment = (float3(1.0f) - stepExtinction) * lit * scatteringAlbedo;
        result.inscatter = result.inscatter + result.transmittance * segment;
        result.transmittance = result.transmittance * stepExtinction;

        if (forgeLuminance(result.transmittance) < 0.003f) {
            break;
        }
    }

    return result;
}

#endif /* ForgeVolumetrics_h */
