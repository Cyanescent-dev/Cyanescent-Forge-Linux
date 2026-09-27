//
//  ForgeAtmosphericDepth.h
//  Cyanescent Forge — shared shading module
//
//  Cinematic aerial perspective evaluated from the already-computed primary
//  ray distance. This must never trace or sample the scene: it is purposely a
//  one-exp, one-mix viewing transform in linear HDR space.
//
#ifndef ForgeAtmosphericDepth_h
#define ForgeAtmosphericDepth_h

#include "ForgeShaderTypes.h"

inline float3 forgeApplyAtmosphericDepth(float3 radiance,
                                         float rayDistance,
                                         constant ForgeSceneParams &sp)
{
    if (sp.atmosphericDepthEnabled < 0.5f) {
        return radiance;
    }

    float clearDistance = max(sp.atmosphericDepthStart, 0.0f);
    float travelledThroughAtmosphere = max(rayDistance - clearDistance, 0.0f);
    float opticalDepth = max(sp.atmosphericDepthDensity, 0.0f) * travelledThroughAtmosphere;
    float attenuation = 1.0f - exp(-opticalDepth);
    float amount = saturate(attenuation * saturate(sp.atmosphericDepthMaximumStrength));
    return mix(radiance, sp.atmosphericDepthColor, amount);
}

#endif /* ForgeAtmosphericDepth_h */
