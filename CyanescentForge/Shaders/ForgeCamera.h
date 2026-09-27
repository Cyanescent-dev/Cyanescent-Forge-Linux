//
//  ForgeCamera.h
//  Cyanescent Forge — shared shading module
//
//  Primary ray generation. The camera basis is built on the CPU by Camera.swift
//  and handed over as three orthonormal vectors, so the shader never has to
//  know how the path was authored. That separation is deliberate: the camera
//  animation is a timeline concern, not a shader concern, and it means an
//  offline render reproduces the preview path exactly because both read the
//  same evaluated basis.
//
#ifndef ForgeCamera_h
#define ForgeCamera_h

#include "ForgeMath.h"
#include "ForgeShaderTypes.h"

struct ForgeRay {
    float3 origin;
    float3 direction;
};

// `pixel` is in full-image pixel coordinates and may carry a sub-pixel jitter.
// `lensSample` is a point in the unit square used for depth of field; pass
// (0.5, 0.5) for a pinhole camera.
inline ForgeRay forgeGenerateRay(constant ForgeUniforms &u, float2 pixel, float2 lensSample)
{
    float2 ndc;
    ndc.x = (pixel.x / u.imageSize.x) * 2.0f - 1.0f;
    // Pixel rows run top to bottom; the world is right-handed and y is up.
    ndc.y = 1.0f - (pixel.y / u.imageSize.y) * 2.0f;

    float3 dir = normalize(u.cameraForward
                           + u.cameraRight * (ndc.x * u.aspect * u.tanHalfFov)
                           + u.cameraUp    * (ndc.y * u.tanHalfFov));

    ForgeRay ray;
    ray.origin = u.cameraPosition;
    ray.direction = dir;

    // Thin lens. Only ever enabled offline: it needs many samples per pixel to
    // resolve without looking like noise, which is exactly the sort of cost
    // Forge is willing to pay and a realtime renderer is not.
    if (u.quality.enableDepthOfField != 0 && u.scene.apertureRadius > 1e-5f) {
        float cosToAxis = max(dot(dir, u.cameraForward), 1e-4f);
        float3 focalPoint = ray.origin + dir * (u.scene.focusDistance / cosToAxis);
        float2 lens = forgeSampleDisc(lensSample) * u.scene.apertureRadius;
        ray.origin = ray.origin + u.cameraRight * lens.x + u.cameraUp * lens.y;
        ray.direction = normalize(focalPoint - ray.origin);
    }
    return ray;
}

// World-space radius of one pixel at unit distance. Drives the adaptive
// surface epsilon, so detail is resolved to the pixel and no finer — the
// reason a 4K offline frame genuinely contains more structure than an upscaled
// preview rather than merely more samples of the same structure.
inline float forgePixelRadius(constant ForgeUniforms &u)
{
    return u.tanHalfFov / max(u.imageSize.y * 0.5f, 1.0f);
}

#endif /* ForgeCamera_h */
