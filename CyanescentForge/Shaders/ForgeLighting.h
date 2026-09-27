//
//  ForgeLighting.h
//  Cyanescent Forge — shared shading module
//
//  Direct lighting, image-based ambient, and the sky the world sits in.
//
//  The lighting model is a compact metallic-roughness BRDF rather than
//  something ad hoc, because physically sensible response is what makes an
//  impossible geometry read as a real place. The key light is the drama; the
//  sky and the bounce term are what keep the shadows from going flat black and
//  give the sense of air between the camera and the far structures.
//
#ifndef ForgeLighting_h
#define ForgeLighting_h

#include "ForgeShaderTypes.h"

#include "ForgeMath.h"
#include "ForgeMaterial.h"
#include "ForgeRaymarch.h"

// Sky / far-field radiance. Never black: a ray that escapes still lands on
// atmosphere, so the frame always has depth behind the geometry.
inline float3 forgeSkyRadiance(float3 direction, constant ForgeSceneParams &sp)
{
    float h = saturate(direction.y * 0.5f + 0.5f);
    float3 sky = mix(sp.skyColorHorizon, sp.skyColorZenith, pow(h, 0.75f));

    // Key light disc and a tight halo. The halo is deliberately narrow: a broad
    // one multiplied by a cinematic key intensity floods the whole sky with a
    // flat pale wash and destroys the silhouettes it is supposed to reveal.
    float sun = saturate(dot(direction, sp.keyLightDirection));
    sky = sky + sp.keyLightColor * (pow(sun, 900.0f) * sp.keyLightIntensity * 0.35f
                                    + pow(sun, 40.0f) * 0.055f
                                    + pow(sun, 6.0f) * 0.012f);

    return sky;
}

// GGX normal distribution, Smith visibility, Schlick Fresnel. Trimmed to the
// terms that matter at these roughnesses.
inline float3 forgeBRDF(float3 normal, float3 view, float3 light, ForgeMaterial m)
{
    float3 h = normalize(view + light);
    float ndl = saturate(dot(normal, light));
    float ndv = saturate(dot(normal, view)) + 1e-4f;
    float ndh = saturate(dot(normal, h));
    float vdh = saturate(dot(view, h));

    float a  = max(m.roughness * m.roughness, 1e-3f);
    float a2 = a * a;

    float denom = ndh * ndh * (a2 - 1.0f) + 1.0f;
    float D = a2 / max(FORGE_PI * denom * denom, 1e-6f);

    float k = a * 0.5f;
    float gv = ndv / (ndv * (1.0f - k) + k);
    float gl = ndl / (ndl * (1.0f - k) + k);
    float G = gv * gl;

    float3 f0 = mix(float3(0.04f), m.albedo, m.metallic);
    float3 F = f0 + (float3(1.0f) - f0) * pow(1.0f - vdh, 5.0f);

    float3 specular = F * (D * G / (4.0f * ndv * max(ndl, 1e-4f)));
    float3 diffuse = m.albedo * (1.0f - m.metallic) * (1.0f / FORGE_PI);

    return (diffuse * (float3(1.0f) - F) + specular) * ndl;
}

// Hemisphere ambient. Cheap stand-in for a full irradiance probe, but it is
// directional, so surfaces facing up pick up the zenith colour and surfaces
// facing down pick up bounce — enough to read as a lit environment.
inline float3 forgeAmbient(float3 normal, float3 view, ForgeMaterial m, float ao, constant ForgeSceneParams &sp)
{
    float up = saturate(normal.y * 0.5f + 0.5f);
    float3 irradiance = mix(sp.skyColorHorizon, sp.skyColorZenith, up) * sp.ambientIntensity;

    // Fresnel rim: grazing angles pick up more of the environment. This is the
    // term that traces the silhouettes of distant structures out of the fog.
    float fresnel = pow(1.0f - saturate(dot(normal, view)), 4.0f);
    float3 rim = irradiance * fresnel * mix(0.35f, 1.0f, m.metallic);

    return (m.albedo * irradiance * (1.0f - m.metallic) + rim) * ao;
}

// The lighting model with its two visibility queries already answered. Split
// out so that worlds with their own distance field (Mandelbrot/Newton) share
// exactly the same light rig, material response and fill terms as the
// Apollonian world instead of a re-implementation that drifts.
inline float3 forgeShadeSurfaceLit(float3 normal,
                                   float3 view,          // points from surface towards camera
                                   float viewDistance,
                                   ForgeMaterial m,
                                   float ao,
                                   float shadow,
                                   constant ForgeSceneParams &sp)
{
    float3 lightDir = normalize(sp.keyLightDirection);
    float3 color = float3(0.0f);
    color = color + forgeBRDF(normal, view, lightDir, m) * sp.keyLightColor * sp.keyLightIntensity * shadow;

    // Inside a recursive packing this dense, almost every surface is occluded
    // from a single distant sun — measured, not assumed: a probe of the column
    // interiors returned shadow = 0.001 nearly everywhere. Without the two
    // terms below the world renders as black mud with a few lit rims, which is
    // a lighting failure rather than a dramatic image.

    // 1. Camera-relative lamp. Unshadowed, inverse-square, and it travels with
    //    the viewer, so the structure the camera is threading through always
    //    has shape and the surfaces closest to us read clearly.
    float headlight = sp.headlightIntensity / (1.0f + viewDistance * viewDistance * 0.35f);
    float wrap = saturate(dot(normal, view) * 0.75f + 0.25f);   // soft wrap, not Lambert
    color = color + m.albedo * sp.fillLightColor * (headlight * wrap);

    // 2. One-bounce colour bleed. Squaring the albedo is the standard cheap
    //    approximation to light that has reflected off a neighbouring surface
    //    of the same material, and it is what fills the crevices with the
    //    structure's own colour instead of grey.
    color = color + m.albedo * m.albedo * sp.fillLightColor * (sp.bounceIntensity * ao);

    // 3. Back-fill from the direction opposite the key, standing in for light
    //    that has travelled around the structure.
    float3 fillDir = normalize(float3(-lightDir.x, -0.35f, -lightDir.z));
    float fill = saturate(dot(normal, fillDir));
    color = color + m.albedo * sp.fogColor * (fill * 0.35f * sp.ambientIntensity) * ao;

    color = color + forgeAmbient(normal, view, m, ao, sp);
    color = color + m.emissive;

    return color;
}

// Full surface shading for one hit point of the Apollonian field.
// `viewDistance` is how far the shaded point is from the camera; the
// camera-relative fill light needs it.
inline float3 forgeShadeSurface(float3 position,
                                float3 normal,
                                float3 view,          // points from surface towards camera
                                float viewDistance,
                                ForgeMaterial m,
                                constant ForgeSceneParams &sp,
                                constant ForgeQuality &q)
{
    float3 lightDir = normalize(sp.keyLightDirection);

    float ao = forgeAmbientOcclusion(position, normal, sp, q);

    float shadow = 1.0f;
    if (q.shadowSteps > 0 && dot(normal, lightDir) > 0.0f) {
        shadow = forgeSoftShadow(position + normal * 0.0025f, lightDir, sp, q, 0.01f, 6.0f);
    }

    return forgeShadeSurfaceLit(normal, view, viewDistance, m, ao, shadow, sp);
}

#endif /* ForgeLighting_h */
