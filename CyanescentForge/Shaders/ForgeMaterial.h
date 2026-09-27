//
//  ForgeMaterial.h
//  Cyanescent Forge — shared shading module
//
//  Surface appearance derived from the orbit traps of the distance estimator.
//
//  Why orbit traps: they are a free, continuous, structure-aware
//  parameterisation of the fractal. Two points on the same nested sphere get
//  nearly the same trap values; points on different generations get different
//  ones. Colouring by them makes the recursion *visible* — you can see which
//  structures belong together — which is the difference between a psychedelic
//  world and coloured noise.
//
#ifndef ForgeMaterial_h
#define ForgeMaterial_h

#include "ForgeShaderTypes.h"

#include "ForgeMath.h"
#include "ForgeApollonian.h"

struct ForgeMaterial {
    float3 albedo;
    float  roughness;
    float  metallic;
    float3 emissive;
};

inline ForgeMaterial forgeSurfaceMaterial(ForgeFieldSample f, constant ForgeSceneParams &sp)
{
    // The remapping constants below are not arbitrary: they come from measuring
    // the actual distribution of these quantities over real primary rays
    // (Tools/CPUReference, `stats` mode). Orbit traps occupy a narrow band, and
    // colouring straight from them collapses the whole world to one hue.
    float radius = sqrt(max(f.minRadius2, 0.0f));
    float radiusN = saturate((radius - 0.11f) * 7.5f);      // ~0.14 .. 0.24
    float tx = saturate(f.trap.x * 7.0f);
    float ty = saturate(f.trap.y * 5.0f);
    float tz = saturate(f.trap.z * 8.0f);

    // logScale is the accumulated magnification at the hit point: it says which
    // generation of the recursion this surface belongs to, and it is by far the
    // widest-ranging quantity available. Driving the palette from it is what
    // makes the nesting legible — each generation of structure reads as its own
    // family of colour, so you can see the world repeating into itself.
    float level = saturate((f.logScale - 4.0f) / 8.5f);

    float ident = 0.46f * level + 0.24f * radiusN + 0.18f * ty + 0.12f * tz;

    // Magenta -> violet -> blue -> cyan -> sea green, in that order along
    // `ident`. One cosine family, so however strange the geometry gets every
    // surface belongs to the same colour world. The default paletteShift and
    // paletteSpread deliberately cover only half a cycle: letting the ramp wrap
    // past a full turn puts the same hue on unrelated structures and the
    // recursion stops reading.
    float3 albedo = forgePalette(sp.paletteShift + sp.paletteSpread * ident,
                                 float3(0.32f, 0.34f, 0.38f),
                                 float3(0.38f, 0.36f, 0.40f),
                                 float3(1.00f, 1.00f, 1.00f),
                                 float3(0.00f, 0.33f, 0.55f));

    // A finer band on a different trap axis. This is the chromatic marbling
    // that reads as intricate mineral surface rather than flat plastic.
    float3 accent = forgePalette(sp.paletteShift * 1.7f + 1.6f * tx,
                                 float3(0.60f), float3(0.40f),
                                 float3(1.0f), float3(0.10f, 0.32f, 0.58f));
    albedo = mix(albedo, albedo * accent * 1.4f, 0.30f * sp.surfaceDetail);

    ForgeMaterial m;
    m.albedo = clamp(albedo, 0.015f, 1.0f);

    // Glossier deep in the recursion, rougher on the big open faces. Gives the
    // structure a wet mineral quality and keeps the specular highlights
    // concentrated where the detail is.
    m.roughness = clamp(sp.roughnessBase * mix(1.35f, 0.30f, level), 0.035f, 1.0f);
    m.metallic  = clamp(sp.metallicBase * mix(0.25f, 1.0f, tz), 0.0f, 1.0f);

    // Emission only at the very tightest points of the packing — roughly the
    // deepest tenth of the surface. Sparse on purpose: it has to read as light
    // trapped inside the structure, not as a coloured wash over everything.
    float core = smoothstep(0.205f, 0.145f, radius);
    float deep = smoothstep(0.55f, 0.95f, level);
    m.emissive = sp.emissiveColor * sp.emissiveStrength * (core * core * 1.2f + deep * 0.25f);

    return m;
}

#endif /* ForgeMaterial_h */
