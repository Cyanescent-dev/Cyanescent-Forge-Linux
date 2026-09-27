//
//  ForgeMandelNewton.h
//  Cyanescent Forge — shared shading module
//
//  The Mandelbrot / Newton Morph world.
//
//  The iteration is the known-good morph: at every step the Mandelbrot map
//  z^2 + c and the Newton step for z^3 - 1 are blended by the balance, and the
//  starting point blends from the critical point (Mandelbrot) to the point
//  itself (Newton). Intermediate balances are therefore a genuinely different
//  dynamical system, not a crossfade of two pictures. Escape gives the smooth
//  iteration count of the Mandelbrot side; convergence gives the root and a
//  smooth convergence depth of the Newton side; orbit traps record the
//  filaments and the unit-circle law boundary.
//
//  Two views share the field: the flat plane, which is the reference image,
//  and the relief, where the convergence depth of the same evaluation is the
//  terrain the camera flies over, lit by Forge's lighting model.
//
#ifndef ForgeMandelNewton_h
#define ForgeMandelNewton_h

#include "ForgeShaderTypes.h"
#include "ForgeMath.h"
#include "ForgeCamera.h"
#include "ForgeMaterial.h"
#include "ForgeLighting.h"
#include "ForgeVolumetrics.h"
#include "ForgeAtmosphericDepth.h"

inline float2 mnmCMul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline float2 mnmCDiv(float2 a, float2 b) {
    float d = dot(b, b) + 1e-9f;
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}

inline float3 mnmPalA(float t) {
    return 0.5f + 0.5f * cos(FORGE_TAU * (t + float3(0.00f, 0.10f, 0.22f)) + float3(0.0f, 0.6f, 1.2f));
}
inline float3 mnmPalB(float t) {
    return 0.5f + 0.5f * cos(FORGE_TAU * (t + float3(0.55f, 0.35f, 0.10f)) + float3(1.0f, 2.0f, 3.0f));
}

struct MNMField {
    float escaped;
    float converged;
    float smoothIter;   // Mandelbrot side: smooth escape count
    float convIter;     // Newton side: smooth convergence depth
    float rootId;
    float trap;         // min |z|^2 over the orbit
    float trapLine;     // min distance to the unit-circle law boundary
    float depth;        // 0..1 normalised depth
    float boundary;     // distance estimate to the fractal boundary, plane units
};

// The shared quality presets count Apollonian recursion levels (8..24); the
// morph iteration needs far more to resolve its boundaries, so the world's
// count is derived from the preset rather than read directly: 64 at the fast
// preview, 192 at the extreme offline preset.
inline int mnmIterations(constant ForgeQuality &q)
{
    return clamp(q.fractalIterations * 8, 64, 512);
}

// Plane coordinate for a point: rotation, zoom, seed drift and warp exactly as
// the reference, with the audio terms replaced by authored parameters.
inline float2 mnmPlanePoint(float2 uv, constant ForgeSceneParams &sp)
{
    float morph = saturate(sp.mnBalance);
    float coexist = smoothstep(0.12f, 0.5f, morph) * smoothstep(0.88f, 0.5f, morph);
    float2 p = forgeRotateZ(float3(uv, 0.0f), sp.mnRotation).xy;
    p = p * (mix(1.35f, 2.4f, morph) * sp.mnZoom);
    p = p + float2(sp.mnCenterX, sp.mnCenterY);
    float warpAmp = sp.mnWarp * coexist;
    p = p + warpAmp * float2(sin(p.y * 7.0f), sin(p.x * 6.0f));
    return p;
}

inline MNMField mnmEvaluate(float2 p, float morph, int iters)
{
    float2 z = p * morph;            // mix(0, p, morph)
    float2 c = p;
    const float2 root0 = float2(1.0f, 0.0f);
    const float2 root1 = float2(-0.5f, 0.8660254f);
    const float2 root2 = float2(-0.5f, -0.8660254f);
    const float convergeThreshold = 0.0008f;

    MNMField f;
    f.escaped = 0.0f; f.converged = 0.0f; f.smoothIter = 0.0f; f.convIter = 0.0f;
    f.rootId = 0.0f; f.trap = 1e9f; f.trapLine = 1e9f; f.depth = 1.0f; f.boundary = 0.0f;

    // dz/dp along the orbit. z0 = morph·p and c = p, so the derivative starts
    // at morph and every step adds the Mandelbrot side's dc/dp = 1.
    float2 dz = float2(morph, 0.0f);
    float2 dzPrevious = dz;
    float maxSensitivity = 0.0f;
    float previousStep = 1.0f;
    for (int i = 0; i < iters; ++i) {
        dzPrevious = dz;
        float2 z2 = mnmCMul(z, z);
        float2 zm = z2 + c;                                         // Mandelbrot
        float2 z3 = mnmCMul(z2, z);
        float2 denom = 3.0f * z2 + float2(1e-6f, 0.0f);
        float2 zn = z - mnmCDiv(z3 - float2(1.0f, 0.0f), denom);   // Newton
        float2 znew = mix(zm, zn, morph);

        // Derivative of the blended step: (1-m)·2z + m·N'(z), N' = 2(z³-1)/(3z³).
        float2 dNewton = mnmCDiv(2.0f * (z3 - float2(1.0f, 0.0f)), 3.0f * z3 + float2(1e-6f, 0.0f));
        float2 dStep = mix(2.0f * z, dNewton, morph);
        dz = mnmCMul(dStep, dz) + float2(1.0f - morph, 0.0f);
        maxSensitivity = max(maxSensitivity, length(dz));

        f.trap = min(f.trap, dot(znew, znew));
        f.trapLine = min(f.trapLine, abs(length(znew) - 1.0f) + 0.35f * abs(znew.y));

        float step = length(znew - z);
        z = znew;

        float m2 = dot(z, z);
        if (m2 > 64.0f) {
            f.smoothIter = float(i) - log2(log2(m2)) + 4.0f;
            f.escaped = 1.0f;
            f.depth = saturate(f.smoothIter / float(iters));
            // The pure escape-time estimate needs the derivative to keep
            // growing along the orbit; in the hybrid the Newton term contracts
            // it, so the orbit's peak sensitivity is used instead — an orbit
            // that was ever this sensitive passed that close to the boundary.
            float r = sqrt(m2);
            f.boundary = 0.5f * r * log(r) / max(maxSensitivity, 1e-20f);
            break;
        }
        if (step < convergeThreshold && morph > 0.02f) {
            float d0 = dot(z - root0, z - root0);
            float d1 = dot(z - root1, z - root1);
            float d2 = dot(z - root2, z - root2);
            f.rootId = 0.0f;
            float dm = d0;
            if (d1 < dm) { dm = d1; f.rootId = 1.0f; }
            if (d2 < dm) { dm = d2; f.rootId = 2.0f; }
            // Fractional convergence count: where log(step) crossed the
            // threshold between the previous iteration and this one. This is
            // continuous whatever the contraction rate (superattracting at
            // the Newton end, geometric in the hybrid zone), so the depth has
            // no bands and the relief no terraces.
            float lp = log(max(previousStep, 1e-30f)), ls = log(max(step, 1e-30f));
            float lt = log(convergeThreshold);
            float fraction = saturate((lp - lt) / max(lp - ls, 1e-6f));
            f.convIter = max(float(i) - 1.0f + fraction, 0.0f);
            f.converged = 1.0f;
            f.depth = saturate(f.convIter / float(iters));
            // Distance to the basin boundary: the residual (the last step —
            // at intermediate balance the attractor is the hybrid's own fixed
            // point, not the cube root) pulled back through the residual's
            // own derivative, d(z_{n+1} - z_n)/dp. Both scale by the
            // multiplier under one more iteration, so the ratio is
            // continuous across the bands where the stopping step changes;
            // using dz itself is not, because of the parameter term.
            f.boundary = 1.6f * step / max(length(dz - dzPrevious), 1e-20f);
            break;
        }
        previousStep = step;
    }
    return f;
}

// The reference colouring, as linear radiance.
// `lod` is 1 at full detail and falls toward 0 where the plane's fine bands
// are smaller than a pixel; the high-frequency band modulation fades with it
// so distant ground converges to its mean colour instead of aliasing.
inline float3 mnmColour(MNMField f, float morph, int iters, constant ForgeSceneParams &sp, float lod)
{
    float coexist = smoothstep(0.12f, 0.5f, morph) * smoothstep(0.88f, 0.5f, morph);
    float drift = sp.mnPaletteDrift;
    float trapGlow = exp(-sqrt(f.trap) * 3.0f);
    float lineGlow = exp(-f.trapLine * 9.0f);
    float3 color = float3(0.015f, 0.02f, 0.05f)
        + trapGlow * mix(float3(0.10f, 0.05f, 0.22f), float3(0.03f, 0.12f, 0.20f), morph) * sp.mnTrapInfluence
        + lineGlow * 0.30f * float3(0.4f, 0.7f, 1.0f) * coexist * sp.mnTrapInfluence;

    if (f.escaped > 0.5f) {
        float t = f.smoothIter * 0.035f * sp.mnBandDensity + drift;
        float3 mCol = mnmPalA(t);
        mCol *= 0.6f + 0.5f * lod * sin(f.smoothIter * 0.35f * sp.mnBandDensity);
        float fil = exp(-f.trapLine * 6.0f);
        mCol += fil * 0.6f * float3(0.9f, 0.8f, 1.0f) * sp.mnTrapInfluence;
        color = mix(color, mCol, 0.92f);
    }
    if (f.converged > 0.5f) {
        float hue = f.rootId / 3.0f + drift + 0.08f * morph;
        float3 nCol = mnmPalB(hue);
        float shade = exp(-f.convIter * 0.08f);
        nCol *= 0.35f + 0.85f * shade;
        float boundary = smoothstep(4.0f, float(iters), f.convIter);
        nCol += boundary * sp.mnBoundaryGlow * float3(0.6f, 0.85f, 1.0f);
        float nMix = mix(0.9f, 0.55f + 0.35f * lod * sin(f.convIter * 0.7f), coexist);
        color = mix(color, nCol, saturate(nMix));
    }
    float lawBeat = coexist * sp.mnCoexistGain * lod * sin(f.trapLine * 20.0f + f.smoothIter * 0.5f);
    color += lawBeat * float3(0.5f, 0.3f, 0.7f);
    return max(color, 0.0f);
}

// ---------------------------------------------------------------------------
// Relief view.
//
// The terrain is the depth of the full evaluation — the same one that colours
// the surface, so geometry and colour agree at every scale. Depth is mapped
// through a saturating curve rather than cut off: the interior asymptotes to
// the full height instead of forming a plateau with cliffs, and the deep
// boundary structure survives as fine relief of small amplitude. The orbit
// traps add ridges everywhere, including across the escape plains, so no
// part of the ground is featureless.
// ---------------------------------------------------------------------------
// The distance estimate is first-order and, on long chaotic hybrid orbits,
// can overshoot badly (the derivative stops growing). Long orbits are close
// to the boundary by the nature of escape and convergence, so the estimate is
// bounded by a smooth function of the orbit length: orbits longer than about
// twenty steps sit on the plateau, short ones keep the accurate estimate.
inline float mnmBoundaryDistance(MNMField f, int iters, float wMax)
{
    return max(f.boundary, 0.0f);
}

inline float mnmReliefFromField(MNMField f, int iters, constant ForgeSceneParams &sp)
{
    // Linear in the boundary distance, clamped at the ramp width: the height
    // a feature contributes is proportional to its size, so the dense
    // boundary regions become rough plateaus whose bumps carry every
    // generation of structure at its own amplitude, while the wide basins
    // read as deep valleys. A logarithmic map would give every filament the
    // full height and turn the plateaus into fur.
    float wMax = max(sp.mnReliefDetail, 1e-3f);
    float d = mnmBoundaryDistance(f, iters, wMax);
    float ramp = saturate(1.0f - d / wMax);
    // Soft valley floor so the ramps meet the basins without a crease.
    float h = sp.mnReliefHeight * (ramp * ramp * (1.5f - 0.5f * ramp));
    // Filament ridges from the orbit traps and the laws' own contours as
    // gentle ripples (escape bands on the Mandelbrot side, convergence rings
    // in the basins): this is what keeps a plain or a basin floor from being
    // featureless. Both fade out where the orbit is long — there the trap
    // and count fields vary far below a pixel (their gradients grow with the
    // orbit's derivative) and would only add spikes to the lace.
    float count = f.escaped > 0.5f ? f.smoothIter : (f.converged > 0.5f ? f.convIter : float(iters));
    float calm = 1.0f - saturate((count - 3.0f) / 7.0f);
    calm *= calm * (1.0f - ramp);
    float ridges = 0.7f * exp(-f.trapLine * 5.0f) + 0.3f * exp(-sqrt(max(f.trap, 0.0f)) * 3.0f);
    h += sp.mnReliefFine * ridges * calm;

    float nu = f.escaped > 0.5f ? f.smoothIter * 0.35f : f.convIter * 0.7f;
    float terminated = max(f.escaped, f.converged);
    h += sp.mnBandRelief * terminated * (0.5f + 0.5f * sin(nu * sp.mnBandDensity)) * ramp * calm;
    return h;
}

inline float mnmSampleRelief(float2 xz, constant ForgeSceneParams &sp, int iters, thread MNMField &f)
{
    float morph = saturate(sp.mnBalance);
    float2 p = mnmPlanePoint(xz * sp.mnWorldScale, sp);
    f = mnmEvaluate(p, morph, iters);
    return mnmReliefFromField(f, iters, sp);
}

inline float mnmHeightOnly(float2 xz, constant ForgeSceneParams &sp, int iters)
{
    MNMField f;
    return mnmSampleRelief(xz, sp, iters, f);
}

// Height filtered over a footprint: four taps on a rotated grid of radius
// `radius` (world units). This is the terrain's equivalent of resolving a
// surface to the pixel and no finer — structure below the footprint is
// averaged into the ground instead of becoming spikes. Zero radius or the
// fast preview uses the single centre sample.
inline float mnmHeightFiltered(float2 xz, constant ForgeSceneParams &sp, int iters, float radius, int taps)
{
    if (taps < 4 || radius <= 0.0f) { return mnmHeightOnly(xz, sp, iters); }
    const float2 o0 = float2( 0.38f,  0.92f);
    const float2 o1 = float2(-0.92f,  0.38f);
    float h = mnmHeightOnly(xz + o0 * radius, sp, iters)
            + mnmHeightOnly(xz - o0 * radius, sp, iters)
            + mnmHeightOnly(xz + o1 * radius, sp, iters)
            + mnmHeightOnly(xz - o1 * radius, sp, iters);
    return h * 0.25f;
}

inline int mnmFilterTaps(constant ForgeQuality &q)
{
    return q.raymarchSteps >= 350 ? 4 : 1;
}

inline float mnmMaxHeight(constant ForgeSceneParams &sp)
{
    return sp.mnReliefHeight + max(sp.mnReliefFine, 0.0f) + max(sp.mnBandRelief, 0.0f) + 1e-3f;
}

// Bound on |dh/dx| in world units: the log ramp's steepest slope (at the
// boundary, 1.5 H / (wMin log(1 + wMax/wMin)) per plane unit) times plane
// units per world unit, plus an allowance for the trap ridges.
inline float mnmMaxSlope(constant ForgeSceneParams &sp)
{
    float wMax = max(sp.mnReliefDetail, 1e-3f);
    float morph = saturate(sp.mnBalance);
    float planePerWorld = sp.mnWorldScale * mix(1.35f, 2.4f, morph) * sp.mnZoom;
    float rampSlope = 1.5f * sp.mnReliefHeight / wMax;
    float ridgeSlope = 6.0f * sp.mnReliefFine + 4.0f * sp.mnBandRelief * sp.mnBandDensity;
    return (rampSlope + ridgeSlope) * planePerWorld;
}

// ---------------------------------------------------------------------------
// Flat view: the reference image, supersampled through Forge's accumulation.
// ---------------------------------------------------------------------------
inline float3 mnmFlatSample(constant ForgeUniforms &u, float2 pixel)
{
    float2 uv = (pixel * 2.0f - u.imageSize) / u.imageSize.y;
    uv.y = -uv.y;
    float morph = saturate(u.scene.mnBalance);
    float2 p = mnmPlanePoint(uv, u.scene);
    int iters = mnmIterations(u.quality);
    MNMField f = mnmEvaluate(p, morph, iters);
    if (u.scene.mnReserved1 > 0.5f) {
        // Diagnostic: red = relief height / max, green = orbit length / 40,
        // blue = 1 for non-terminated orbits.
        float h = mnmReliefFromField(f, iters, u.scene) / mnmMaxHeight(u.scene);
        float bad = (isnan(h) || isinf(h) || isnan(f.boundary) || isinf(f.boundary)) ? 1.0f : 0.0f;
        float count = f.escaped > 0.5f ? f.smoothIter : (f.converged > 0.5f ? f.convIter : float(iters));
        if (u.scene.mnReserved1 > 1.5f) {
            // Thresholded height so the tone map cannot hide it.
            return float3(h > 0.95f ? 1.0f : 0.0f, h > 0.6f ? 1.0f : 0.0f, h > 0.3f ? 1.0f : 0.0f);
        }
        return float3(bad > 0.5f ? 0.0f : saturate(h), bad > 0.5f ? 0.0f : saturate(count / 40.0f), bad);
    }
    float3 color = mnmColour(f, morph, iters, u.scene, 1.0f);
    color *= 1.0f - 0.35f * dot(uv, uv) * 0.4f;
    return color;
}

struct MNMHit {
    int hit;
    float t;
    float3 position;
    MNMField field;
};

inline MNMHit mnmMarchRelief(float3 origin, float3 direction, constant ForgeSceneParams &sp,
                             constant ForgeQuality &q, float pixelRadius, int iters, float tStart)
{
    MNMHit h;
    h.hit = 0; h.t = tStart; h.position = origin;
    float maxH = mnmMaxHeight(sp);
    float farLimit = min(q.maxDistance, 80.0f);
    float slope = mnmMaxSlope(sp);
    int taps = mnmFilterTaps(q);
    float t = tStart;
    if (origin.y > maxH && direction.y < 0.0f) { t = max(t, (origin.y - maxH) / -direction.y); }
    float previousT = t;
    for (int i = 0; i < q.raymarchSteps; ++i) {
        float3 p = origin + direction * t;
        if (p.y > maxH && direction.y >= 0.0f) { break; }
        if (t > farLimit) { break; }
        float radius = t * pixelRadius * 0.75f;
        float delta = p.y - mnmHeightFiltered(p.xz, sp, iters, radius, taps);
        if (delta < 0.0f) {
            float lo = previousT, hi = t;
            for (int k = 0; k < 10; ++k) {
                float mid = 0.5f * (lo + hi);
                float3 pm = origin + direction * mid;
                if (pm.y - mnmHeightFiltered(pm.xz, sp, iters, mid * pixelRadius * 0.75f, taps) < 0.0f) { hi = mid; } else { lo = mid; }
            }
            h.hit = 1;
            h.t = 0.5f * (lo + hi);
            h.position = origin + direction * h.t;
            mnmSampleRelief(h.position.xz, sp, iters, h.field);
            return h;
        }
        previousT = t;
        // Clearance can be spent partly horizontally because the slopes are
        // bounded, but the fine relief is steep, so most of it is kept.
        float lipschitz = abs(direction.y) + slope * length(direction.xz);
        float stepLength = max(delta * 0.9f / lipschitz, t * pixelRadius * q.detailScale * 0.5f);
        stepLength = max(stepLength, 0.0005f);
        t += stepLength;
    }
    float3 last = origin + direction * t;
    if (t <= farLimit && last.y <= maxH && direction.y < 0.0f) {
        h.hit = 1;
        h.t = t;
        h.position = last;
        mnmSampleRelief(last.xz, sp, iters, h.field);
        return h;
    }
    h.t = t;
    h.position = last;
    return h;
}

inline float3 mnmReliefNormal(float2 xz, constant ForgeSceneParams &sp, int iters, float eps, float radius, int taps)
{
    float hx1 = mnmHeightFiltered(xz + float2(eps, 0.0f), sp, iters, radius, taps);
    float hx0 = mnmHeightFiltered(xz - float2(eps, 0.0f), sp, iters, radius, taps);
    float hz1 = mnmHeightFiltered(xz + float2(0.0f, eps), sp, iters, radius, taps);
    float hz0 = mnmHeightFiltered(xz - float2(0.0f, eps), sp, iters, radius, taps);
    return normalize(float3(hx0 - hx1, 2.0f * eps, hz0 - hz1));
}

inline float mnmReliefShadow(float3 origin, float3 lightDir, constant ForgeSceneParams &sp,
                             constant ForgeQuality &q, int iters, float softness, int steps)
{
    float maxH = mnmMaxHeight(sp);
    float res = 1.0f;
    float t = 0.01f;
    for (int i = 0; i < steps; ++i) {
        float3 p = origin + lightDir * t;
        if (p.y > maxH) { break; }
        float delta = p.y - mnmHeightOnly(p.xz, sp, iters);
        res = min(res, softness * delta / t);
        if (res < 0.002f) { break; }
        t += clamp(delta * 0.8f, 0.003f, 0.10f);
    }
    return saturate(res);
}

inline float mnmReliefAO(float3 position, float3 normal, constant ForgeSceneParams &sp,
                         constant ForgeQuality &q, int iters)
{
    float occlusion = 0.0f;
    float scale = 1.0f;
    int n = max(q.aoSamples, 1);
    for (int i = 0; i < n; ++i) {
        float h = 0.008f + 0.16f * float(i) / float(n);
        float3 p = position + normal * h;
        float delta = p.y - mnmHeightOnly(p.xz, sp, iters);
        occlusion += max(h - delta, 0.0f) * scale;
        scale *= 0.85f;
    }
    return saturate(1.0f - 2.0f * occlusion);
}

// Surface appearance: the reference colouring as albedo, with the boundary
// emphasis held to a rim instead of a white-out, glossier in the basins than
// on the escape plains, and the filaments carried as emission.
inline ForgeMaterial mnmMaterial(MNMField f, float morph, int iters, float lod, float steep, constant ForgeSceneParams &sp)
{
    float wMax = max(sp.mnReliefDetail, 1e-3f);
    float depth = saturate(1.0f - mnmBoundaryDistance(f, iters, wMax) / wMax);
    float nu = f.escaped > 0.5f ? f.smoothIter : (f.converged > 0.5f ? f.convIter : float(iters));

    // Near the camera and at low iteration counts the reference colouring is
    // resolved: escape bands, basin shading, filaments. Deep in the boundary
    // regions its bands are far below a pixel and average to grey, so there
    // the colour follows the terrain instead — the same palettes, driven by
    // the boundary distance, so height and hue agree.
    float3 reference = mnmColour(f, morph, iters, sp, lod * 0.5f);
    float dense = saturate((nu - 10.0f) / 28.0f);
    float t = depth * 0.45f + sp.mnPaletteDrift + (f.converged > 0.5f ? f.rootId / 3.0f + 0.08f * morph : 0.0f);
    float3 terrain = f.converged > 0.5f ? mnmPalB(t) : mnmPalA(t + 0.3f);
    terrain *= mix(0.45f, 0.8f, depth);
    float interior = 1.0f - max(f.escaped, f.converged);
    float3 colour = mix(reference, terrain, dense * (1.0f - 0.25f * lod) * (1.0f - interior));
    // Steep faces: the plane colouring stretches vertically there, so fade
    // to the terrain palette by slope.
    colour = mix(colour, terrain, steep);

    ForgeMaterial m;
    m.albedo = clamp(colour, 0.01f, 1.0f);
    float basin = f.converged;
    m.roughness = clamp(sp.roughnessBase * mix(mix(1.1f, 0.5f, basin), 1.6f, depth * dense), 0.03f, 1.0f);
    m.metallic = clamp(sp.metallicBase * mix(0.4f, 1.0f, basin) * (1.0f - 0.6f * dense), 0.0f, 1.0f);
    float filament = exp(-f.trapLine * 6.0f) * lod * (1.0f - dense);
    float rim = smoothstep(0.75f, 1.0f, depth) * (1.0f - 0.7f * dense);
    m.emissive = sp.emissiveStrength * (m.albedo * (0.5f * filament + 0.2f * rim)
                                        + float3(0.4f, 0.7f, 1.0f) * 0.12f * rim * sp.mnBoundaryGlow);
    return m;
}

// Participating medium over the relief: fog pools in the valleys and the key
// light carves shafts through it. Density reads a reduced-iteration height —
// the large forms are all it needs — and each sample probes the light.
inline ForgeVolumeResult mnmMarchVolume(float3 origin, float3 direction, float tEnd,
                                        constant ForgeSceneParams &sp, constant ForgeQuality &q,
                                        unsigned int rngSeed)
{
    ForgeVolumeResult result;
    result.inscatter = float3(0.0f);
    result.transmittance = float3(1.0f);
    int steps = q.volumetricSamples;
    if (steps <= 0 || sp.fogDensity <= 1e-5f) {
        float ext = 1.0f - exp(-sp.fogDensity * tEnd);
        result.transmittance = float3(1.0f - ext);
        result.inscatter = sp.fogColor * (sp.ambientIntensity * 1.7f) * ext;
        return result;
    }
    int coarse = 24;
    float far = min(tEnd, 8.0f);
    float stepLength = far / float(steps);
    float3 lightDir = normalize(sp.keyLightDirection);
    float phase = forgeHenyeyGreenstein(dot(direction, lightDir), 0.5f);
    float3 extinctionTint = float3(1.25f, 1.05f, 0.85f);
    float maxH = mnmMaxHeight(sp);
    for (int i = 0; i < steps; ++i) {
        float jitter = forgeRandom(forgeHashCombine(rngSeed, (unsigned int)(i + 1)));
        float t = (float(i) + jitter) * stepLength;
        float3 p = origin + direction * t;
        float clearance = p.y - (p.y < maxH + 0.6f ? mnmHeightOnly(p.xz, sp, coarse) : maxH);
        float proximity = exp(-max(clearance, 0.0f) * 5.0f);
        float density = sp.fogDensity * (1.0f + sp.fogHeightFalloff * proximity);
        float3 stepExtinction = exp(-density * extinctionTint * stepLength);
        float3 lit = sp.fogColor * (sp.ambientIntensity * 1.7f);
        if (sp.godRayStrength > 1e-4f && q.shadowSteps > 0) {
            float shadow = mnmReliefShadow(p, lightDir, sp, q, coarse, 8.0f, min(q.shadowSteps, 24));
            lit += sp.keyLightColor * sp.keyLightIntensity * shadow * phase * sp.godRayStrength;
        }
        const float scatteringAlbedo = 0.82f;
        float3 segment = (float3(1.0f) - stepExtinction) * lit * scatteringAlbedo;
        result.inscatter += result.transmittance * segment;
        result.transmittance *= stepExtinction;
        if (forgeLuminance(result.transmittance) < 0.003f) { break; }
    }
    // Beyond the marched range the analytic extinction continues.
    if (tEnd > far) {
        float ext = 1.0f - exp(-sp.fogDensity * (tEnd - far));
        result.inscatter += result.transmittance * sp.fogColor * (sp.ambientIntensity * 1.7f) * ext;
        result.transmittance *= (1.0f - ext);
    }
    return result;
}

inline float3 mnmSky(float3 direction, constant ForgeSceneParams &sp)
{
    float3 sky = forgeSkyRadiance(direction, sp);
    // Horizon glow: the atmosphere thickens toward the ground plane.
    float horizon = pow(1.0f - saturate(abs(direction.y)), 6.0f);
    return sky + sp.fogColor * (0.8f * horizon * sp.ambientIntensity);
}

inline float3 mnmReliefTrace(ForgeRay ray, constant ForgeSceneParams &sp, constant ForgeQuality &q,
                             float pixelRadius, unsigned int rngSeed)
{
    int iters = mnmIterations(q);
    float morph = saturate(sp.mnBalance);
    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);
    int maxBounces = max(q.reflectionBounces, 0) + 1;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        MNMHit hit = mnmMarchRelief(ray.origin, ray.direction, sp, q, pixelRadius, iters,
                                    bounce == 0 ? 0.0f : 0.002f);
        float segmentEnd = hit.hit != 0 ? hit.t : q.maxDistance;

        ForgeVolumeResult volume;
        if (bounce == 0) {
            volume = mnmMarchVolume(ray.origin, ray.direction, segmentEnd, sp, q, rngSeed);
        } else {
            float ext = 1.0f - exp(-sp.fogDensity * segmentEnd * 1.5f);
            volume.transmittance = float3(1.0f - ext);
            volume.inscatter = sp.fogColor * (sp.ambientIntensity * 1.7f) * ext;
        }

        float3 surface;
        int continueTrace = 0;
        float3 nextOrigin = float3(0.0f), nextDirection = float3(0.0f);
        if (hit.hit != 0) {
            float eps = max(hit.t * pixelRadius * q.detailScale, 2e-5f);
            float3 normal = mnmReliefNormal(hit.position.xz, sp, iters, eps,
                                            hit.t * pixelRadius * 0.75f, mnmFilterTaps(q));
            float3 view = -ray.direction;
            float footprint = hit.t * pixelRadius * sp.mnWorldScale * mix(1.35f, 2.4f, morph) * sp.mnZoom
                            / max(abs(dot(normal, view)), 0.15f);
            float lod = saturate(1.0f - footprint * 40.0f * sp.mnBandDensity);
            float steep = smoothstep(0.75f, 0.35f, normal.y);
            ForgeMaterial m = mnmMaterial(hit.field, morph, iters, lod, steep, sp);
            float3 lightDir = normalize(sp.keyLightDirection);
            float ao = mnmReliefAO(hit.position, normal, sp, q, iters);
            float shadow = 1.0f;
            if (q.shadowSteps > 0 && dot(normal, lightDir) > 0.0f) {
                shadow = mnmReliefShadow(hit.position + normal * 0.002f, lightDir, sp, q, iters,
                                         max(q.shadowSoftness, 1.0f), q.shadowSteps);
            }
            surface = forgeShadeSurfaceLit(normal, view, hit.t, m, ao, shadow, sp);

            if (bounce + 1 < maxBounces) {
                float3 reflected = ray.direction - normal * (2.0f * dot(ray.direction, normal));
                nextDirection = normalize(mix(reflected, normal, m.roughness * 0.45f));
                nextOrigin = hit.position + normal * (eps * 8.0f + 1e-4f);
                float fresnel = 0.04f + 0.96f * pow(1.0f - saturate(dot(normal, view)), 5.0f);
                float reflectance = saturate(mix(fresnel, 1.0f, m.metallic) * (1.0f - m.roughness * 0.8f));
                surface *= (1.0f - reflectance);
                throughput *= reflectance;
                continueTrace = 1;
            }
        } else {
            surface = mnmSky(ray.direction, sp);
        }

        float3 segmentRadiance = surface * volume.transmittance + volume.inscatter;
        if (bounce == 0) {
            segmentRadiance = forgeApplyAtmosphericDepth(segmentRadiance, segmentEnd, sp);
            radiance += segmentRadiance;
        } else {
            radiance += throughput * segmentRadiance;
        }
        if (continueTrace == 0) { break; }
        ray.origin = nextOrigin;
        ray.direction = nextDirection;
    }
    return radiance;
}

inline float3 mnmRenderPixel(constant ForgeUniforms &u, unsigned int px, unsigned int py)
{
    float pixelRadius = forgePixelRadius(u);
    unsigned int pixelSeed = forgeHashCombine(forgeHashCombine(u.seed, u.frameIndex),
                                              px * 73856093u ^ py * 19349663u);
    int spp = max(u.quality.samplesPerPixel, 1);
    float3 sum = float3(0.0f);
    for (int s = 0; s < spp; ++s) {
        unsigned int globalSample = (unsigned int)(u.quality.sampleOffset + s);
        float2 jitter = forgeHalton(globalSample, pixelSeed);
        float2 pixel = float2(float(px) + jitter.x, float(py) + jitter.y);
        if (u.scene.mnReliefMode > 0.5f) {
            float2 lens = forgeHalton(globalSample, forgeHashCombine(pixelSeed, 0x5bd1e995u));
            ForgeRay ray = forgeGenerateRay(u, pixel, lens);
            sum += mnmReliefTrace(ray, u.scene, u.quality, pixelRadius,
                                  forgeHashCombine(pixelSeed, globalSample + 1u));
        } else {
            sum += mnmFlatSample(u, pixel);
        }
    }
    return sum / float(spp);
}

kernel void mandelnewtonRenderKernel(device float4 *accumulation [[buffer(ForgeBufferIndexAccumulation)]],
                                     constant ForgeUniforms &u [[buffer(ForgeBufferIndexUniforms)]],
                                     uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uint(u.tileSize.x) || tid.y >= uint(u.tileSize.y)) { return; }
    uint2 pixel = uint2(uint(u.tileOrigin.x) + tid.x, uint(u.tileOrigin.y) + tid.y);
    if (pixel.x >= uint(u.imageSize.x) || pixel.y >= uint(u.imageSize.y)) { return; }
    float3 radiance = mnmRenderPixel(u, pixel.x, pixel.y);
    float weight = float(max(u.quality.samplesPerPixel, 1));
    uint index = pixel.y * uint(u.imageSize.x) + pixel.x;
    float4 prior = u.accumulate != 0 ? accumulation[index] : float4(0.0f);
    accumulation[index] = float4(prior.xyz + radiance * weight, prior.w + weight);
}

#endif /* ForgeMandelNewton_h */
