//
//  ForgeApollonian.h
//  Cyanescent Forge — shared shading module
//
//  The geometry of the Apollonian World.
//
//  Mathematical intention
//  ----------------------
//  An Apollonian gasket is what you get by repeatedly inverting space in a
//  sphere and packing the result. The distance estimator below is the
//  three-dimensional Kleinian form popularised by Inigo Quilez:
//
//      p  <-  -1 + 2 * fract(0.5 * p + 0.5)     // fold into the unit cell
//      p  <-  p * (s / |p|^2)                   // invert in a sphere
//
//  Two properties make it the right choice for Forge:
//
//  1. The fold is periodic, so the structure fills all of space. The camera is
//     never looking at "an object against a black background" — it is always
//     *inside* the world, with more world in every direction.
//
//  2. The inversion blows up the neighbourhood of the inversion centre. Points
//     that were an indistinguishable speck at one recursion level become a
//     full-size structure at the next. That is literally the brief's target:
//     detail that looked like surface texture resolving into whole nested
//     worlds as you approach.
//
//  The estimator returns a lower bound on the true distance, which is what
//  sphere tracing requires. Raising `iterations` adds finer generations of
//  structure; it never changes the large shapes. This is why the preview and
//  the offline render are recognisably the same image, and why the offline
//  render simply contains more of it.
//
#ifndef ForgeApollonian_h
#define ForgeApollonian_h

#include "ForgeShaderTypes.h"

#include "ForgeMath.h"

// Result of one field evaluation. The orbit traps are the cheapest source of
// coherent surface variation we have: they record how the point moved through
// the recursion, so two points on the same nested structure get similar values
// and points on different structures get different ones.
struct ForgeFieldSample {
    float  distance;
    float3 trap;        // per-axis closest approach over the whole orbit
    float  minRadius2;  // closest approach to the inversion centre
    float  logScale;    // log2 of accumulated scale => which recursion level
};

// One layer of the Apollonian recursion.
inline ForgeFieldSample forgeApollonianLayer(float3 p,
                                             float s,
                                             int iterations,
                                             float twist,
                                             float cavityBias)
{
    float scale = 1.0f;
    float3 trap = float3(1e10f);
    float minR2 = 1e10f;
    // `twist` is fixed for this entire field evaluation. The handedness below
    // only changes its sign, so calculate the transcendental pair once rather
    // than once per recursive level.
    float twistCos = cos(twist);
    float twistSin = sin(twist);

    for (int i = 0; i < iterations; ++i) {
        // Fold into [-1,1]^3. Infinite world in one line.
        p = -1.0f + 2.0f * fract(0.5f * p + 0.5f);

        // A small twist per level.  The repeated cell maps opposite sides of
        // an X or Z boundary to reflected local coordinates.  A fixed Y
        // rotation does not respect that reflection, so its two sides entered
        // incompatible recursive frames and broke an arch at the boundary.
        // Reversing the twist's handedness in alternating X/Z quadrants makes
        // the transform reflection-equivariant: T(reflect(p)) = reflect(T(p)).
        // The following folds, inversion and |y| distance then preserve that
        // equality, making the field continuous without changing its period,
        // scale compensation or recursive construction.
        float twistHandedness = p.x * p.z < 0.0f ? -1.0f : 1.0f;
        float signedTwistSin = twistHandedness * twistSin;
        p = float3(twistCos * p.x + signedTwistSin * p.z,
                   p.y,
                   -signedTwistSin * p.x + twistCos * p.z);

        float r2 = dot(p, p);

        trap = min(trap, abs(p));
        minR2 = min(minR2, r2);

        // Sphere inversion. Clamped denominator so a ray that passes exactly
        // through the inversion centre produces a large-but-finite step rather
        // than a NaN that would punch a black hole in the frame.
        float k = s / max(r2, 1e-4f);
        p = p * k;
        scale = scale * k;
    }

    ForgeFieldSample r;
    // Distance to the plane y = 0 measured in the deepest folded frame and
    // pulled back through the accumulated scale. The 0.25 is the conservative
    // Lipschitz fudge that keeps this a lower bound.
    r.distance   = (0.25f * abs(p.y) - cavityBias) / scale;
    r.trap       = trap;
    r.minRadius2 = minR2;
    r.logScale   = log2(max(scale, 1e-8f));
    return r;
}

// The full world: a fine layer the camera moves through, unioned with a much
// larger copy of the same construction. The macro layer is what gives the
// image its cathedral-scale silhouettes and stops every shot from reading at
// the same spatial frequency.
inline ForgeFieldSample forgeApollonianWorld(float3 p, constant ForgeSceneParams &sp, int iterations, int macroIterations)
{
    // worldScale sets how big the world is around the camera. The recursion is
    // scale-free, so shrinking the coordinates and expanding the result gives a
    // genuinely larger universe rather than a magnified picture of a small one.
    float ws = max(sp.worldScale, 1e-4f);

    ForgeFieldSample fine = forgeApollonianLayer(p * ws,
                                                 sp.foldScale,
                                                 iterations,
                                                 sp.levelTwist,
                                                 sp.cavityBias);
    fine.distance = fine.distance / ws;

    if (sp.macroWeight <= 0.001f || macroIterations <= 0) {
        return fine;
    }

    // Evaluating the same field in shrunken coordinates produces the same
    // structure at 1/macroScale the size; dividing the result restores true
    // world distance, so the union remains a valid distance bound.
    float ms = max(sp.macroScale, 1e-3f) * ws;
    ForgeFieldSample macro = forgeApollonianLayer(p * ms,
                                                  sp.macroFoldScale,
                                                  macroIterations,
                                                  sp.levelTwist * 0.5f,
                                                  sp.cavityBias * 0.5f);
    macro.distance = macro.distance / ms;

    // macroWeight fades the layer in by pushing it away rather than by
    // blending distances, which would create a surface that is neither.
    macro.distance = macro.distance + (1.0f - sp.macroWeight) * 1e6f;

    if (macro.distance < fine.distance) {
        return macro;
    }
    return fine;
}

// Scalar convenience wrapper for the marchers that do not need orbit traps.
inline float forgeApollonianDistance(float3 p, constant ForgeSceneParams &sp, int iterations, int macroIterations)
{
    return forgeApollonianWorld(p, sp, iterations, macroIterations).distance;
}

// Surface normal by central differences. `eps` should scale with the hit
// distance: too small and the normal is quantisation noise, too large and fine
// recursion detail is smoothed away. The offline renderer uses a much smaller
// epsilon than preview, which is a large part of why offline frames look
// crisper rather than merely bigger.
inline float3 forgeApollonianNormal(float3 p, constant ForgeSceneParams &sp, int iterations, int macroIterations, float eps)
{
    float2 e = float2(1.0f, -1.0f) * eps;
    float3 n =
        float3( e.x, e.y, e.y) * forgeApollonianDistance(p + float3( e.x, e.y, e.y), sp, iterations, macroIterations) +
        float3( e.y, e.y, e.x) * forgeApollonianDistance(p + float3( e.y, e.y, e.x), sp, iterations, macroIterations) +
        float3( e.y, e.x, e.y) * forgeApollonianDistance(p + float3( e.y, e.x, e.y), sp, iterations, macroIterations) +
        float3( e.x, e.x, e.x) * forgeApollonianDistance(p + float3( e.x, e.x, e.x), sp, iterations, macroIterations);
    return normalize(n);
}

#endif /* ForgeApollonian_h */
