//
//  ForgeShaderTypes.h
//  Cyanescent Forge
//
//  The single shared definition of everything that crosses the CPU/GPU
//  boundary. This file is compiled three times:
//
//    1. by the Metal compiler, as part of ApollonianWorld.metal
//    2. by the Swift compiler, through ForgeBridging.h
//    3. by g++, as part of the CPU reference harness in Tools/CPUReference
//
//  Because all three see the identical declaration there is no possibility of
//  the uniform struct layout drifting between Swift and MSL — historically the
//  single most common cause of "it renders garbage" in Metal projects.
//
//  Layout discipline: fields are grouped so that every 16-byte-aligned vector
//  starts on a 16-byte boundary, and scalar runs are padded to multiples of
//  four floats. Keep it that way when adding fields.
//
#ifndef ForgeShaderTypes_h
#define ForgeShaderTypes_h

#if defined(__METAL_VERSION__)
    #include <metal_stdlib>
    // Pulled in here rather than in the .metal file so every shading module is
    // self-contained and can be included in any order.
    using namespace metal;
    typedef float2 ForgeVec2;
    typedef float3 ForgeVec3;
    typedef float4 ForgeVec4;
    typedef int   ForgeInt;
    typedef unsigned int ForgeUInt;
#elif defined(FORGE_CPU_REFERENCE)
    typedef float2 ForgeVec2;
    typedef float3 ForgeVec3;
    typedef float4 ForgeVec4;
    typedef int ForgeInt;
    typedef unsigned int ForgeUInt;
#else
    #include <stdint.h>
    #include <simd/simd.h>
    typedef simd_float2 ForgeVec2;
    typedef simd_float3 ForgeVec3;
    typedef simd_float4 ForgeVec4;
    typedef int32_t  ForgeInt;
    typedef uint32_t ForgeUInt;
#endif

// Buffer indices shared by the Swift renderer and the Metal kernel.
#define ForgeBufferIndexUniforms 0
#define ForgeBufferIndexAccumulation 1
#define ForgeTextureIndexWorldField 0

// ---------------------------------------------------------------------------
// ForgeQuality
//
// Everything that trades render time for image quality. Preview presets use
// small numbers here; the offline renderer is free to use numbers that would
// be absurd in a realtime engine. Nothing in this struct changes *what* the
// world is — only how accurately it is resolved — so a preview and an offline
// render of the same frame are the same image at different fidelities.
// ---------------------------------------------------------------------------
typedef struct {
    ForgeInt raymarchSteps;        // primary ray march iteration limit
    ForgeInt fractalIterations;    // Apollonian recursion depth
    ForgeInt shadowSteps;          // soft shadow march iterations
    ForgeInt aoSamples;            // ambient occlusion taps

    ForgeInt volumetricSamples;    // participating-medium march steps
    ForgeInt reflectionBounces;    // 0 = none, 1 = single specular bounce
    ForgeInt samplesPerPixel;      // samples accumulated for THIS dispatch
    ForgeInt totalSamples;         // samples accumulated across all dispatches

    ForgeInt sampleOffset;         // index of the first sample in this dispatch
    ForgeInt macroLayerIterations; // recursion depth of the large-scale layer
    ForgeInt enableDepthOfField;   // 0/1
    ForgeInt previewMode;          // explicit interactive request; never inferred from offline quality

    float detailScale;             // multiplies the adaptive surface epsilon
                                   // (<1 = finer detail, much slower)
    float maxDistance;             // ray march far plane, in world units
    float shadowSoftness;
    float _padQ1;
} ForgeQuality;

// ---------------------------------------------------------------------------
// ForgeSceneParams
//
// The authorable description of the Apollonian world at one instant. Every
// field here is animatable from the timeline. None of it is quality-related:
// changing these changes the picture, changing ForgeQuality does not.
// ---------------------------------------------------------------------------
typedef struct {
    float foldScale;          // the Apollonian inversion radius; the single
                              // most expressive parameter in the whole world
    float macroFoldScale;     // same, for the large-scale layer
    float macroScale;         // size ratio of the large-scale layer
    float macroWeight;        // 0 = macro layer off, 1 = fully present

    float levelTwist;         // radians of rotation applied per recursion level
    float cavityBias;         // pushes the surface inwards, opening cavities
    float surfaceDetail;      // amplitude of fine high-frequency displacement
    float paletteShift;       // rotates the colour palette

    float paletteSpread;      // how far apart the palette bands sit
    float emissiveStrength;   // glow radiating out of the deep recursion
    float roughnessBase;
    float metallicBase;

    float fogDensity;         // atmospheric extinction per world unit
    float fogHeightFalloff;
    float godRayStrength;     // inscattering along shadow rays
    float exposure;

    // Atmospheric Depth is deliberately a post-lighting perspective effect,
    // not another volumetric march.  It consumes the primary ray's already
    // known distance and therefore adds no field samples.
    float atmosphericDepthEnabled;
    float atmosphericDepthDensity;
    float atmosphericDepthStart;
    float atmosphericDepthMaximumStrength;

    float worldScale;         // size of the camera relative to the structure.
                              // Small values make the world enormous. Animating
                              // it is a continuous zoom through scales, which is
                              // the most distinctive move the geometry affords.
    float glowStrength;       // light bleeding out of geometry the ray passed
    float headlightIntensity; // camera-relative lamp; see ForgeLighting.h
    float bounceIntensity;    // cheap one-bounce colour bleed

    ForgeVec3 fogColor;
    ForgeVec3 atmosphericDepthColor;
    ForgeVec3 emissiveColor;
    ForgeVec3 keyLightColor;
    ForgeVec3 fillLightColor;
    ForgeVec3 skyColorZenith;
    ForgeVec3 skyColorHorizon;
    ForgeVec3 keyLightDirection;  // normalised, pointing *towards* the light

    float keyLightIntensity;
    float ambientIntensity;
    float apertureRadius;         // depth of field: lens radius, world units
    float focusDistance;

    // Morphogenesis — Gray-Scott reaction/diffusion parameters.  They live in
    // the shared block so a world runtime can evolve its volume without a CPU
    // readback, while the normal timeline/automation path remains unchanged.
    float morphFeed;
    float morphKill;
    float morphDiffusionA;
    float morphDiffusionB;

    float morphStepsPerBeat;
    float morphSurfaceThreshold;
    float morphSeedDensity;
    float morphWarp;

    float morphWarpScale;
    float morphSimulationScale;
    float morphMaterialDetail;
    float morphSeedMode;

    // Hypercrystal — a 4D cut-and-project lattice.  XW/YW/ZW are true 4D
    // rotation planes; they are not aliases for an ordinary 3D rotation.
    float hyperLatticeScale;
    float hyperAcceptanceWindow;
    float hyperCrystalThreshold;
    float hyperConnectionRadius;

    float hyperDensity;
    float hyperProjectionScale;
    float hyperXWRotation;
    float hyperYWRotation;

    float hyperZWRotation;
    float hyperXYRotation;
    float hyperStructuralWarp;
    float hyperMaterialDetail;

    // Mandelbrot / Newton morph world.  The iteration blends the Mandelbrot
    // step and the Newton step for z^3 - 1 inside the loop; see
    // ForgeMandelNewton.h.
    float mnBalance;          // 0 = pure Mandelbrot, 1 = pure Newton
    float mnZoom;             // plane magnification
    float mnCenterX;          // Mandelbrot seed neighbourhood (drifts to the Newton origin)
    float mnCenterY;

    float mnRotation;         // plane rotation, radians
    float mnWarp;             // plane-bending warp amplitude
    float mnBandDensity;      // colour banding density of the escape region
    float mnBoundaryGlow;     // emphasis of slow-converging basin boundaries

    float mnPaletteDrift;     // rotates both palettes
    float mnCoexistGain;      // strength of the interference between the two laws
    float mnReliefHeight;     // height of the relief; 0 renders the flat plane
    float mnReliefMode;       // 0 flat, 1 relief

    float mnWorldScale;       // camera scale in relief mode
    float mnTrapInfluence;    // orbit-trap glow strength
    float mnReliefSharpness;  // how steeply the relief rises toward the boundaries
    float mnReliefDetail;     // iteration cut-off that shapes the terrain

    float mnReliefFine;       // height of the orbit-trap filament ridges
    float mnBandRelief;       // height of the law-contour ripples
    float mnReserved1;
    float mnReserved2;
    // Duoverse: one dividing surface, two interlocked networks (luminous
    // glass and dark metal), and inside every network the same law again,
    // generation after generation. Shading: ForgeDuoverse.h; the CPU side
    // (inversion centre, camera voyage): DuoverseField.swift.
    float duoSeparation;      // how far apart the two networks stand
    float duoBifurcation;     // how visibly each generation splits in two
    float duoDepth;           // generations of doubling
    float duoHandedness;      // chirality balance: left, domains of both, right

    float duoFold;            // fold intensity: junctions morph toward four-way crossings
    float duoDensity;         // lattice cells per world radius
    float duoEvolution;       // generations sliding through their parents
    float duoWarp;            // topology warp: drift of the inversion centre

    float duoGlow;            // light in the seams and the core
    float duoIridescence;     // thin-film colour
    float duoMacroScale;      // size of the world: the inversion radius
    float duoCameraMode;      // 1 procedural voyage, 0 keyframed rig

    float duoCameraActivity;  // pace and banking of the voyage
    float duoDebugView;       // 1 flat shading by network and generation
    float duoReserved0;
    float duoReserved1;

    // MCPM: agent/field transport, independent of the other world engines.
    float mcpmPopulation;
    float mcpmAttractorCount;
    float mcpmExploration;
    float mcpmSensorDistance;
    float mcpmSensorAngle;
    float mcpmSteering;
    float mcpmDeposit;
    float mcpmDiffusion;
    float mcpmDecay;
    float mcpmReinforcement;
    float mcpmPruning;
    float mcpmClustering;
    float mcpmHubBias;
    float mcpmVoidScale;
    float mcpmWorldExtent;
    float mcpmEvolutionSpeed;
    float mcpmThickness;
    float mcpmBrightness;
    float mcpmFineDetail;
    float mcpmVolumeDensity;
    float mcpmDebugMode;
    float mcpmCameraMode;
    float mcpmCameraMovement;
    float mcpmInitialMaturity;

    // Apollonian Physarum Genesis.  The structural values drive the CPU
    // ecosystem simulation (ApollonianPhysarumEcosystem.swift); the rest are
    // read by ForgeApollonianPhysarum.h.
    float apGrowthActivity;
    float apColonyCount;
    float apNetworkDensity;
    float apFusionActivity;

    float apReproductionRate;
    float apCytoplasmicFlow;
    float apApollonianDepth;
    float apCellScale;

    float apDistortion;
    float apSurfaceDetail;
    float apGlow;
    float apCameraActivity;

    float apCameraDirector;
    float apDebugView;
    float apPlanetCurvature;
    float apReserved0;
} ForgeSceneParams;

// ---------------------------------------------------------------------------
// Apollonian Physarum Genesis — per-frame world data.
//
// The ecosystem is simulated on the CPU (deterministically, from the project
// seed) and each frame's interpolated snapshot is uploaded as these arrays,
// plus a 2D uniform grid over the periodic planet tile that lists which items
// touch each cell. Every field is a 16-byte vector so the layout is identical
// in Swift, MSL and C.
// ---------------------------------------------------------------------------
#define APBufferIndexHeader 2
#define APBufferIndexTubes 3
#define APBufferIndexGaskets 4
#define APBufferIndexSpores 5
#define APBufferIndexCells 6
#define APBufferIndexRefs 7

#define APRefTube 0u
#define APRefGasket 1u
#define APRefSpore 2u

typedef struct {
    ForgeVec4 domain;        // x: period L, y: cell size, z: 1 / cell size, w: grid cells per side
    ForgeVec4 terrainPhaseA; // terrain sinusoid phases 0..3
    ForgeVec4 terrainPhaseB; // terrain sinusoid phases 4..7
    ForgeVec4 time;          // x: beats, y: seconds, z: global pulse phase, w: layer top (max item height above ground)
    ForgeVec4 cameraAnchor;  // xz: camera ground position for planet curvature, z: planet radius, w: margin
    ForgeUInt tubeCount;
    ForgeUInt gasketCount;
    ForgeUInt sporeCount;
    ForgeUInt gridCells;     // cells per side
} APHeader;

// One vascular tube: two quadratic Bezier arcs on the ground joined with a
// continuous tangent at their midpoint (a quadratic B-spline), draped over
// the terrain.
typedef struct {
    ForgeVec4 ends;    // xz of the start, xz of the end (unwrapped relative to the start)
    ForgeVec4 ctrl;    // xz of the first control point, radius at start, radius at end
    ForgeVec4 ctrl2;   // xz of the second control point, width wobble phase, unused
    ForgeVec4 extent;  // visible parameter range [x, y], z: tip bulb (growing tendril), w: arc length
    ForgeVec4 flow;    // x: packet phase (world units), y: speed (units / beat, signed), z: activity, w: fusion energy
    ForgeVec4 color0;  // identity colour at the start, w: collapse (0 alive, 0..1 withering, 2 residue trail)
    ForgeVec4 color1;  // identity colour at the end, w: hash
} APTube;

// One Apollonian cluster: a hemisphere packing inside a disc on the ground.
typedef struct {
    ForgeVec4 disc;    // xz centre, territory radius, dome flattening
    ForgeVec4 mobius;  // disc automorphism parameter a (xy), rotation, cell gap
    ForgeVec4 growth;  // maturity (generations revealed), activity, breathing phase, kind
    ForgeVec4 color;   // identity colour, fusion energy
    ForgeVec4 sac;     // sporangium: shell radius factor, rupture 0..1, embryo glow, hash
} APGasket;

typedef struct {
    ForgeVec4 position; // xyz above the local ground, radius
    ForgeVec4 velocity; // xyz (units / beat), brightness
    ForgeVec4 color;    // rgb, trail length (beats)
} APSporeGPU;

typedef struct {
    ForgeUInt start;
    ForgeUInt count;
    float top;          // highest point of any listed item above the local ground
    float pad;
} APCell;

// ---------------------------------------------------------------------------
// Duoverse — per-frame world data.
//
// The world is a periodic field in "lattice space" mapped to the world by a
// sphere inversion: lattice infinity lands on the world centre, where the
// structure shrinks without end, and the lattice centre (on the dividing
// surface) becomes the sky beyond the colossal outer shell.
// ---------------------------------------------------------------------------
#define DuoBufferIndexHeader 2

typedef struct {
    ForgeVec4 inversion;  // xyz world centre, w: R^2
    ForgeVec4 lattice;    // xyz lattice-space centre, w: lattice frequency k
    ForgeVec4 axisX;      // lattice orientation (row 0), w: network threshold
    ForgeVec4 axisY;      // row 1, w: generation gap threshold
    ForgeVec4 axisZ;      // row 2, w: world radius of the outer shell
    ForgeVec4 time;       // x beats, y seconds, z evolution phase, w chirality bias
    ForgeVec4 centre;     // xyz gyroid flat point (domains and fold fade in away from it), w: fold
} DuoHeader;

typedef struct {
    ForgeUInt resolution;
    ForgeUInt agentCount;
    ForgeUInt attractorCount;
    ForgeUInt stepIndex;
    ForgeUInt seed;
    ForgeUInt warmup;
    ForgeUInt pad0;
    ForgeUInt pad1;
    float timeBeats;
    float timeSeconds;
    float deltaScale;
    float pad2;
    ForgeSceneParams scene;
} MCPMSimulationUniforms;

// ---------------------------------------------------------------------------
// Loop closure — the rendered-state safety net (ForgeLoopClosure.h).
// ---------------------------------------------------------------------------
typedef struct {
    ForgeUInt width;          // beauty image size in pixels
    ForgeUInt height;
    float weight;             // closure fallback weight, 0..1
    ForgeUInt seed;           // project seed: the stagger pattern is fixed per render
} ForgeLoopCompositeParams;

// ---------------------------------------------------------------------------
// ForgeUniforms
//
// One buffer, one dispatch. `tileOrigin`/`tileSize` describe the region of the
// output image this dispatch is responsible for, which is what makes tiled and
// progressive rendering possible without touching the kernel.
// ---------------------------------------------------------------------------
typedef struct {
    ForgeVec3 cameraPosition;
    ForgeVec3 cameraForward;
    ForgeVec3 cameraRight;
    ForgeVec3 cameraUp;

    ForgeVec2 imageSize;      // full output image size in pixels
    ForgeVec2 tileOrigin;     // pixel coordinate of this tile's top-left
    ForgeVec2 tileSize;       // this tile's size in pixels
    ForgeVec2 loopClosureTime; // loop closure: x = opening pre-roll beats (<= 0),
                               // y = weight of the opening's time; (0, 0) when off

    float tanHalfFov;
    float aspect;
    float timeSeconds;        // timeline position, NOT wall clock
    float timeBeats;

    ForgeUInt seed;           // project seed; all randomness derives from this
    ForgeUInt frameIndex;     // timeline frame number, for deterministic jitter
    ForgeInt  accumulate;     // 1 = add into the target, 0 = overwrite
    ForgeInt  _padU1;

    ForgeQuality quality;
    ForgeSceneParams scene;
} ForgeUniforms;

#endif /* ForgeShaderTypes_h */
