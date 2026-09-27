//
//  ForgeTimeline.hpp
//  Cyanescent Forge — Linux worker
//
//  C++ port of the portable, platform-free parts of Forge's CPU model:
//
//    TempoMap                      Core/MusicalTime.swift
//    AnimationTrack, Catmull-Rom   Core/Timeline.swift
//    ForgeAutomationLane           Core/Automation.swift
//    CameraState/Basis/Rig         Core/Camera.swift
//    ForgeParameterDescriptor      Core/ForgeScene.swift (validatedValue)
//    ForgeQualitySettings/presets  Core/RenderQuality.swift
//
//  Each function keeps the Swift evaluation order and precision (Double for
//  beats and seconds, Float for values) so the Linux worker evaluates the
//  same frame state as the Swift worker. The worker is built with
//  -ffp-contract=off: Swift never fuses a*b+c into an FMA, while clang (and
//  gcc in GNU mode) may on aarch64, which can change last bits.
//  `--snapshot-hashes` compares the result bit for bit with the Mac.
//
#ifndef FORGE_TIMELINE_HPP
#define FORGE_TIMELINE_HPP

#include "ForgeHostTypes.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace forge {

// Swift Float literals on arm64 (the M3 reference) are converted through
// Float64, i.e. decimal -> double -> float. F() spells the same rounding.
constexpr float F(double v) { return static_cast<float>(v); }

// --- small float3 helpers (simd semantics) ----------------------------------
inline float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float3 operator*(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(float3 a, float3 b);
float lengthSquared(float3 a);
float length(float3 a);
float3 normalize(float3 a);
float3 cross(float3 a, float3 b);

// --- MusicalTime.swift ------------------------------------------------------
struct TempoMap {
    double beatsPerMinute = 145.0;
    int beatsPerBar = 4;

    double secondsPerBeat() const;
    double seconds(double beats) const;
    double beatsFromSeconds(double seconds) const;
    int frameIndex(double beats, double frameRate) const;
};

// --- Timeline.swift ---------------------------------------------------------
enum class Interpolation { Hold, Linear, Smooth, Spline };

float lerp(float a, float b, float t);
float catmullRom(float p0, float p1, float p2, float p3, float t);
float3 lerp(float3 a, float3 b, float t);
float3 catmullRom(float3 p0, float3 p1, float3 p2, float3 p3, float t);

template <typename Value>
struct Keyframe {
    double beats;
    Value value;
    Interpolation interpolation = Interpolation::Spline;
};

template <typename Value>
struct AnimationTrack {
    std::vector<Keyframe<Value>> keyframes; // sorted by beats
    std::optional<Value> value(double beats) const;
    Value value(double beats, Value fallback) const { return value(beats).value_or(fallback); }
};

// --- Automation.swift -------------------------------------------------------
enum class AutomationInterpolation { Step, Linear, Smooth, Spline };

struct AutomationPoint {
    double beats;
    float value;
    AutomationInterpolation interpolation = AutomationInterpolation::Smooth;
};

struct AutomationLane {
    std::string parameterID;
    bool isEnabled = false;
    std::vector<AutomationPoint> points; // sorted by beats
    std::optional<float> value(double beats) const;
};

using AutomationLanes = std::map<std::string, AutomationLane>;

// --- ForgeScene.swift: ForgeParameterDescriptor.validatedValue --------------
enum class ParameterType { Float, Integer, Boolean, Enumeration };

struct ParameterDescriptor {
    std::string id;
    float lower;
    float upper;
    float defaultValue;
    ParameterType type = ParameterType::Float;
    float validatedValue(float value) const;
};

// --- Camera.swift -----------------------------------------------------------
struct CameraState {
    float3 position{0, 0, -3};
    float3 target{0, 0, 0};
    float fovDegrees = 62;
    float roll = 0;
    float focusDistance = 2;
    float apertureRadius = 0;
};

struct CameraBasis {
    float3 position, forward, right, up;
    float tanHalfFov;
    explicit CameraBasis(const CameraState &state);
};

struct CameraRig {
    AnimationTrack<float3> position, target;
    AnimationTrack<float> fovDegrees, roll;
    struct Rest {
        float3 position{0, 0, -3};
        float3 target{0, 0, 0};
        float fovDegrees = 62;
        float roll = 0;
    } restState;

    CameraState state(double beats, const ForgeSceneParams &sceneParams) const;
};

// --- RenderQuality.swift ----------------------------------------------------
struct QualitySettings {
    int raymarchSteps = 190;
    int fractalIterations = 8;
    int shadowSteps = 18;
    int aoSamples = 5;
    int volumetricSamples = 8;
    int reflectionBounces = 0;
    int samplesPerPixel = 1;
    int samplesPerBatch = 1;
    int macroLayerIterations = 5;
    bool depthOfField = false;
    float detailScale = F(1.7);
    float maxDistance = 55;
    float shadowSoftness = 12;
    float previewScale = F(1.0);
    int tileSize = 512;

    ForgeQuality gpuQuality(int sampleOffset, int sampleCount) const;
};

enum class QualityPreset { PreviewFast, PreviewHigh, OfflineStandard, OfflineHigh, OfflineExtreme, QuickLoopTest };

QualitySettings qualitySettings(QualityPreset preset);
/// Forge's display name (ForgeQualityPreset.rawValue), e.g. "Offline — High".
const char *qualityDisplayName(QualityPreset preset);
/// The worker's CLI names: standard, high, extreme, quick, fast.
std::optional<QualityPreset> qualityFromCLIName(const std::string &name);

} // namespace forge

#endif // FORGE_TIMELINE_HPP
