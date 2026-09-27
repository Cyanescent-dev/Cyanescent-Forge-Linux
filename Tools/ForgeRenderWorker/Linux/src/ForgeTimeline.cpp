//
//  ForgeTimeline.cpp
//  Cyanescent Forge — Linux worker
//
//  See ForgeTimeline.hpp. Swift line references are given where the order of
//  operations matters for bit-exact agreement.
//
#include "ForgeTimeline.hpp"

#include <algorithm>
#include <cmath>

namespace forge {

// simd_dot / simd_length_squared: products summed left to right.
float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float lengthSquared(float3 a) { return dot(a, a); }
float length(float3 a) { return std::sqrt(lengthSquared(a)); }
// simd_normalize (precise): x * rsqrt(length_squared(x)), rsqrt = 1 / sqrt.
float3 normalize(float3 a) { return a * (1.0f / std::sqrt(lengthSquared(a))); }
float3 cross(float3 a, float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// --- TempoMap ---------------------------------------------------------------
double TempoMap::secondsPerBeat() const { return 60.0 / std::max(beatsPerMinute, 1.0); }
double TempoMap::seconds(double beats) const { return beats * secondsPerBeat(); }
double TempoMap::beatsFromSeconds(double s) const { return s / secondsPerBeat(); }
int TempoMap::frameIndex(double beats, double frameRate) const {
    return static_cast<int>(std::floor(seconds(beats) * frameRate)); // .rounded(.down)
}

// --- interpolation ----------------------------------------------------------
float lerp(float a, float b, float t) { return a + (b - a) * t; }

float catmullRom(float p0, float p1, float p2, float p3, float t) {
    const float t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2 * p1)
                   + (-p0 + p2) * t
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }

float3 catmullRom(float3 p0, float3 p1, float3 p2, float3 p3, float t) {
    return {catmullRom(p0.x, p1.x, p2.x, p3.x, t),
            catmullRom(p0.y, p1.y, p2.y, p3.y, t),
            catmullRom(p0.z, p1.z, p2.z, p3.z, t)};
}

// Timeline.swift AnimationTrack.value(atBeats:)
template <typename Value>
std::optional<Value> AnimationTrack<Value>::value(double beats) const {
    if (keyframes.empty()) return std::nullopt;
    const auto &first = keyframes.front();
    const auto &last = keyframes.back();
    if (keyframes.size() == 1 || beats <= first.beats) return first.value;
    if (beats >= last.beats) return last.value;

    size_t i = 0;
    while (i < keyframes.size() - 1 && keyframes[i + 1].beats <= beats) ++i;

    const auto &k1 = keyframes[i];
    const auto &k2 = keyframes[std::min(i + 1, keyframes.size() - 1)];
    const double span = std::max(k2.beats - k1.beats, 1e-6);
    float t = static_cast<float>((beats - k1.beats) / span);

    switch (k1.interpolation) {
    case Interpolation::Hold:
        return k1.value;
    case Interpolation::Linear:
        return lerp(k1.value, k2.value, t);
    case Interpolation::Smooth:
        t = t * t * (3 - 2 * t);
        return lerp(k1.value, k2.value, t);
    case Interpolation::Spline: {
        const auto &k0 = keyframes[i > 0 ? i - 1 : 0];
        const auto &k3 = keyframes[std::min(i + 2, keyframes.size() - 1)];
        t = t * t * (3 - 2 * t);
        return catmullRom(k0.value, k1.value, k2.value, k3.value, t);
    }
    }
    return k1.value;
}

template struct AnimationTrack<float>;
template struct AnimationTrack<float3>;

// Automation.swift ForgeAutomationLane.value(atBeats:)
std::optional<float> AutomationLane::value(double beats) const {
    if (points.empty()) return std::nullopt;
    const auto &first = points.front();
    const auto &last = points.back();
    if (points.size() == 1 || beats <= first.beats) return first.value;
    if (beats >= last.beats) return last.value;

    size_t index = 0;
    while (index + 1 < points.size() && points[index + 1].beats <= beats) ++index;

    const auto &a = points[index];
    const auto &b = points[index + 1];
    const double span = std::max(b.beats - a.beats, 1e-9);
    float t = static_cast<float>((beats - a.beats) / span);

    switch (a.interpolation) {
    case AutomationInterpolation::Step:
        return a.value;
    case AutomationInterpolation::Linear:
        return lerp(a.value, b.value, t);
    case AutomationInterpolation::Smooth:
        t = t * t * (3 - 2 * t);
        return lerp(a.value, b.value, t);
    case AutomationInterpolation::Spline: {
        const float p0 = points[index > 0 ? index - 1 : 0].value;
        const float p3 = points[std::min(index + 2, points.size() - 1)].value;
        t = t * t * (3 - 2 * t);
        return catmullRom(p0, a.value, b.value, p3, t);
    }
    }
    return a.value;
}

// Swift's min/max: min(x, y) = y < x ? y : x; max(x, y) = y >= x ? y : x.
static float swiftMin(float x, float y) { return y < x ? y : x; }
static float swiftMax(float x, float y) { return y >= x ? y : x; }

float ParameterDescriptor::validatedValue(float value) const {
    if (!std::isfinite(value)) return defaultValue;
    const float clamped = swiftMin(swiftMax(value, lower), upper);
    switch (type) {
    case ParameterType::Float:
        return clamped;
    case ParameterType::Integer:
    case ParameterType::Enumeration:
        return swiftMin(swiftMax(std::round(clamped), lower), upper); // .rounded(): half away from zero
    case ParameterType::Boolean:
        return clamped >= 0.5f ? 1.0f : 0.0f;
    }
    return clamped;
}

// --- Camera.swift -----------------------------------------------------------
CameraBasis::CameraBasis(const CameraState &state) {
    const float3 delta = state.target - state.position;
    const float3 fwd = length(delta) > 1e-5f ? normalize(delta) : float3{0, 0, 1};

    float3 worldUp{0, 1, 0};
    if (std::fabs(dot(fwd, worldUp)) > F(0.995)) worldUp = float3{0, 0, 1};

    const float3 r = normalize(cross(fwd, worldUp));
    const float3 u = cross(r, fwd);

    const float c = std::cos(state.roll), s = std::sin(state.roll);
    position = state.position;
    forward = fwd;
    right = r * c + u * s;
    up = u * c - r * s;
    // Swift's Float.pi is pi rounded toward zero (0x1.921fb4p+1), not to nearest.
    constexpr float swiftFloatPi = 0x1.921fb4p+1f;
    tanHalfFov = std::tan(state.fovDegrees * 0.5f * swiftFloatPi / 180);
}

CameraState CameraRig::state(double beats, const ForgeSceneParams &sceneParams) const {
    CameraState s;
    s.position = position.value(beats, restState.position);
    s.target = target.value(beats, restState.target);
    s.fovDegrees = fovDegrees.value(beats, restState.fovDegrees);
    s.roll = roll.value(beats, restState.roll);
    s.focusDistance = swiftMax(length(s.target - s.position), F(0.05));
    s.apertureRadius = sceneParams.apertureRadius;
    return s;
}

// --- RenderQuality.swift ----------------------------------------------------
ForgeQuality QualitySettings::gpuQuality(int sampleOffset, int sampleCount) const {
    ForgeQuality q{};
    q.raymarchSteps = raymarchSteps;
    q.fractalIterations = fractalIterations;
    q.shadowSteps = shadowSteps;
    q.aoSamples = aoSamples;
    q.volumetricSamples = volumetricSamples;
    q.reflectionBounces = reflectionBounces;
    q.samplesPerPixel = std::max(sampleCount, 1);
    q.totalSamples = std::max(samplesPerPixel, 1);
    q.sampleOffset = sampleOffset;
    q.macroLayerIterations = macroLayerIterations;
    q.enableDepthOfField = depthOfField ? 1 : 0;
    q.detailScale = detailScale;
    q.maxDistance = maxDistance;
    q.shadowSoftness = shadowSoftness;
    return q;
}

QualitySettings qualitySettings(QualityPreset preset) {
    QualitySettings q;
    switch (preset) {
    case QualityPreset::PreviewFast:
        q.raymarchSteps = 190; q.fractalIterations = 8; q.shadowSteps = 18;
        q.aoSamples = 5; q.volumetricSamples = 8; q.reflectionBounces = 0;
        q.samplesPerPixel = 1; q.samplesPerBatch = 1; q.macroLayerIterations = 5;
        q.depthOfField = false;
        q.detailScale = F(1.7); q.maxDistance = 55; q.shadowSoftness = 12;
        q.previewScale = F(0.6);
        break;
    case QualityPreset::PreviewHigh:
        q.raymarchSteps = 380; q.fractalIterations = 12; q.shadowSteps = 34;
        q.aoSamples = 7; q.volumetricSamples = 20; q.reflectionBounces = 1;
        q.samplesPerPixel = 1; q.samplesPerBatch = 1; q.macroLayerIterations = 7;
        q.depthOfField = false;
        q.detailScale = F(0.85); q.maxDistance = 90; q.shadowSoftness = 16;
        q.previewScale = F(1.0);
        break;
    case QualityPreset::OfflineStandard:
        q.raymarchSteps = 700; q.fractalIterations = 16; q.shadowSteps = 56;
        q.aoSamples = 9; q.volumetricSamples = 32; q.reflectionBounces = 1;
        q.samplesPerPixel = 8; q.samplesPerBatch = 2; q.macroLayerIterations = 9;
        q.depthOfField = true;
        q.detailScale = F(0.50); q.maxDistance = 120; q.shadowSoftness = 20;
        q.tileSize = 512;
        break;
    case QualityPreset::OfflineHigh:
        q.raymarchSteps = 900; q.fractalIterations = 20; q.shadowSteps = 72;
        q.aoSamples = 10; q.volumetricSamples = 48; q.reflectionBounces = 1;
        q.samplesPerPixel = 16; q.samplesPerBatch = 2; q.macroLayerIterations = 10;
        q.depthOfField = true;
        q.detailScale = F(0.35); q.maxDistance = 140; q.shadowSoftness = 22;
        q.tileSize = 384;
        break;
    case QualityPreset::QuickLoopTest:
        q.raymarchSteps = 300; q.fractalIterations = 10; q.shadowSteps = 16;
        q.aoSamples = 4; q.volumetricSamples = 8; q.reflectionBounces = 0;
        q.samplesPerPixel = 2; q.samplesPerBatch = 2; q.macroLayerIterations = 6;
        q.depthOfField = false;
        q.detailScale = F(1.2); q.maxDistance = 120; q.shadowSoftness = 14;
        q.tileSize = 512;
        break;
    case QualityPreset::OfflineExtreme:
        q.raymarchSteps = 1400; q.fractalIterations = 24; q.shadowSteps = 110;
        q.aoSamples = 14; q.volumetricSamples = 96; q.reflectionBounces = 2;
        q.samplesPerPixel = 36; q.samplesPerBatch = 2; q.macroLayerIterations = 12;
        q.depthOfField = true;
        q.detailScale = F(0.22); q.maxDistance = 200; q.shadowSoftness = 26;
        q.tileSize = 256;
        break;
    }
    return q;
}

const char *qualityDisplayName(QualityPreset preset) {
    switch (preset) {
    case QualityPreset::PreviewFast: return "Preview — Fast";
    case QualityPreset::PreviewHigh: return "Preview — High";
    case QualityPreset::OfflineStandard: return "Offline — Standard";
    case QualityPreset::OfflineHigh: return "Offline — High";
    case QualityPreset::OfflineExtreme: return "Offline — Extreme";
    case QualityPreset::QuickLoopTest: return "Quick Loop Test";
    }
    return "";
}

std::optional<QualityPreset> qualityFromCLIName(const std::string &name) {
    if (name == "standard") return QualityPreset::OfflineStandard;
    if (name == "high") return QualityPreset::OfflineHigh;
    if (name == "extreme") return QualityPreset::OfflineExtreme;
    if (name == "quick") return QualityPreset::QuickLoopTest;
    if (name == "fast") return QualityPreset::PreviewFast;
    return std::nullopt;
}

} // namespace forge
