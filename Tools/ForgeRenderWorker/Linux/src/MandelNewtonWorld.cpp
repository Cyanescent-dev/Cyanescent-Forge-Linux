//
//  MandelNewtonWorld.cpp
//  Cyanescent Forge — Linux worker
//
//  See MandelNewtonWorld.hpp. Values are copied from the Swift sources; keep
//  them in step with MandelNewtonWorldScene.swift and
//  ForgeWorldSceneSupport.swift (`--snapshot-hashes` on both workers shows
//  any drift as a differing sceneParams / camera hash).
//
#include "MandelNewtonWorld.hpp"

#include <functional>

namespace forge {

namespace {

using PT = ParameterType;

ParameterDescriptor desc(const char *id, double lo, double hi, double def, PT type = PT::Float) {
    return ParameterDescriptor{id, F(lo), F(hi), F(def), type};
}

// MandelNewtonWorldScene.worldDescriptors, in order.
const std::vector<ParameterDescriptor> &worldDescriptors() {
    static const std::vector<ParameterDescriptor> list = {
        desc("mnBalance", 0, 1, 0.5),
        desc("mnZoom", 0.02, 4, 0.5),
        desc("mnCenterX", -2.5, 2.5, -1.2),
        desc("mnCenterY", -2.5, 2.5, 0.25),
        desc("mnRotation", -6.2832, 6.2832, 0.0),
        desc("mnWarp", 0, 0.12, 0.02),
        desc("mnReliefMode", 0, 1, 1, PT::Boolean),
        desc("mnReliefHeight", 0, 2, 0.45),
        desc("mnReliefFine", 0, 0.3, 0.03),
        desc("mnBandRelief", 0, 0.1, 0.015),
        desc("mnReliefDetail", 0.05, 4, 0.45),
        desc("mnWorldScale", 0.1, 4, 1.0),
        desc("mnBandDensity", 0.2, 4, 1.0),
        desc("mnBoundaryGlow", 0, 2, 0.35),
        desc("mnTrapInfluence", 0, 2, 1.0),
        desc("mnPaletteDrift", 0, 1, 0.0),
        desc("mnDebugView", 0, 2, 0, PT::Integer),
        desc("mnCoexistGain", 0, 1, 0.25),
    };
    return list;
}

// ForgeWorldSceneSupport.appearanceDescriptors, in order.
const std::vector<ParameterDescriptor> &appearanceDescriptors() {
    static const std::vector<ParameterDescriptor> list = {
        desc("paletteShift", 0, 1, 0.28),
        desc("paletteSpread", 0.10, 1.40, 0.72),
        desc("emissiveStrength", 0, 2.5, 0.22),
        desc("roughnessBase", 0.05, 1, 0.46),
        desc("metallicBase", 0, 1, 0.22),
        desc("keyLightIntensity", 0, 8, 2.4),
        desc("ambientIntensity", 0, 2.5, 0.62),
        desc("headlightIntensity", 0, 7, 2.1),
        desc("exposure", 0.15, 3, 1.05),
        desc("atmosphericDepthEnabled", 0, 1, 1, PT::Boolean),
        desc("atmosphericDepthDensity", 0, 0.35, 0.035),
        desc("atmosphericDepthStart", 0, 80, 3),
        desc("atmosphericDepthMaximumStrength", 0, 1, 0.72),
        desc("atmosphericDepthColorRed", 0, 1, 0.025),
        desc("atmosphericDepthColorGreen", 0, 1, 0.095),
        desc("atmosphericDepthColorBlue", 0, 1, 0.190),
    };
    return list;
}

// MandelNewtonWorldScene.appearanceOverrides
const std::map<std::string, float> &appearanceOverrides() {
    static const std::map<std::string, float> overrides = {
        {"atmosphericDepthStart", F(1.5)},
        {"atmosphericDepthColorRed", F(0.13)},
        {"atmosphericDepthColorGreen", F(0.15)},
        {"atmosphericDepthColorBlue", F(0.28)},
        {"ambientIntensity", F(0.32)},
        {"emissiveStrength", F(0.4)},
        {"roughnessBase", F(0.5)},
        {"metallicBase", F(0.15)},
        {"keyLightIntensity", F(6.0)},
        {"headlightIntensity", F(0.35)},
        {"exposure", F(0.85)},
        {"atmosphericDepthDensity", F(0.035)},
        {"atmosphericDepthMaximumStrength", F(0.85)},
    };
    return overrides;
}

// ForgeWorldSceneSupport.apply(_:_:to:)
bool applyAppearance(const std::string &id, float value, ForgeSceneParams &s) {
    if (id == "paletteShift") s.paletteShift = value;
    else if (id == "paletteSpread") s.paletteSpread = value;
    else if (id == "emissiveStrength") s.emissiveStrength = value;
    else if (id == "roughnessBase") s.roughnessBase = value;
    else if (id == "metallicBase") s.metallicBase = value;
    else if (id == "keyLightIntensity") s.keyLightIntensity = value;
    else if (id == "ambientIntensity") s.ambientIntensity = value;
    else if (id == "headlightIntensity") s.headlightIntensity = value;
    else if (id == "exposure") s.exposure = value;
    else if (id == "atmosphericDepthEnabled") s.atmosphericDepthEnabled = value;
    else if (id == "atmosphericDepthDensity") s.atmosphericDepthDensity = value;
    else if (id == "atmosphericDepthStart") s.atmosphericDepthStart = value;
    else if (id == "atmosphericDepthMaximumStrength") s.atmosphericDepthMaximumStrength = value;
    else if (id == "atmosphericDepthColorRed") s.atmosphericDepthColor.x = value;
    else if (id == "atmosphericDepthColorGreen") s.atmosphericDepthColor.y = value;
    else if (id == "atmosphericDepthColorBlue") s.atmosphericDepthColor.z = value;
    else return false;
    return true;
}

// MandelNewtonWorldScene.setWorldParameter
bool setWorldParameter(const std::string &id, float value, ForgeSceneParams &s) {
    if (id == "mnBalance") s.mnBalance = value;
    else if (id == "mnZoom") s.mnZoom = value;
    else if (id == "mnCenterX") s.mnCenterX = value;
    else if (id == "mnCenterY") s.mnCenterY = value;
    else if (id == "mnRotation") s.mnRotation = value;
    else if (id == "mnWarp") s.mnWarp = value;
    else if (id == "mnReliefMode") s.mnReliefMode = value;
    else if (id == "mnReliefHeight") s.mnReliefHeight = value;
    else if (id == "mnReliefFine") s.mnReliefFine = value;
    else if (id == "mnBandRelief") s.mnBandRelief = value;
    else if (id == "mnReliefDetail") s.mnReliefDetail = value;
    else if (id == "mnWorldScale") s.mnWorldScale = value;
    else if (id == "mnBandDensity") s.mnBandDensity = value;
    else if (id == "mnBoundaryGlow") s.mnBoundaryGlow = value;
    else if (id == "mnTrapInfluence") s.mnTrapInfluence = value;
    else if (id == "mnPaletteDrift") s.mnPaletteDrift = value;
    else if (id == "mnCoexistGain") s.mnCoexistGain = value;
    else if (id == "mnDebugView") s.mnReserved1 = value;
    else return false;
    return true;
}

// MandelNewtonWorldScene.worldDefaults
void worldDefaults(ForgeSceneParams &s) {
    s.mnBalance = F(0.5);
    s.mnZoom = F(0.5);
    s.mnCenterX = F(-1.2);
    s.mnCenterY = F(0.25);
    s.mnRotation = F(0.0);
    s.mnWarp = F(0.02);
    s.mnBandDensity = F(1.0);
    s.mnBoundaryGlow = F(0.35);
    s.mnPaletteDrift = F(0.0);
    s.mnCoexistGain = F(0.25);
    s.mnReliefHeight = F(0.45);
    s.mnReliefMode = 1;
    s.mnWorldScale = F(1.0);
    s.mnTrapInfluence = F(1.0);
    s.mnReliefSharpness = F(1.0);
    s.mnReliefDetail = F(0.45);
    s.mnReliefFine = F(0.03);
    s.mnBandRelief = F(0.015);
}

AutomationLane lane(const char *id, std::initializer_list<std::pair<double, double>> points) {
    AutomationLane result;
    result.parameterID = id;
    result.isEnabled = true;
    for (const auto &p : points) {
        result.points.push_back(AutomationPoint{p.first, F(p.second), AutomationInterpolation::Smooth});
    }
    return result;
}

struct CameraKey {
    double beats;
    float3 position;
    float3 target;
    float fov;
    float roll;
};

// ForgeWorldSceneSupport.makeCamera
CameraRig makeCamera(const std::vector<CameraKey> &keys) {
    CameraRig rig;
    for (const auto &k : keys) {
        rig.position.keyframes.push_back({k.beats, k.position, Interpolation::Spline});
        rig.target.keyframes.push_back({k.beats, k.target, Interpolation::Spline});
        rig.fovDegrees.keyframes.push_back({k.beats, k.fov, Interpolation::Spline});
        rig.roll.keyframes.push_back({k.beats, k.roll, Interpolation::Spline});
    }
    if (!keys.empty()) {
        rig.restState.position = keys.front().position;
        rig.restState.target = keys.front().target;
        rig.restState.fovDegrees = keys.front().fov;
        rig.restState.roll = keys.front().roll;
    }
    return rig;
}

float3 v3(double x, double y, double z) { return float3{F(x), F(y), F(z)}; }

} // namespace

ForgeSceneParams apollonianJourneyDefaults() {
    ForgeSceneParams s{};
    s.foldScale = F(1.38000);
    s.macroFoldScale = F(1.07000);
    s.macroScale = F(0.15000);
    s.macroWeight = F(0.00000);
    s.levelTwist = F(0.15000);
    s.cavityBias = F(0.00000);
    s.surfaceDetail = F(0.50000);
    s.paletteShift = F(0.15000);
    s.paletteSpread = F(0.50000);
    s.emissiveStrength = F(0.80000);
    s.roughnessBase = F(0.42000);
    s.metallicBase = F(0.55000);
    s.fogDensity = F(0.01400);
    s.fogHeightFalloff = F(0.60000);
    s.godRayStrength = F(0.05500);
    s.exposure = F(1.05000);
    s.atmosphericDepthEnabled = F(0.00000);
    s.atmosphericDepthDensity = F(0.05500);
    s.atmosphericDepthStart = F(4.00000);
    s.atmosphericDepthMaximumStrength = F(0.75000);
    s.worldScale = F(0.15000);
    s.glowStrength = F(0.03000);
    s.headlightIntensity = F(0.45000);
    s.bounceIntensity = F(0.30000);
    s.fogColor = v3(0.08500, 0.21500, 0.33000);
    s.atmosphericDepthColor = v3(0.03000, 0.11000, 0.26000);
    s.emissiveColor = v3(0.95000, 0.32000, 0.78000);
    s.keyLightColor = v3(1.00000, 0.87000, 0.70000);
    s.fillLightColor = v3(0.62000, 0.88000, 1.00000);
    s.skyColorZenith = v3(0.00600, 0.01400, 0.04800);
    s.skyColorHorizon = v3(0.03600, 0.08600, 0.17000);
    s.keyLightDirection = v3(0.50000, 0.70000, -0.45000);
    s.keyLightIntensity = F(7.00000);
    s.ambientIntensity = F(0.34000);
    s.apertureRadius = F(0.00600);
    s.focusDistance = F(1.40000);
    s.keyLightDirection = normalize(s.keyLightDirection);
    return s;
}

ForgeSceneParams worldAppearanceDefaults() {
    ForgeSceneParams s = apollonianJourneyDefaults();
    s.fogDensity = 0;
    s.godRayStrength = 0;
    s.atmosphericDepthEnabled = 1;
    s.atmosphericDepthDensity = F(0.035);
    s.atmosphericDepthStart = F(3.0);
    s.atmosphericDepthMaximumStrength = F(0.72);
    s.atmosphericDepthColor = v3(0.025, 0.095, 0.190);
    s.paletteShift = F(0.28);
    s.paletteSpread = F(0.72);
    s.emissiveStrength = F(0.22);
    s.roughnessBase = F(0.46);
    s.metallicBase = F(0.22);
    s.exposure = F(1.05);
    s.keyLightIntensity = F(2.4);
    s.ambientIntensity = F(0.62);
    s.headlightIntensity = F(2.1);
    s.bounceIntensity = F(0.22);
    s.apertureRadius = 0;
    return s;
}

std::vector<ParameterDescriptor> MandelNewtonWorld::parameterDescriptors() {
    // worldDescriptors + appearanceDescriptors(overriding: appearanceOverrides)
    std::vector<ParameterDescriptor> result = worldDescriptors();
    for (ParameterDescriptor d : appearanceDescriptors()) {
        auto it = appearanceOverrides().find(d.id);
        if (it != appearanceOverrides().end()) d.defaultValue = it->second;
        result.push_back(d);
    }
    return result;
}

ForgeSceneParams MandelNewtonWorld::sceneParams(double beats, const std::map<std::string, float> &overrides,
                                                const AutomationLanes &lanes) {
    ForgeSceneParams s = worldAppearanceDefaults();
    s.fogDensity = F(0.04);
    s.fogHeightFalloff = F(6.0);
    s.godRayStrength = F(0.10);
    s.fogColor = v3(0.13, 0.15, 0.28);
    s.skyColorHorizon = v3(0.16, 0.17, 0.30);
    s.skyColorZenith = v3(0.010, 0.016, 0.05);
    s.bounceIntensity = F(0.35);
    s.keyLightDirection = normalize(v3(-0.62, 0.30, 0.45));
    s.keyLightColor = v3(1.0, 0.82, 0.62);
    s.fillLightColor = v3(0.55, 0.75, 1.0);
    worldDefaults(s);

    // ForgeWorldSceneSupport.evaluatedValues + apply(values:to:descriptors:setWorld:)
    for (const ParameterDescriptor &d : parameterDescriptors()) {
        float raw;
        auto laneIt = lanes.find(d.id);
        if (laneIt != lanes.end() && laneIt->second.isEnabled) {
            raw = laneIt->second.value(beats).value_or(d.defaultValue);
        } else {
            auto overrideIt = overrides.find(d.id);
            raw = overrideIt != overrides.end() ? overrideIt->second : d.defaultValue;
        }
        const float value = d.validatedValue(raw);
        if (!applyAppearance(d.id, value, s)) setWorldParameter(d.id, value, s);
    }
    return s;
}

CameraRig MandelNewtonWorld::defaultCameraRig() {
    return makeCamera({
        {0,   v3(0.30, 0.72, -0.95), v3(-0.20, 0.12, 0.25), F(66), F(0.00)},
        {24,  v3(0.55, 0.62, -0.40), v3(-0.15, 0.10, 0.30), F(68), F(0.03)},
        {48,  v3(0.35, 0.78, 0.45),  v3(-0.35, 0.10, 0.05), F(70), F(-0.03)},
        {72,  v3(-0.35, 0.80, 0.55), v3(-0.05, 0.08, -0.35), F(66), F(0.02)},
        {96,  v3(-0.62, 0.60, -0.15), v3(0.25, 0.10, -0.10), F(64), F(-0.02)},
        {128, v3(-0.15, 0.72, -0.90), v3(0.15, 0.10, 0.20), F(66), F(0.00)},
    });
}

AutomationLanes MandelNewtonWorld::defaultAutomationLanes() {
    AutomationLanes lanes;
    lanes["mnBalance"] = lane("mnBalance", {{0, 0.30}, {32, 0.52}, {64, 0.78}, {96, 0.60}, {128, 0.36}});
    lanes["mnPaletteDrift"] = lane("mnPaletteDrift", {{0, 0.0}, {64, 0.18}, {128, 0.36}});
    lanes["mnZoom"] = lane("mnZoom", {{0, 0.55}, {64, 0.42}, {128, 0.55}});
    lanes["mnRotation"] = lane("mnRotation", {{0, 0.0}, {128, 0.6}});
    return lanes;
}

} // namespace forge
