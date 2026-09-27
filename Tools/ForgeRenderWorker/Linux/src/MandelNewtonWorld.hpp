//
//  MandelNewtonWorld.hpp
//  Cyanescent Forge — Linux worker
//
//  C++ port of the authoring side of the Mandelbrot / Newton Morph world:
//
//    MandelNewtonWorldScene          Scenes/MandelNewtonWorldScene.swift
//    ForgeWorldSceneSupport          Scenes/ForgeWorldSceneSupport.swift
//    ApollonianJourney.defaults()    Scenes/ApollonianJourney.swift
//
//  Same defaults, descriptors (ranges, types, validation), appearance
//  overrides, default camera journey and automation lanes, evaluated in the
//  same order, so sceneParams(beats) is the ForgeSceneParams Forge renders.
//
#ifndef FORGE_MANDEL_NEWTON_WORLD_HPP
#define FORGE_MANDEL_NEWTON_WORLD_HPP

#include "ForgeTimeline.hpp"

namespace forge {

struct MandelNewtonWorld {
    static constexpr const char *identifier = "mandelnewton.world";
    static constexpr const char *cliName = "mandelnewton";
    static constexpr const char *displayName = "Mandelbrot / Newton Morph";
    static constexpr double defaultDurationBeats = 128;

    static std::vector<ParameterDescriptor> parameterDescriptors();
    static ForgeSceneParams sceneParams(double beats, const std::map<std::string, float> &overrides,
                                        const AutomationLanes &lanes);
    static CameraRig defaultCameraRig();
    static AutomationLanes defaultAutomationLanes();
};

/// ApollonianJourney.defaults(): the shared cinematic palette every world starts from.
ForgeSceneParams apollonianJourneyDefaults();
/// ForgeWorldSceneSupport.appearanceDefaults()
ForgeSceneParams worldAppearanceDefaults();

} // namespace forge

#endif // FORGE_MANDEL_NEWTON_WORLD_HPP
