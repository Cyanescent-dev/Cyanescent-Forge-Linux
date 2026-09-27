//
//  ForgeFrame.hpp
//  Cyanescent Forge — Linux worker
//
//  From a worker project and an absolute frame number to the GPU uniform
//  block, as the macOS worker does it:
//
//    makeWorkerProject     Tools/ForgeRenderWorker/main.swift workerProject()
//    baseFrame/frameRequest  Core/RemoteRenderJob.swift ForgeRemoteRender
//    evaluateFrame         Render/ForgeRenderer.swift evaluateFrame() (loop closure off)
//    makeUniforms          Render/ForgeRenderer.swift uniforms(for:tile:...captured...)
//    tiles/sampleBatches   Render/ForgeRenderer.swift
//
//  Loop closure is off for every CLI render (the macOS worker's CLI project
//  sets it off too), so the closure branches reduce to their "off" values.
//
#ifndef FORGE_FRAME_HPP
#define FORGE_FRAME_HPP

#include "ForgeTimeline.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace forge {

/// ForgeProject, reduced to what a CLI worker render reads.
struct WorkerProject {
    std::string sceneIdentifier;
    TempoMap tempo;
    uint32_t seed = 12345;
    double durationBeats = 128;
    CameraRig camera;
    AutomationLanes automationLanes;
    std::map<std::string, float> parameterOverrides;
    int width = 1920;
    int height = 1080;
    double frameRate = 30;
    double startBeats = 0;
    double endBeats = 8;
    QualityPreset quality = QualityPreset::OfflineHigh;
    std::string filenamePrefix = "frame";

    int renderFrameCount() const;
    double beatsForRenderFrame(int index) const;
};

/// The project Forge makes for this world on a fresh project, seeded, with
/// a render range from frame 0 through `lastFrame` (main.swift workerProject).
WorkerProject makeWorkerProject(int width, int height, double fps, uint32_t seed,
                                QualityPreset quality, int lastFrame);

/// Absolute frame number of the project's first render frame.
int baseFrame(const WorkerProject &project);

struct FrameRequest {
    double beats = 0;
    int width = 0;
    int height = 0;
    QualitySettings quality;
    uint32_t frameIndex = 0;
    bool isInteractivePreview = false;
};

FrameRequest frameRequest(const WorkerProject &project, int absoluteFrame);

struct EvaluatedFrame {
    ForgeSceneParams params;
    CameraState camera;
    float2 closureTime;
};

EvaluatedFrame evaluateFrame(const WorkerProject &project, const FrameRequest &request);

struct Tile {
    int x, y, width, height;
};

ForgeUniforms makeUniforms(const WorkerProject &project, const FrameRequest &request, const Tile &tile,
                           int sampleOffset, int sampleCount, bool accumulate,
                           const EvaluatedFrame &frame);

std::vector<Tile> tiles(int width, int height, int tileSize);

struct SampleBatch {
    int offset;
    int count;
};
std::vector<SampleBatch> sampleBatches(const QualitySettings &quality);

} // namespace forge

#endif // FORGE_FRAME_HPP
