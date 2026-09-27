//
//  ForgeFrame.cpp
//  Cyanescent Forge — Linux worker
//
#include "ForgeFrame.hpp"
#include "MandelNewtonWorld.hpp"

#include <algorithm>
#include <cmath>

namespace forge {

int WorkerProject::renderFrameCount() const {
    const double span = std::max(endBeats - startBeats, 0.0);
    return std::max(static_cast<int>(std::round(tempo.seconds(span) * frameRate)), 1);
}

double WorkerProject::beatsForRenderFrame(int index) const {
    return startBeats + tempo.beatsFromSeconds(static_cast<double>(index) / std::max(frameRate, 1.0));
}

WorkerProject makeWorkerProject(int width, int height, double fps, uint32_t seed,
                                QualityPreset quality, int lastFrame) {
    WorkerProject p;
    p.sceneIdentifier = MandelNewtonWorld::identifier;
    p.seed = seed;
    p.tempo = TempoMap{145.0, 4}; // ApollonianJourney.referenceBPM
    p.durationBeats = MandelNewtonWorld::defaultDurationBeats;
    p.camera = MandelNewtonWorld::defaultCameraRig();
    p.automationLanes = MandelNewtonWorld::defaultAutomationLanes();
    p.parameterOverrides.clear();
    p.width = width;
    p.height = height;
    p.frameRate = fps;
    p.startBeats = 0;
    p.endBeats = p.tempo.beatsFromSeconds(static_cast<double>(lastFrame + 1) / fps);
    p.filenamePrefix = "frame";
    p.quality = quality;
    return p;
}

int baseFrame(const WorkerProject &project) {
    return project.tempo.frameIndex(project.startBeats, project.frameRate);
}

FrameRequest frameRequest(const WorkerProject &project, int absoluteFrame) {
    FrameRequest request;
    request.quality = qualitySettings(project.quality);
    request.quality.tileSize = std::min(request.quality.tileSize, 512);
    request.beats = project.beatsForRenderFrame(absoluteFrame - baseFrame(project));
    request.width = project.width;
    request.height = project.height;
    request.frameIndex = static_cast<uint32_t>(std::max(absoluteFrame, 0));
    return request;
}

EvaluatedFrame evaluateFrame(const WorkerProject &project, const FrameRequest &request) {
    EvaluatedFrame frame;
    frame.params = MandelNewtonWorld::sceneParams(request.beats, project.parameterOverrides,
                                                  project.automationLanes);
    // Depth of field is an offline luxury; preview presets switch it off.
    if (!request.quality.depthOfField) frame.params.apertureRadius = 0;
    // Mandelbrot / Newton has no procedural camera: the project's keyframed rig.
    frame.camera = project.camera.state(request.beats, frame.params);
    frame.closureTime = float2{0, 0};
    return frame;
}

ForgeUniforms makeUniforms(const WorkerProject &project, const FrameRequest &request, const Tile &tile,
                           int sampleOffset, int sampleCount, bool accumulate,
                           const EvaluatedFrame &frame) {
    ForgeSceneParams params = frame.params;
    if (!request.quality.depthOfField) params.apertureRadius = 0;
    params.focusDistance = frame.camera.focusDistance;
    const CameraBasis basis(frame.camera);

    ForgeUniforms u{};
    u.cameraPosition = basis.position;
    u.cameraForward = basis.forward;
    u.cameraRight = basis.right;
    u.cameraUp = basis.up;

    u.imageSize = float2{static_cast<float>(request.width), static_cast<float>(request.height)};
    u.tileOrigin = float2{static_cast<float>(tile.x), static_cast<float>(tile.y)};
    u.tileSize = float2{static_cast<float>(tile.width), static_cast<float>(tile.height)};

    u.tanHalfFov = basis.tanHalfFov;
    u.aspect = static_cast<float>(request.width) / static_cast<float>(std::max(request.height, 1));
    u.timeSeconds = static_cast<float>(project.tempo.seconds(request.beats));
    u.timeBeats = static_cast<float>(request.beats);

    u.seed = project.seed;
    u.frameIndex = request.frameIndex;
    u.accumulate = accumulate ? 1 : 0;

    u.quality = request.quality.gpuQuality(sampleOffset, sampleCount);
    u.quality.previewMode = request.isInteractivePreview ? 1 : 0;
    u.scene = params;
    u.loopClosureTime = frame.closureTime;
    return u;
}

std::vector<Tile> tiles(int width, int height, int tileSize) {
    if (!(tileSize > 0 && tileSize < std::max(width, height))) {
        return {Tile{0, 0, width, height}};
    }
    std::vector<Tile> result;
    for (int y = 0; y < height; y += tileSize) {
        const int h = std::min(tileSize, height - y);
        for (int x = 0; x < width; x += tileSize) {
            const int w = std::min(tileSize, width - x);
            result.push_back(Tile{x, y, w, h});
        }
    }
    return result;
}

std::vector<SampleBatch> sampleBatches(const QualitySettings &quality) {
    const int total = std::max(quality.samplesPerPixel, 1);
    const int batch = std::max(std::min(quality.samplesPerBatch, total), 1);
    std::vector<SampleBatch> result;
    for (int offset = 0; offset < total; offset += batch) {
        result.push_back(SampleBatch{offset, std::min(batch, total - offset)});
    }
    return result;
}

} // namespace forge
