//
//  SnapshotHash.hpp
//  Cyanescent Forge — Linux worker
//
//  Port of Tools/ForgeRenderWorker/SnapshotHash.swift for the sections a
//  Mandelbrot / Newton CLI render has: "frame", "loopClosure", "sceneParams"
//  and "camera". The byte streams are the same canonical little-endian,
//  field-by-field serialization, so each section's SHA-256 can be compared
//  directly with `ForgeRenderWorker --snapshot-hashes` on a Mac.
//
//  The "project" section (Forge's Codable project document as a value tree)
//  is not reproduced: the Linux worker has no ForgeProject JSON encoder, and
//  every renderer-facing value that document feeds is already covered by
//  the sections above.
//
#ifndef FORGE_SNAPSHOT_HASH_HPP
#define FORGE_SNAPSHOT_HASH_HPP

#include "ForgeFrame.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace forge {

std::string sha256Hex(const std::vector<uint8_t> &bytes);

struct CanonicalWriter {
    std::vector<uint8_t> bytes;
    void u32(uint32_t v);
    void u64(uint64_t v);
    void i64(int64_t v);
    void f32(float v);
    void f64(double v);
    void boolean(bool v);
    void v2(float2 v);
    void v3(float3 v);
};

/// ForgeSceneParams float by float, skipping each 3-vector's padding lane.
void canonicalSceneParams(const ForgeSceneParams &p, CanonicalWriter &w);
/// ForgeUniforms the same way (used by the CPU fidelity harness).
std::vector<uint32_t> canonicalUniformWords(const ForgeUniforms &u);

struct SnapshotSections {
    int frame;
    std::vector<std::pair<std::string, std::string>> hashes;          // name, sha256
    std::vector<std::pair<std::string, std::vector<uint8_t>>> bytes;  // name, canonical bytes
};

SnapshotSections snapshotHashes(int absoluteFrame, const WorkerProject &project, const FrameRequest &request,
                                const EvaluatedFrame &frame);

} // namespace forge

#endif // FORGE_SNAPSHOT_HASH_HPP
