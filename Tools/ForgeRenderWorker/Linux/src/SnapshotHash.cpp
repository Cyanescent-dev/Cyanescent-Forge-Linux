//
//  SnapshotHash.cpp
//  Cyanescent Forge — Linux worker
//
#include "SnapshotHash.hpp"

#include <cstddef>
#include <cstring>
#include <set>

namespace forge {

// --- SHA-256 (FIPS 180-4), portable: no intrinsics, explicit big-endian ------
namespace {

const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

} // namespace

std::string sha256Hex(const std::vector<uint8_t> &input) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(input);
    const uint64_t bitLength = static_cast<uint64_t>(input.size()) * 8;
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back(static_cast<uint8_t>(bitLength >> (i * 8)));

    for (size_t chunkStart = 0; chunkStart < m.size(); chunkStart += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t *p = &m[chunkStart + i * 4];
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char *hex = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h) {
        for (int i = 7; i >= 0; --i) out.push_back(hex[(v >> (i * 4)) & 0xf]);
    }
    return out;
}

// --- canonical writer: little-endian by construction ------------------------
void CanonicalWriter::u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(v >> (i * 8)));
}
void CanonicalWriter::u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>(v >> (i * 8)));
}
void CanonicalWriter::i64(int64_t v) { u64(static_cast<uint64_t>(v)); }
void CanonicalWriter::f32(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    u32(bits);
}
void CanonicalWriter::f64(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    u64(bits);
}
void CanonicalWriter::boolean(bool v) { bytes.push_back(v ? 1 : 0); }
void CanonicalWriter::v2(float2 v) { f32(v.x); f32(v.y); }
void CanonicalWriter::v3(float3 v) { f32(v.x); f32(v.y); f32(v.z); }

namespace {

// Padding lanes of every 3-vector in ForgeSceneParams, by offset.
std::set<size_t> sceneParamPadding() {
    return {offsetof(ForgeSceneParams, fogColor) + 12, offsetof(ForgeSceneParams, atmosphericDepthColor) + 12,
            offsetof(ForgeSceneParams, emissiveColor) + 12, offsetof(ForgeSceneParams, keyLightColor) + 12,
            offsetof(ForgeSceneParams, fillLightColor) + 12, offsetof(ForgeSceneParams, skyColorZenith) + 12,
            offsetof(ForgeSceneParams, skyColorHorizon) + 12, offsetof(ForgeSceneParams, keyLightDirection) + 12};
}

// Swift's MemoryLayout<ForgeSceneParams>.size: up to the end of the last field.
constexpr size_t sceneParamsSize = offsetof(ForgeSceneParams, apReserved0) + 4;

void appendWords(const uint8_t *raw, size_t size, const std::set<size_t> &padding, std::vector<uint32_t> &out) {
    for (size_t offset = 0; offset < size; offset += 4) {
        if (padding.count(offset)) continue;
        uint32_t word;
        std::memcpy(&word, raw + offset, 4);
        out.push_back(word);
    }
}

} // namespace

void canonicalSceneParams(const ForgeSceneParams &p, CanonicalWriter &w) {
    static_assert(sceneParamsSize % 4 == 0, "ForgeSceneParams size must be a multiple of 4");
    std::vector<uint32_t> words;
    appendWords(reinterpret_cast<const uint8_t *>(&p), sceneParamsSize, sceneParamPadding(), words);
    for (uint32_t word : words) w.u32(word);
}

std::vector<uint32_t> canonicalUniformWords(const ForgeUniforms &u) {
    std::set<size_t> padding = {offsetof(ForgeUniforms, cameraPosition) + 12, offsetof(ForgeUniforms, cameraForward) + 12,
                                offsetof(ForgeUniforms, cameraRight) + 12, offsetof(ForgeUniforms, cameraUp) + 12};
    for (size_t pad : sceneParamPadding()) padding.insert(offsetof(ForgeUniforms, scene) + pad);
    std::vector<uint32_t> words;
    appendWords(reinterpret_cast<const uint8_t *>(&u), offsetof(ForgeUniforms, scene) + sceneParamsSize, padding, words);
    return words;
}

SnapshotSections snapshotHashes(int absoluteFrame, const WorkerProject &project, const FrameRequest &request,
                                const EvaluatedFrame &frame) {
    SnapshotSections result;
    result.frame = absoluteFrame;
    auto section = [&](const char *name, const CanonicalWriter &w) {
        result.hashes.emplace_back(name, sha256Hex(w.bytes));
        result.bytes.emplace_back(name, w.bytes);
    };
    const int renderIndex = absoluteFrame - baseFrame(project);
    {
        CanonicalWriter w;
        w.i64(absoluteFrame);
        w.f64(request.beats);
        w.f64(static_cast<double>(renderIndex) / project.frameRate);
        w.f64(1 / project.frameRate);
        w.u32(request.frameIndex);
        w.u32(project.seed);
        w.i64(request.width);
        w.i64(request.height);
        w.boolean(request.isInteractivePreview);
        section("frame", w);
    }
    {
        CanonicalWriter w;
        w.boolean(false); // no native loop closure: CLI renders have loop closure off
        w.v2(frame.closureTime);
        section("loopClosure", w);
    }
    {
        CanonicalWriter w;
        canonicalSceneParams(frame.params, w);
        section("sceneParams", w);
    }
    {
        CanonicalWriter w;
        w.v3(frame.camera.position);
        w.v3(frame.camera.target);
        w.f32(frame.camera.fovDegrees);
        w.f32(frame.camera.roll);
        w.f32(frame.camera.focusDistance);
        w.f32(frame.camera.apertureRadius);
        section("camera", w);
    }
    return result;
}

} // namespace forge
