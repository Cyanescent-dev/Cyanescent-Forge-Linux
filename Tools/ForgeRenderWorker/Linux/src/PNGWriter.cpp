//
//  PNGWriter.cpp
//  Cyanescent Forge — Linux worker
//
#include "PNGWriter.hpp"

#include <zlib.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <unistd.h>

namespace forge {

namespace {

void be32(std::vector<uint8_t> &v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

void chunk(std::vector<uint8_t> &out, const char type[4], const std::vector<uint8_t> &data) {
    be32(out, static_cast<uint32_t>(data.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uLong crc = crc32(0L, out.data() + crcStart, static_cast<uInt>(out.size() - crcStart));
    be32(out, static_cast<uint32_t>(crc));
}

} // namespace

std::vector<uint8_t> encodePNG16(const std::vector<uint16_t> &rgba16, int width, int height) {
    if (width <= 0 || height <= 0 || rgba16.size() != static_cast<size_t>(width) * height * 4) {
        throw std::runtime_error("PNG encode: pixel buffer does not match the image size.");
    }
    // Scanlines: filter byte 0 (None) + RGB16 big-endian.
    const size_t rowBytes = static_cast<size_t>(width) * 6;
    std::vector<uint8_t> raw((rowBytes + 1) * height);
    for (int y = 0; y < height; ++y) {
        uint8_t *row = raw.data() + static_cast<size_t>(y) * (rowBytes + 1);
        row[0] = 0;
        for (int x = 0; x < width; ++x) {
            const uint16_t *p = rgba16.data() + (static_cast<size_t>(y) * width + x) * 4;
            for (int c = 0; c < 3; ++c) {
                row[1 + x * 6 + c * 2] = static_cast<uint8_t>(p[c] >> 8);
                row[1 + x * 6 + c * 2 + 1] = static_cast<uint8_t>(p[c] & 0xff);
            }
        }
    }
    uLongf compressedSize = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> compressed(compressedSize);
    if (compress2(compressed.data(), &compressedSize, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) {
        throw std::runtime_error("PNG encode: zlib compression failed.");
    }
    compressed.resize(compressedSize);

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> ihdr;
    be32(ihdr, static_cast<uint32_t>(width));
    be32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(16); // bit depth
    ihdr.push_back(2);  // colour type: truecolour (RGB)
    ihdr.push_back(0);  // compression
    ihdr.push_back(0);  // filter
    ihdr.push_back(0);  // interlace
    chunk(out, "IHDR", ihdr);
    chunk(out, "sRGB", {0}); // perceptual; the values are sRGB-encoded (forgeLinearToSRGB)
    chunk(out, "IDAT", compressed);
    chunk(out, "IEND", {});
    return out;
}

void writeFileAtomically(const std::string &path, const std::vector<uint8_t> &bytes) {
    const std::string temporary = path + ".tmp-" + std::to_string(static_cast<long>(getpid()));
    FILE *f = std::fopen(temporary.c_str(), "wb");
    if (!f) throw std::runtime_error("Cannot write " + temporary + ": " + std::strerror(errno));
    const size_t written = bytes.empty() ? 0 : std::fwrite(bytes.data(), 1, bytes.size(), f);
    const bool ok = written == bytes.size() && std::fflush(f) == 0 && fsync(fileno(f)) == 0;
    std::fclose(f);
    if (!ok || std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("Cannot write " + path + ": " + std::strerror(errno));
    }
}

} // namespace forge
