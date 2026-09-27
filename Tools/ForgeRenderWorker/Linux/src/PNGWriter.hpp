//
//  PNGWriter.hpp
//  Cyanescent Forge — Linux worker
//
//  The Linux counterpart of ForgeImageWriter / encodeWorkerImage: a 16-bit
//  RGB PNG (alpha dropped, as CGImage noneSkipLast does) tagged sRGB, from
//  the rgba16 pixels the resolve pass produced. zlib does the deflate.
//  Samples are written big-endian as PNG requires, by shifting, so the host
//  byte order never matters.
//
#ifndef FORGE_PNG_WRITER_HPP
#define FORGE_PNG_WRITER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace forge {

/// Encodes to memory (timed separately from the disk write, as on macOS).
std::vector<uint8_t> encodePNG16(const std::vector<uint16_t> &rgba16, int width, int height);

/// Writes `bytes` to `path` atomically (temporary file in the same directory, then rename).
void writeFileAtomically(const std::string &path, const std::vector<uint8_t> &bytes);

} // namespace forge

#endif // FORGE_PNG_WRITER_HPP
