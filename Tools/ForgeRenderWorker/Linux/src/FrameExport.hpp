//
//  FrameExport.hpp
//  Cyanescent Forge — Linux worker
//
//  Export for the Linux worker, following Forge's own path: every frame is
//  written as a 16-bit sRGB PNG (the master output), and an MP4 is made
//  afterwards by handing the completed PNG sequence to ffmpeg with the exact
//  arguments Forge uses on macOS (CyanescentForge/Core/MP4Export.swift).
//
//  PNG encoding (zlib, CPU) runs on a small thread pool while the GPU renders
//  the next frame, so on a fast GPU the render loop is not serialised behind
//  deflate. A bounded queue keeps memory in check: when the encoders fall
//  behind, submit() blocks and the wait is reported.
//
#ifndef FORGE_FRAME_EXPORT_HPP
#define FORGE_FRAME_EXPORT_HPP

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace forge {

struct PNGExportTimings {
    int frame = 0;
    double encodeSeconds = 0;
    double writeSeconds = 0;
    double queuedSeconds = 0; // submitted -> an encoder picked it up
};

class PNGExporter {
public:
    /// `threads` 0 writes synchronously on the submitting thread.
    PNGExporter(std::string directory, std::string prefix, int width, int height, int threads, size_t maxQueued);
    ~PNGExporter();
    PNGExporter(const PNGExporter &) = delete;
    PNGExporter &operator=(const PNGExporter &) = delete;

    /// Queues (or, synchronously, writes) one frame. Returns the seconds the
    /// caller was blocked by a full queue. Rethrows an encoder's failure.
    double submit(int frame, std::vector<uint16_t> &&rgba16);
    /// Waits for every queued frame; rethrows the first failure.
    void finish();
    /// Per-frame timings, in submission order (complete after finish()).
    std::vector<PNGExportTimings> timings() const;
    std::string path(int frame) const;

private:
    struct Job {
        int frame;
        std::vector<uint16_t> pixels;
        double submitted;
    };
    void work();
    PNGExportTimings write(const Job &job);

    std::string directory_, prefix_;
    int width_, height_;
    size_t maxQueued_;
    std::vector<std::thread> threads_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, space_;
    std::deque<Job> queue_;
    size_t busy_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;
    std::vector<PNGExportTimings> timings_;
};

enum class MP4Quality { Standard, High, VeryHigh };

/// Forge's ForgeMP4Encoding.arguments (after the executable).
std::vector<std::string> mp4Arguments(const std::string &inputPattern, int startNumber, int frameCount,
                                      double frameRate, MP4Quality quality, const std::string &output);

/// Runs ffmpeg (FORGE_FFMPEG or `ffmpeg` on PATH) to completion. Throws on failure.
void runFFmpeg(const std::vector<std::string> &arguments);

/// ffmpeg's executable if found, else empty.
std::string findFFmpeg();

} // namespace forge

#endif // FORGE_FRAME_EXPORT_HPP
