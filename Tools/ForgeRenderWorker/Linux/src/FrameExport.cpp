//
//  FrameExport.cpp
//  Cyanescent Forge — Linux worker
//
#include "FrameExport.hpp"

#include "PNGWriter.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace forge {

namespace {

double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

} // namespace

PNGExporter::PNGExporter(std::string directory, std::string prefix, int width, int height, int threads,
                         size_t maxQueued)
    : directory_(std::move(directory)), prefix_(std::move(prefix)), width_(width), height_(height),
      maxQueued_(maxQueued == 0 ? 1 : maxQueued) {
    for (int i = 0; i < threads; ++i) threads_.emplace_back([this] { work(); });
}

PNGExporter::~PNGExporter() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    space_.notify_all();
    for (auto &t : threads_) t.join();
}

std::string PNGExporter::path(int frame) const {
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%06d.png", prefix_.c_str(), frame);
    return directory_ + "/" + name;
}

PNGExportTimings PNGExporter::write(const Job &job) {
    PNGExportTimings t;
    t.frame = job.frame;
    const double start = now();
    t.queuedSeconds = start - job.submitted;
    const std::vector<uint8_t> png = encodePNG16(job.pixels, width_, height_);
    const double encoded = now();
    writeFileAtomically(path(job.frame), png);
    t.encodeSeconds = encoded - start;
    t.writeSeconds = now() - encoded;
    return t;
}

double PNGExporter::submit(int frame, std::vector<uint16_t> &&rgba16) {
    if (threads_.empty()) {
        const PNGExportTimings t = write(Job{frame, std::move(rgba16), now()});
        std::lock_guard<std::mutex> lock(mutex_);
        timings_.push_back(t);
        return 0;
    }
    const double start = now();
    std::unique_lock<std::mutex> lock(mutex_);
    space_.wait(lock, [&] { return error_ || queue_.size() + busy_ < maxQueued_; });
    if (error_) std::rethrow_exception(error_);
    const double blocked = now() - start;
    timings_.push_back(PNGExportTimings{frame, 0, 0, 0});
    queue_.push_back(Job{frame, std::move(rgba16), now()});
    lock.unlock();
    wake_.notify_one();
    return blocked;
}

void PNGExporter::work() {
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) return;
        Job job = std::move(queue_.front());
        queue_.pop_front();
        ++busy_;
        lock.unlock();
        PNGExportTimings t;
        std::exception_ptr failure;
        try {
            t = write(job);
        } catch (...) {
            failure = std::current_exception();
        }
        lock.lock();
        --busy_;
        if (failure && !error_) error_ = failure;
        for (auto &row : timings_) {
            if (row.frame == job.frame) row = t;
        }
        lock.unlock();
        space_.notify_all();
    }
}

void PNGExporter::finish() {
    std::unique_lock<std::mutex> lock(mutex_);
    space_.wait(lock, [&] { return queue_.empty() && busy_ == 0; });
    if (error_) std::rethrow_exception(error_);
}

std::vector<PNGExportTimings> PNGExporter::timings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timings_;
}

std::vector<std::string> mp4Arguments(const std::string &inputPattern, int startNumber, int frameCount,
                                      double frameRate, MP4Quality quality, const std::string &output) {
    // CyanescentForge/Core/MP4Export.swift, ForgeMP4Encoding.arguments.
    int crf = 18;
    const char *preset = "slow";
    switch (quality) {
    case MP4Quality::Standard: crf = 20; preset = "medium"; break;
    case MP4Quality::High: crf = 18; preset = "slow"; break;
    case MP4Quality::VeryHigh: crf = 15; preset = "slower"; break;
    }
    char rate[32];
    std::snprintf(rate, sizeof(rate), "%.8g", frameRate);
    return {"-hide_banner", "-nostats", "-v", "error", "-n",
            "-framerate", rate,
            "-start_number", std::to_string(startNumber),
            "-i", inputPattern,
            "-frames:v", std::to_string(frameCount),
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", std::to_string(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output};
}

std::string findFFmpeg() {
    const char *env = std::getenv("FORGE_FFMPEG");
    if (env && *env) return access(env, X_OK) == 0 ? env : "";
    const char *pathVariable = std::getenv("PATH");
    std::string path = pathVariable ? pathVariable : "/usr/local/bin:/usr/bin:/bin";
    size_t start = 0;
    while (start <= path.size()) {
        const size_t end = path.find(':', start);
        const std::string dir = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!dir.empty()) {
            const std::string candidate = dir + "/ffmpeg";
            struct stat st {};
            if (stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(candidate.c_str(), X_OK) == 0) {
                return candidate;
            }
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return "";
}

void runFFmpeg(const std::vector<std::string> &arguments) {
    const std::string ffmpeg = findFFmpeg();
    if (ffmpeg.empty()) {
        throw std::runtime_error("ffmpeg was not found (install it, e.g. apt-get install ffmpeg, or set FORGE_FFMPEG).");
    }
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(ffmpeg.c_str()));
    for (const auto &a : arguments) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error(std::string("Cannot start ffmpeg: ") + std::strerror(errno));
    if (pid == 0) {
        execv(ffmpeg.c_str(), argv.data());
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error(std::string("waitpid(ffmpeg): ") + std::strerror(errno));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("ffmpeg failed (" +
                                 (WIFEXITED(status) ? "exit status " + std::to_string(WEXITSTATUS(status))
                                                    : std::string("killed by a signal")) +
                                 ").");
    }
}

} // namespace forge
