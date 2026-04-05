#pragma once

#include <cstdint>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Compute a 2D Euclidean signed distance field from an alpha mask.
// Returns one float per pixel: negative = inside solid, positive = outside.
std::vector<float> compute_sdf(uint32_t width, uint32_t height,
                                const std::vector<float>& alpha);

struct SdfJob {
    uint32_t           width, height;
    std::vector<float> alpha;  // per-pixel alpha [0,1]; > 0.5 = solid
};

struct SdfResult {
    uint32_t           width, height;
    std::vector<float> sdf;   // signed distance in pixels; negative = inside solid
};

static constexpr uint32_t SDF_STATIC_TAG = 0xFFFFFFFFu;

// Single worker thread that computes 2D Euclidean signed distance fields.
// submit() and poll() are thread-safe.
class SdfGenerator {
public:
    SdfGenerator();
    ~SdfGenerator();

    // Enqueue a job. Returns immediately.
    void submit(SdfJob job);

    // Drain completed results into cb, called on the invoking thread.
    // Call once per frame from the main thread.
    void poll(std::function<void(SdfResult&&)> cb);

    // Finish pending work and stop the worker thread.
    void shutdown();

private:
    void worker_loop();

    std::thread             _thread;
    std::mutex              _mutex;
    std::condition_variable _cv;
    std::queue<SdfJob>      _pending;
    std::vector<SdfResult>  _done;
    bool                    _stop{false};
};
