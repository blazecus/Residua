#include "sdf_generator.h"
#include <cmath>
#include <limits>

// ─── 1D distance transform ────────────────────────────────────────────────────
// Felzenszwalb & Huttenlocher parabola-envelope method.
// f[i]: squared distance input (0.0 at seed pixels, INF elsewhere).
// d[i]: squared distance output to nearest seed.

static void dt1d(const float* f, float* d, int n) {
    constexpr float INF = 1e18f;

    std::vector<int>   v(n);
    std::vector<float> z(n + 1);

    int k = 0;
    v[0]  = 0;
    z[0]  = -INF;
    z[1]  =  INF;

    for (int q = 1; q < n; q++) {
        float s;
        do {
            int r = v[k];
            s = ((f[q] + (float)(q * q)) - (f[r] + (float)(r * r)))
                / (2.0f * (float)(q - r));
            if (s > z[k]) break;
            --k;
        } while (k >= 0);

        ++k;
        v[k]     = q;
        z[k]     = s;
        z[k + 1] = INF;
    }

    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < (float)q) ++k;
        int r = v[k];
        d[q]  = (float)((q - r) * (q - r)) + f[r];
    }
}

// ─── 2D Euclidean distance transform ─────────────────────────────────────────
// Returns sqrt(squared distance) from each pixel to the nearest seed pixel.

static std::vector<float> edt2d(const std::vector<bool>& mask, uint32_t w, uint32_t h) {
    constexpr float INF = 1e9f;

    std::vector<float> tmp(w * h);
    {
        std::vector<float> row_in(w), row_out(w);
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++)
                row_in[x] = mask[y * w + x] ? 0.0f : INF;
            dt1d(row_in.data(), row_out.data(), (int)w);
            for (uint32_t x = 0; x < w; x++)
                tmp[y * w + x] = row_out[x];
        }
    }

    std::vector<float> dist(w * h);
    {
        std::vector<float> col_in(h), col_out(h);
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t y = 0; y < h; y++)
                col_in[y] = tmp[y * w + x];
            dt1d(col_in.data(), col_out.data(), (int)h);
            for (uint32_t y = 0; y < h; y++)
                dist[y * w + x] = std::sqrt(col_out[y]);
        }
    }

    return dist;
}

// ─── Public free function ─────────────────────────────────────────────────────

std::vector<float> compute_sdf(uint32_t w, uint32_t h, const std::vector<float>& alpha) {
    uint32_t n = w * h;

    std::vector<bool> solid(n), empty(n);
    for (uint32_t i = 0; i < n; i++) {
        solid[i] = alpha[i] > 0.5f;
        empty[i] = !solid[i];
    }

    auto dist_to_solid = edt2d(solid, w, h);
    auto dist_to_empty = edt2d(empty, w, h);

    std::vector<float> sdf(n);
    for (uint32_t i = 0; i < n; i++)
        sdf[i] = solid[i] ? -dist_to_empty[i] : dist_to_solid[i];

    return sdf;
}

// ─── Worker thread ────────────────────────────────────────────────────────────

SdfGenerator::SdfGenerator() {
    _thread = std::thread(&SdfGenerator::worker_loop, this);
}

SdfGenerator::~SdfGenerator() {
    shutdown();
}

void SdfGenerator::submit(SdfJob job) {
    {
        std::lock_guard<std::mutex> lk(_mutex);
        _pending.push(std::move(job));
    }
    _cv.notify_one();
}

void SdfGenerator::poll(std::function<void(SdfResult&&)> cb) {
    std::vector<SdfResult> batch;
    {
        std::lock_guard<std::mutex> lk(_mutex);
        std::swap(batch, _done);
    }
    for (auto& r : batch)
        cb(std::move(r));
}

void SdfGenerator::shutdown() {
    {
        std::lock_guard<std::mutex> lk(_mutex);
        _stop = true;
    }
    _cv.notify_one();
    if (_thread.joinable())
        _thread.join();
}

void SdfGenerator::worker_loop() {
    for (;;) {
        SdfJob job;
        {
            std::unique_lock<std::mutex> lk(_mutex);
            _cv.wait(lk, [this] { return !_pending.empty() || _stop; });
            if (_stop && _pending.empty()) return;
            job = std::move(_pending.front());
            _pending.pop();
        }

        SdfResult result;
        result.width  = job.width;
        result.height = job.height;
        result.sdf    = compute_sdf(job.width, job.height, job.alpha);

        {
            std::lock_guard<std::mutex> lk(_mutex);
            _done.push_back(std::move(result));
        }
    }
}
