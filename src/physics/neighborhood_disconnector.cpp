#include "neighborhood_disconnector.h"

#include <queue>
#include <algorithm>
#include <limits>

/**
* Full O(w*h) scan 
*/
static std::vector<PixelComponent> full_scan(const glm::vec4* pixels, uint32_t w, uint32_t h)
{
    std::vector<bool> visited(w * h, false);
    std::vector<PixelComponent> components;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t idx = y * w + x;
            if (visited[idx] || pixels[idx].a <= 0.f) continue;

            PixelComponent comp;
            comp.x0 = comp.y0 = std::numeric_limits<int>::max();
            comp.x1 = comp.y1 = std::numeric_limits<int>::min();

            std::queue<glm::ivec2> q;
            q.push({ (int)x, (int)y });
            visited[idx] = true;

            while (!q.empty()) {
                glm::ivec2 cur = q.front(); q.pop();
                int cx = cur.x, cy = cur.y;

                comp.pixels.push_back({ cx, cy });
                comp.x0 = std::min(comp.x0, cx);
                comp.y0 = std::min(comp.y0, cy);
                comp.x1 = std::max(comp.x1, cx);
                comp.y1 = std::max(comp.y1, cy);

                static constexpr int dxs[4] = { 1, -1, 0, 0 };
                static constexpr int dys[4] = { 0, 0, 1, -1 };
                for (int k = 0; k < 4; k++) {
                    int nx = cx + dxs[k], ny = cy + dys[k];
                    if (nx < 0 || ny < 0 || (uint32_t)nx >= w || (uint32_t)ny >= h) continue;
                    uint32_t nidx = (uint32_t)ny * w + (uint32_t)nx;
                    if (!visited[nidx] && pixels[nidx].a > 0.f) {
                        visited[nidx] = true;
                        q.push({ nx, ny });
                    }
                }
            }

            components.push_back(std::move(comp));
        }
    }

    return components;
}

/**
* given a shape with removed pixels, finds resulting shapes
* covers case with no new shapes, or any amount of new cases, through BFS flood fill near removed pixels 
*/
std::vector<PixelComponent> find_components(
    const glm::vec4* pixels, uint32_t w, uint32_t h,
    int ex0, int ey0, int ex1, int ey1)
{
    if (ex0 < 0 || ey0 < 0 || ex0 > ex1 || ey0 > ey1)
        return full_scan(pixels, w, h);

    // Collect solid pixels on the 1-pixel border ring just outside the erased bbox.
    std::vector<glm::ivec2> seeds;
    auto try_add = [&](int x, int y) {
        if (x < 0 || y < 0 || (uint32_t)x >= w || (uint32_t)y >= h) return;
        if (pixels[(uint32_t)y * w + (uint32_t)x].a > 0.f)
            seeds.push_back({ x, y });
    };
    for (int x = ex0 - 1; x <= ex1 + 1; x++) {
        try_add(x, ey0 - 1);  
        try_add(x, ey1 + 1); 
    }
    for (int y = ey0; y <= ey1; y++) {
        try_add(ex0 - 1, y); 
        try_add(ex1 + 1, y);  
    }

    if (seeds.empty()) {
        // No solid pixels adjacent to erased region — body may be fully erased.
        return full_scan(pixels, w, h);
    }

    // BFS from seeds[0], tracking only bbox (no pixel list needed for the no-split fast-path).
    std::vector<bool> visited(w * h, false);
    std::queue<glm::ivec2> q;

    auto enqueue = [&](int x, int y) {
        if (x < 0 || y < 0 || (uint32_t)x >= w || (uint32_t)y >= h) return;
        uint32_t idx = (uint32_t)y * w + (uint32_t)x;
        if (!visited[idx] && pixels[idx].a > 0.f) {
            visited[idx] = true;
            q.push({ x, y });
        }
    };

    PixelComponent comp;
    comp.x0 = comp.y0 = std::numeric_limits<int>::max();
    comp.x1 = comp.y1 = std::numeric_limits<int>::min();

    visited[(uint32_t)seeds[0].y * w + (uint32_t)seeds[0].x] = true;
    q.push(seeds[0]);

    while (!q.empty()) {
        glm::ivec2 cur = q.front(); q.pop();
        comp.x0 = std::min(comp.x0, cur.x);
        comp.y0 = std::min(comp.y0, cur.y);
        comp.x1 = std::max(comp.x1, cur.x);
        comp.y1 = std::max(comp.y1, cur.y);
        enqueue(cur.x + 1, cur.y);
        enqueue(cur.x - 1, cur.y);
        enqueue(cur.x, cur.y + 1);
        enqueue(cur.x, cur.y - 1);
    }

    // Check that every other seed was reached — if not, a split exists.
    for (size_t i = 1; i < seeds.size(); i++) {
        if (!visited[(uint32_t)seeds[i].y * w + (uint32_t)seeds[i].x])
            return full_scan(pixels, w, h);  // split — need full component info
    }

    // Single component confirmed
    return { comp };
}
