#include "shape_gen.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <limits>

// ─── Marching Squares ─────────────────────────────────────────────────────────
//
// For a W×H alpha grid we iterate over (W-1)×(H-1) cells.
// Cell (col, row) has four corners:
//   TL = (col,   row  ),  TR = (col+1, row  )
//   BL = (col,   row+1),  BR = (col+1, row+1)
//
// Case index: bit3=TL bit2=TR bit1=BR bit0=BL  (1 = solid)
//
// Edges within a cell:
//   0 = top    (TL→TR),  1 = right (TR→BR)
//   2 = bottom (BL→BR),  3 = left  (TL→BL)
//
// kSegTable[case] lists up to two unordered edge pairs {e0,e1} and {e2,e3}.
// -1 means the slot is unused.
// Saddle cases (5 and 10) use the simpler fixed resolution (no centre sample).

static const int8_t kSegTable[16][4] = {
    {-1, -1, -1, -1},  //  0: (empty)
    { 2,  3, -1, -1},  //  1: BL
    { 1,  2, -1, -1},  //  2: BR
    { 1,  3, -1, -1},  //  3: BL+BR
    { 0,  1, -1, -1},  //  4: TR
    { 0,  1,  2,  3},  //  5: TR+BL  (saddle: two islands)
    { 0,  2, -1, -1},  //  6: TR+BR
    { 0,  3, -1, -1},  //  7: TR+BR+BL
    { 0,  3, -1, -1},  //  8: TL
    { 0,  2, -1, -1},  //  9: TL+BL
    { 0,  3,  1,  2},  // 10: TL+BR  (saddle: two islands)
    { 0,  1, -1, -1},  // 11: TL+BR+BL
    { 1,  3, -1, -1},  // 12: TL+TR
    { 1,  2, -1, -1},  // 13: TL+TR+BL
    { 2,  3, -1, -1},  // 14: TL+TR+BR
    {-1, -1, -1, -1},  // 15: (full)
};

// Compute the position of the contour crossing on a given edge of cell (col,row).
// Interpolates linearly between the two corner values to find where alpha==threshold.
static glm::vec2 edge_point(int edge, int col, int row,
                             float vTL, float vTR, float vBR, float vBL,
                             float threshold)
{
    // t in [0,1]: how far along the edge the crossing sits.
    auto lerp_t = [&](float va, float vb) -> float {
        float d = vb - va;
        return (std::abs(d) < 1e-6f) ? 0.5f : std::clamp((threshold - va) / d, 0.f, 1.f);
    };

    switch (edge) {
        case 0: { float t = lerp_t(vTL, vTR); return { col + t,         (float)row     }; }
        case 1: { float t = lerp_t(vTR, vBR); return { (float)(col + 1), row + t       }; }
        case 2: { float t = lerp_t(vBL, vBR); return { col + t,         (float)(row+1) }; }
        case 3: { float t = lerp_t(vTL, vBL); return { (float)col,       row + t       }; }
        default: return {};
    }
}

std::vector<glm::vec2> marching_squares(
    uint32_t width, uint32_t height,
    const std::vector<glm::vec4>& img,
    float threshold)
{
    assert(img.size() == (size_t)width * height);

    struct Seg { glm::vec2 a, b; };
    std::vector<Seg> segs;
    segs.reserve((width + height) * 2);

    // Pixels outside the image are treated as transparent (0).
    // This ensures solid pixels at the image boundary generate proper edge segments.
    auto v = [&](int x, int y) -> float {
        if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) return 0.f;
        return img[y * width + x].w;
    };

    // Iterate over a virtual padded grid: one extra cell on every side.
    for (int row = -1; row < (int)height; row++) {
        for (int col = -1; col < (int)width; col++) {
            float vTL = v(col,   row);
            float vTR = v(col+1, row);
            float vBR = v(col+1, row+1);
            float vBL = v(col,   row+1);

            int idx =
                ((vTL > threshold) ? 8 : 0) |
                ((vTR > threshold) ? 4 : 0) |
                ((vBR > threshold) ? 2 : 0) |
                ((vBL > threshold) ? 1 : 0);

            const int8_t* s = kSegTable[idx];
            if (s[0] >= 0)
                segs.push_back({ edge_point(s[0], col, row, vTL, vTR, vBR, vBL, threshold),
                                  edge_point(s[1], col, row, vTL, vTR, vBR, vBL, threshold) });
            if (s[2] >= 0)
                segs.push_back({ edge_point(s[2], col, row, vTL, vTR, vBR, vBL, threshold),
                                  edge_point(s[3], col, row, vTL, vTR, vBR, vBL, threshold) });
        }
    }

    std::vector<glm::vec2> edges;
    for (Seg& seg : segs) {
        edges.push_back(seg.a);
        edges.push_back(seg.b);
    }

    return edges;
}

// ─── Douglas–Peucker ─────────────────────────────────────────────────────────

static float pt_seg_dist_sq(glm::vec2 p, glm::vec2 a, glm::vec2 b)
{
    glm::vec2 ab = b - a;
    float len2 = glm::dot(ab, ab);
    if (len2 < 1e-12f) {
        glm::vec2 ap = p - a;
        return glm::dot(ap, ap);
    }
    float t = std::clamp(glm::dot(p - a, ab) / len2, 0.f, 1.f);
    glm::vec2 proj = a + t * ab;
    glm::vec2 diff = p - proj;
    return glm::dot(diff, diff);
}

static void dp_recurse(const std::vector<glm::vec2>& pts,
                        int s, int e, float eps2,
                        std::vector<bool>& keep)
{
    if (e <= s + 1) return;

    float max_d2 = 0.f;
    int   max_i  = s;
    for (int i = s + 1; i < e; i++) {
        float d2 = pt_seg_dist_sq(pts[i], pts[s], pts[e]);
        if (d2 > max_d2) { max_d2 = d2; max_i = i; }
    }

    if (max_d2 > eps2) {
        keep[max_i] = true;
        dp_recurse(pts, s,     max_i, eps2, keep);
        dp_recurse(pts, max_i, e,     eps2, keep);
    }
}

std::vector<glm::vec2> douglas_peucker(
    const std::vector<glm::vec2>& contour, float epsilon)
{
    if (contour.size() < 3) return contour;

    // Treat the closed polygon as an open polyline by appending the first vertex.
    std::vector<glm::vec2> pts = contour;

    pts.push_back(contour[0]);

    std::vector<bool> keep(pts.size(), false);
    keep.front() = true;
    keep.back()  = true;

    dp_recurse(pts, 0, (int)pts.size() - 1, epsilon * epsilon, keep);

    std::vector<glm::vec2> result;
    // Skip the duplicated closing vertex (last element).
    for (size_t i = 0; i + 1 < pts.size(); i++)
        if (keep[i]) result.push_back(pts[i]);

    return result;
}

// ─── Segment stitching ────────────────────────────────────────────────────────

// Quantise a vertex to a fixed-point key so that endpoints computed from
// the same marching-squares interpolation compare equal.
struct IVec2 {
    int32_t x, y;
    bool operator==(const IVec2& o) const { return x == o.x && y == o.y; }
};
struct IVec2Hash {
    size_t operator()(const IVec2& k) const {
        return std::hash<int32_t>()(k.x) ^ (std::hash<int32_t>()(k.y) * 2654435761u);
    }
};
static IVec2 quantize(glm::vec2 p) {
    // Scale by 1024 so that sub-pixel interpolated points get distinct keys.
    return { (int32_t)std::round(p.x * 1024.f),
             (int32_t)std::round(p.y * 1024.f) };
}

std::vector<std::vector<glm::vec2>> stitch_contours(
    const std::vector<glm::vec2>& segments)
{
    size_t n = segments.size() / 2;
    if (n == 0) return {};

    // Build adjacency: endpoint key → list of (segment index, which end).
    struct HalfEdge { size_t seg; bool is_a; };
    std::unordered_map<IVec2, std::vector<HalfEdge>, IVec2Hash> adj;
    adj.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        adj[quantize(segments[i * 2    ])].push_back({i, true });
        adj[quantize(segments[i * 2 + 1])].push_back({i, false});
    }

    std::vector<bool> used(n, false);
    std::vector<std::vector<glm::vec2>> contours;

    for (size_t start = 0; start < n; start++) {
        if (used[start]) continue;

        std::vector<glm::vec2> contour;
        used[start] = true;
        contour.push_back(segments[start * 2    ]);
        contour.push_back(segments[start * 2 + 1]);

        bool closed = false;
        while (!closed) {
            IVec2 tail_key = quantize(contour.back());

            // Close the loop if the tail reaches the start.
            if (contour.size() > 2 && tail_key == quantize(contour.front())) {
                contour.pop_back();
                closed = true;
                break;
            }

            // Walk to the next unused segment connected to the tail.
            auto it = adj.find(tail_key);
            if (it == adj.end()) break;

            bool found = false;
            for (HalfEdge& he : it->second) {
                if (used[he.seg]) continue;
                used[he.seg] = true;
                // Tail matches one end of this segment — append the other end.
                contour.push_back(he.is_a ? segments[he.seg * 2 + 1]
                                          : segments[he.seg * 2    ]);
                found = true;
                break;
            }
            if (!found) break;
        }

        if (contour.size() >= 3)
            contours.push_back(std::move(contour));
    }

    return contours;
}

std::vector<glm::vec2> generate_shape(
    uint32_t width, uint32_t height,
    const std::vector<glm::vec4>& img,
    float threshold,
    float epsilon)
{
    std::vector<glm::vec2> ms = marching_squares(width, height, img, threshold);
    if (ms.empty()) return {};

    auto contours = stitch_contours(ms);
    if (contours.empty()) return {};

    // Simplify each contour with DP, then return the one with the most vertices
    // (largest boundary = outer hull of the shape).
    std::vector<glm::vec2> best;
    for (auto& c : contours) {
        auto simplified = douglas_peucker(c, epsilon);
        if (simplified.size() > best.size())
            best = std::move(simplified);
    }
    return best;
}


static void bfs_distance(uint32_t w, uint32_t h,
                          const std::vector<bool>& seed,
                          std::vector<float>& out_dist)
{
    const float INF = std::numeric_limits<float>::max();
    out_dist.assign(w * h, INF);

    // dx/dy/dd for 8 neighbors
    static const int   dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int   dy[8] = {-1,-1,-1,  0, 0,  1, 1, 1};
    static const float dd[8] = { 1.41421356f, 1.f, 1.41421356f,
                                  1.f,         1.f,
                                  1.41421356f, 1.f, 1.41421356f };

    using Entry = std::pair<float, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    for (uint32_t i = 0; i < w * h; i++) {
        if (seed[i]) {
            out_dist[i] = 0.f;
            pq.push({0.f, i});
        }
    }

    while (!pq.empty()) {
        auto [d, idx] = pq.top(); pq.pop();
        if (d > out_dist[idx]) continue;

        int x = (int)(idx % w);
        int y = (int)(idx / w);

        for (int k = 0; k < 8; k++) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (nx < 0 || ny < 0 || nx >= (int)w || ny >= (int)h) continue;

            uint32_t ni = (uint32_t)(ny * (int)w + nx);
            float    nd = d + dd[k];
            if (nd < out_dist[ni]) {
                out_dist[ni] = nd;
                pq.push({nd, ni});
            }
        }
    }
}

std::vector<float> generate_sdf(const LoadedBodyImage& img)
{
    uint32_t w = img.width;
    uint32_t h = img.height;
    uint32_t n = w * h;

    std::vector<bool> inside(n);
    for (uint32_t i = 0; i < n; i++)
        inside[i] = img.pixels[i].w > 0.5f;

    // Seeds: every pixel that borders a pixel of the opposite type (4-connectivity).
    // All such pixels are on the shape boundary and get distance 0.
    static const int dx4[4] = {-1, 1, 0, 0};
    static const int dy4[4] = { 0, 0,-1, 1};

    std::vector<bool> seed(n, false);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t i   = y * w + x;
            bool     in  = inside[i];
            for (int k = 0; k < 4; k++) {
                int nx = (int)x + dx4[k];
                int ny = (int)y + dy4[k];
                bool neighbor_in = (nx >= 0 && ny >= 0 && nx < (int)w && ny < (int)h)
                                   ? inside[ny * w + nx] : false;
                if (in != neighbor_in) { seed[i] = true; break; }
            }
        }
    }

    // One BFS gives unsigned distance-to-boundary for every pixel.
    std::vector<float> dist;
    bfs_distance(w, h, seed, dist);

    // Sign: negative inside the shape, positive outside.
    std::vector<float> sdf(n);
    for (uint32_t i = 0; i < n; i++)
        sdf[i] = inside[i] ? -dist[i] : dist[i];

    return sdf;
}