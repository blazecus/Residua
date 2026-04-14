#include "collision.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <array>
#include <limits>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static glm::vec2 world_vert(const RigidBody2& rb, uint32_t vi) {
    return glm::vec2(rb.position) + glm::rotate(rb.shape[vi], rb.position.z);
}

static std::pair<float, float> project_tri(const glm::vec2 v[3], glm::vec2 axis) {
    float d0 = glm::dot(v[0], axis);
    float d1 = glm::dot(v[1], axis);
    float d2 = glm::dot(v[2], axis);
    return { std::min({d0, d1, d2}), std::max({d0, d1, d2}) };
}

// ─── SAT ──────────────────────────────────────────────────────────────────────

struct SATResult {
    bool      separated;
    float     depth;
    glm::vec2 normal;  // from A toward B
};

static SATResult sat_triangles(const glm::vec2 a[3], const glm::vec2 b[3]) {
    float     min_depth = std::numeric_limits<float>::max();
    glm::vec2 best_normal{};

    glm::vec2 centA = (a[0] + a[1] + a[2]) / 3.f;
    glm::vec2 centB = (b[0] + b[1] + b[2]) / 3.f;

    const glm::vec2* tris[2] = { a, b };
    for (int t = 0; t < 2; t++) {
        const glm::vec2* v = tris[t];
        for (int e = 0; e < 3; e++) {
            glm::vec2 edge = v[(e + 1) % 3] - v[e];
            float len = glm::length(edge);
            if (len < 1e-6f) continue;
            glm::vec2 axis{ -edge.y / len, edge.x / len };

            auto [minA, maxA] = project_tri(a, axis);
            auto [minB, maxB] = project_tri(b, axis);

            if (maxA <= minB || maxB <= minA)
                return { true, 0.f, {} };

            float depth = std::min(maxA - minB, maxB - minA);
            if (depth < min_depth) {
                min_depth = depth;
                best_normal = axis;
                if (glm::dot(best_normal, centB - centA) < 0.f)
                    best_normal = -best_normal;
            }
        }
    }

    return { false, min_depth, best_normal };
}

// ─── Contact point generation ─────────────────────────────────────────────────

static std::vector<glm::vec2> clip(
    const std::vector<glm::vec2>& poly, glm::vec2 n, float d)
{
    std::vector<glm::vec2> out;
    for (size_t i = 0; i < poly.size(); i++) {
        glm::vec2 curr = poly[i];
        glm::vec2 next = poly[(i + 1) % poly.size()];
        float dc = glm::dot(curr, n) - d;
        float dn = glm::dot(next, n) - d;
        if (dc >= 0.f) out.push_back(curr);
        if ((dc >= 0.f) != (dn >= 0.f)) {
            float t = dc / (dc - dn);
            out.push_back(curr + t * (next - curr));
        }
    }
    return out;
}

static std::pair<int, int> best_edge(const glm::vec2 verts[3], glm::vec2 dir) {
    float best = -std::numeric_limits<float>::max();
    int   bi   = 0;
    for (int i = 0; i < 3; i++) {
        glm::vec2 edge = verts[(i + 1) % 3] - verts[i];
        glm::vec2 en   = glm::normalize(glm::vec2(-edge.y, edge.x));
        float     d    = glm::dot(en, dir);
        if (d > best) { best = d; bi = i; }
    }
    return { bi, (bi + 1) % 3 };
}

// Generate contact points for one penetrating triangle pair and append to `out`.
static void contact_points(
    const glm::vec2 a[3], const glm::vec2 b[3],
    glm::vec2 normal, std::vector<ManifoldPoint>& out)
{
    auto [ri0, ri1] = best_edge(a, normal);
    glm::vec2 ref0 = a[ri0], ref1 = a[ri1];
    glm::vec2 ref_edge_dir = glm::normalize(ref1 - ref0);
    glm::vec2 ref_normal   = glm::vec2(-ref_edge_dir.y, ref_edge_dir.x);
    float     ref_d        = glm::dot(ref_normal, ref0);

    auto [ii0, ii1] = best_edge(b, -normal);
    std::vector<glm::vec2> incident = { b[ii0], b[ii1] };

    incident = clip(incident,  ref_edge_dir, glm::dot( ref_edge_dir, ref0));
    if (incident.empty()) return;
    incident = clip(incident, -ref_edge_dir, glm::dot(-ref_edge_dir, ref1));
    if (incident.empty()) return;

    for (glm::vec2 p : incident) {
        float depth = ref_d - glm::dot(ref_normal, p);
        if (depth >= 0.f)
            out.push_back({ p, normal, depth });
    }
}

std::vector<ManifoldPoint> reduce_manifold(std::vector<ManifoldPoint> points, int n) {
    if ((int)points.size() <= n) return points;

    std::vector<ManifoldPoint> result;
    result.reserve(n);

    // Seed with the deepest point.
    int seed = 0;
    for (int i = 1; i < (int)points.size(); i++)
        if (points[i].depth > points[seed].depth) seed = i;
    result.push_back(points[seed]);
    points.erase(points.begin() + seed);

    // Greedily pick the point farthest from the already-selected set.
    while ((int)result.size() < n && !points.empty()) {
        float best_dist = -1.f;
        int   best_i    = 0;
        for (int i = 0; i < (int)points.size(); i++) {
            float min_dist = std::numeric_limits<float>::max();
            for (const ManifoldPoint& r : result) {
                glm::vec2 d = points[i].position - r.position;
                min_dist = std::min(min_dist, glm::dot(d, d));
            }
            if (min_dist > best_dist) { best_dist = min_dist; best_i = i; }
        }
        result.push_back(points[best_i]);
        points.erase(points.begin() + best_i);
    }

    return result;
}

std::vector<ManifoldPoint> build_manifold(const RigidBody2& a, const RigidBody2& b) {
    std::vector<ManifoldPoint> points;

    for (const Triangle& ta : a.triangles) {
        glm::vec2 wa[3] = {
            world_vert(a, ta.a), world_vert(a, ta.b), world_vert(a, ta.c)
        };
        for (const Triangle& tb : b.triangles) {
            glm::vec2 wb[3] = {
                world_vert(b, tb.a), world_vert(b, tb.b), world_vert(b, tb.c)
            };

            SATResult sat = sat_triangles(wa, wb);
            if (sat.separated) continue;

            contact_points(wa, wb, sat.normal, points);
        }
    }

    return points;
}
