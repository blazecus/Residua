#include "collision.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <iostream>
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

// Generate contact points for one penetrating triangle pair and append to `out`.
// Uses the support-point of A along the SAT normal as the depth reference so that
// the result is independent of how A is triangulated (avoids the "diagonal edge"
// mis-classification that occurs when A is a rectangle split into two triangles).
static void contact_points(
    const glm::vec2 a[3], const glm::vec2 b[3],
    glm::vec2 normal, std::vector<ManifoldPoint>& out)
{
    // A's support point in the direction of normal (the surface of A most facing B).
    float support = -1e20f;
    for (int i = 0; i < 3; i++)
        support = std::max(support, glm::dot(a[i], normal));

    // Each vertex of B that lies past A's support plane is a contact.
    for (int i = 0; i < 3; i++) {
        float depth = support - glm::dot(b[i], normal);
        if (depth > 0.f)
            out.push_back({ b[i], normal, depth });
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
