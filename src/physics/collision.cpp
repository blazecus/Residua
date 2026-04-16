#include "collision.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <limits>

// ─── SDF sampling ─────────────────────────────────────────────────────────────

static float sample_sdf(const RigidBody2& body, glm::vec2 img_pos)
{
    float px = img_pos.x;
    float py = img_pos.y;

    int x0 = (int)px, y0 = (int)py;
    int x1 = x0 + 1,  y1 = y0 + 1;

    int w = (int)body.sdf_w;
    int h = (int)body.sdf_h;

    auto clamp_idx = [&](int x, int y) -> float {
        if (x < 0 || y < 0 || x >= w || y >= h) return (float)(w + h);
        return body.sdf[y * w + x];
    };

    float fx = px - (float)x0;
    float fy = py - (float)y0;

    return clamp_idx(x0, y0) * (1.f - fx) * (1.f - fy)
         + clamp_idx(x1, y0) *        fx  * (1.f - fy)
         + clamp_idx(x0, y1) * (1.f - fx) *        fy
         + clamp_idx(x1, y1) *        fx  *        fy;
}

// Finite-difference gradient in image/local space (NOT yet world space).
static glm::vec2 sdf_gradient(const RigidBody2& body, glm::vec2 img_pos)
{
    float dx = sample_sdf(body, img_pos + glm::vec2(1.f, 0.f))
             - sample_sdf(body, img_pos - glm::vec2(1.f, 0.f));
    float dy = sample_sdf(body, img_pos + glm::vec2(0.f, 1.f))
             - sample_sdf(body, img_pos - glm::vec2(0.f, 1.f));
    return { dx * 0.5f, dy * 0.5f };
}

static glm::vec2 local_to_image(const RigidBody2& body, glm::vec2 local_pt)
{
    return local_pt + glm::vec2(body.sdf_w * 0.5f, body.sdf_h * 0.5f);
}

// ─── Manifold reduction ───────────────────────────────────────────────────────

std::vector<ManifoldPoint> reduce_manifold(std::vector<ManifoldPoint> points, int n)
{
    if ((int)points.size() <= n) return points;

    std::vector<ManifoldPoint> result;
    result.reserve(n);

    int seed = 0;
    for (int i = 1; i < (int)points.size(); i++)
        if (points[i].depth > points[seed].depth) seed = i;
    result.push_back(points[seed]);
    points.erase(points.begin() + seed);

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

// ─── Build manifold ───────────────────────────────────────────────────────────

struct Candidate {
    glm::vec2 world_pt;
    float     depth;
    glm::vec2 grad_local;  // gradient in ref body's local/image space
    float     ref_angle;   // ref body's world rotation, to convert grad to world space
    float     normal_sign; // +1 if ref=A, -1 if ref=B
};

static void collect_candidates(
    const RigidBody2& ref, const RigidBody2& other,
    float normal_sign,
    std::vector<Candidate>& out)
{
    int n = (int)other.shape.size();
    for (int vi = 0; vi < n; vi++) {
        glm::vec2 lv0 = other.shape[vi];
        glm::vec2 lv1 = other.shape[(vi + 1) % n];

        // Transform both endpoints to world space, then ref's image space.
        glm::vec2 w0 = glm::vec2(other.position) + glm::rotate(lv0, other.position.z);
        glm::vec2 w1 = glm::vec2(other.position) + glm::rotate(lv1, other.position.z);

        glm::vec2 img0 = local_to_image(ref,
            glm::rotate(w0 - glm::vec2(ref.position), -ref.position.z));
        glm::vec2 img1 = local_to_image(ref,
            glm::rotate(w1 - glm::vec2(ref.position), -ref.position.z));

        float d0 = sample_sdf(ref, img0);
        float d1 = sample_sdf(ref, img1);

        // Both outside — no contact on this edge.
        if (d0 >= 0.f && d1 >= 0.f) continue;

        glm::vec2 contact_img;
        float     depth;

        if ((d0 < 0.f) != (d1 < 0.f)) {
            // Sign change: interpolate to find the surface crossing.
            float t    = d0 / (d0 - d1);
            contact_img = img0 + t * (img1 - img0);
            depth       = std::max(-d0, -d1);
        } else {
            // Both inside: use the shallower endpoint (closest to surface).
            if (d0 > d1) { contact_img = img0; depth = -d0; }
            else          { contact_img = img1; depth = -d1; }
        }

        // Reconstruct world-space contact position from the image-space point.
        glm::vec2 ref_local = contact_img - glm::vec2(ref.sdf_w * 0.5f, ref.sdf_h * 0.5f);
        glm::vec2 world_pt  = glm::vec2(ref.position) + glm::rotate(ref_local, ref.position.z);

        out.push_back({ world_pt, depth,
                        sdf_gradient(ref, contact_img),
                        ref.position.z, normal_sign });
    }
}

std::vector<ManifoldPoint> build_manifold(const RigidBody2& a, const RigidBody2& b)
{
    std::vector<Candidate> candidates;
    collect_candidates(a, b, +1.f, candidates);
    collect_candidates(b, a, -1.f, candidates);

    if (candidates.empty()) return {};

    // Pick the shallowest candidate: it's closest to the contact surface,
    // so its gradient is least likely to have flipped through the body center.
    int best = 0;
    for (int i = 1; i < (int)candidates.size(); i++)
        if (candidates[i].depth < candidates[best].depth) best = i;

    const Candidate& c = candidates[best];
    float len = glm::length(c.grad_local);
    if (len < 1e-6f) return {};

    // Convert gradient from ref's local/image space to world space, then apply sign.
    glm::vec2 n = glm::rotate(c.grad_local / len, c.ref_angle) * c.normal_sign;

    // Guarantee n points from A toward B.
    glm::vec2 ab = glm::vec2(b.position) - glm::vec2(a.position);
    if (glm::length(ab) > 1e-6f && glm::dot(n, ab) < 0.f)
        n = -n;

    // Emit all penetrating candidates with the shared world-space normal.
    std::vector<ManifoldPoint> points;
    for (const Candidate& ci : candidates)
        points.push_back({ ci.world_pt, n, ci.depth });

    return points;
}
