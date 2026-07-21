#include "physics_engine.h"
#include "neighborhood_disconnector.h"
#include "joint.h"
#include "spring.h"
#include "../renderer/residua_engine.h"
#include <glm/gtx/rotate_vector.hpp>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

// -- Jagged line-cut fracture pattern -----------------------------------------
/**
* Fractures a shape upon a fracture event
* crack forms originating from contact point towards the closest edge point in a cone in the normal direction
* applies a jagged edge to this line
* if the resulting shape is estimated to be too small, instead cut from the thickest portion to the closest edge point 
*/

static uint32_t g_fracture_seed = 0xAEDBFEADu;

static constexpr float MIN_FRAGMENT_AREA = 30.f;
static constexpr int   MAX_FRAGMENTS     = 5;

static constexpr float MIN_VIABLE_FRAGMENT_AREA = 20.f;
static constexpr float MIN_BODY_AREA_TO_FRACTURE = 60.f;

/**
* 4-connected flood fill over the body's solid pixels, refusing to cross
* barrier cells. Barrier cells (like non-solid ones) simply aren't included
* in any component, which is what gives the crack a visible gap.
*/
static std::vector<PixelComponent> components_from_barrier(
    const RigidBody& rb, const std::vector<uint8_t>& barrier, uint32_t bw, uint32_t bh)
{
    std::vector<PixelComponent> out;
    std::vector<bool> visited(bw * bh, false);

    auto passable = [&](uint32_t idx) {
        return rb.sprite.pixels[idx].a > 0.f && !barrier[idx];
    };

    for (uint32_t y = 0; y < bh; y++) {
        for (uint32_t x = 0; x < bw; x++) {
            uint32_t idx = y * bw + x;
            if (visited[idx] || !passable(idx)) continue;

            PixelComponent comp;
            comp.x0 = comp.y0 = std::numeric_limits<int>::max();
            comp.x1 = comp.y1 = std::numeric_limits<int>::min();

            std::queue<glm::ivec2> q;
            q.push({ (int)x, (int)y });
            visited[idx] = true;

            while (!q.empty()) {
                glm::ivec2 cur = q.front(); q.pop();
                comp.pixels.push_back(cur);
                comp.x0 = std::min(comp.x0, cur.x);
                comp.y0 = std::min(comp.y0, cur.y);
                comp.x1 = std::max(comp.x1, cur.x);
                comp.y1 = std::max(comp.y1, cur.y);

                static constexpr int dxs[4] = { 1, -1, 0, 0 };
                static constexpr int dys[4] = { 0, 0, 1, -1 };
                for (int k = 0; k < 4; k++) {
                    int nx = cur.x + dxs[k], ny = cur.y + dys[k];
                    if (nx < 0 || ny < 0 || (uint32_t)nx >= bw || (uint32_t)ny >= bh) continue;
                    uint32_t nidx = (uint32_t)ny * bw + (uint32_t)nx;
                    if (!visited[nidx] && passable(nidx)) {
                        visited[nidx] = true;
                        q.push({ nx, ny });
                    }
                }
            }

            out.push_back(std::move(comp));
        }
    }

    return out;
}

/**
* jagged edge noise
*/
static float value_noise(glm::vec2 p, uint32_t seed)
{
    auto hash = [seed](int x, int y) -> float {
        uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + seed;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= h >> 16;
        return (float)(h & 0xFFFFFFu) / (float)0xFFFFFFu;
    };

    int x0 = (int)std::floor(p.x), y0 = (int)std::floor(p.y);
    float fx = p.x - (float)x0, fy = p.y - (float)y0;
    float v00 = hash(x0,     y0),     v10 = hash(x0 + 1, y0);
    float v01 = hash(x0, y0 + 1), v11 = hash(x0 + 1, y0 + 1);
    float sx = fx * fx * (3.f - 2.f * fx);
    float sy = fy * fy * (3.f - 2.f * fy);
    float a = v00 + (v10 - v00) * sx;
    float b = v01 + (v11 - v01) * sx;
    return a + (b - a) * sy;
}

struct PolyHit {
    int       edge{0};
    float     t{0.f};
    glm::vec2 pos{0.f};
};

/**
* finds nearest point on edges of shape
*/
static PolyHit nearest_point_on_polygon(const std::vector<glm::vec2>& poly, glm::vec2 p)
{
    PolyHit best;
    float best_d2 = std::numeric_limits<float>::max();
    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        glm::vec2 a = poly[i], b = poly[(i + 1) % n];
        glm::vec2 ab = b - a;
        float len2 = glm::dot(ab, ab);
        float t = (len2 > 1e-12f) ? std::clamp(glm::dot(p - a, ab) / len2, 0.f, 1.f) : 0.f;
        glm::vec2 pos = a + t * ab;
        float d2 = glm::dot(p - pos, p - pos);
        if (d2 < best_d2) { best_d2 = d2; best = { i, t, pos }; }
    }
    return best;
}

/**
 * estimate area of partition from a cut into a polygon
 */
static float chord_area(
    const std::vector<glm::vec2>& poly, int edge0, glm::vec2 pos0, int edge1, glm::vec2 pos1)
{
    int n = (int)poly.size();
    std::vector<glm::vec2> sub;
    sub.reserve(n + 2);
    sub.push_back(pos0);

    int i = (edge0 + 1) % n;
    for (;;) {
        sub.push_back(poly[i]);
        if (i == edge1) break;
        i = (i + 1) % n;
    }
    sub.push_back(pos1);

    float area2 = 0.f;
    int m = (int)sub.size();
    for (int k = 0; k < m; k++) {
        glm::vec2 a = sub[k], b = sub[(k + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    return std::abs(area2) * 0.5f;
}

/**
 * Cheaply estimates whether a cut from p0 to p1 would leave both sides at
 * or above MIN_FRAGMENT_AREA, by walking the polygon boundary
 */
static bool cut_area_is_viable(
    const std::vector<glm::vec2>& shape, const PolyHit& p0, const PolyHit& p1, float total_area)
{
    float side_area  = chord_area(shape, p0.edge, p0.pos, p1.edge, p1.pos);
    float other_area = total_area - side_area;
    return std::min(side_area, other_area) >= MIN_FRAGMENT_AREA;
}

/**
 * merges edge case fractures to enforce limits (size limits etc)
 * not used by impact fracture events
 */
static void enforce_fragment_limits(
    std::vector<PixelComponent>& components, uint32_t bw, uint32_t bh, int max_frags)
{
    for (;;) {
        if (components.size() <= 1) break;
        bool over_budget = (int)components.size() > max_frags;

        int    smallest_i = -1;
        size_t smallest_n = 0;
        for (int i = 0; i < (int)components.size(); i++) {
            bool candidate = over_budget || (float)components[i].pixels.size() < MIN_FRAGMENT_AREA;
            if (candidate && (smallest_i < 0 || components[i].pixels.size() < smallest_n)) {
                smallest_i = i;
                smallest_n = components[i].pixels.size();
            }
        }
        if (smallest_i < 0) break;

        std::vector<int> owner(bw * bh, -1);
        for (int i = 0; i < (int)components.size(); i++)
            for (auto& p : components[i].pixels)
                owner[(uint32_t)p.y * bw + (uint32_t)p.x] = i;

        std::unordered_map<int, int> border_count;
        static constexpr int dxs[4] = { 1, -1, 0, 0 };
        static constexpr int dys[4] = { 0, 0, 1, -1 };
        for (auto& p : components[smallest_i].pixels) {
            for (int k = 0; k < 4; k++) {
                int nx = p.x + dxs[k], ny = p.y + dys[k];
                if (nx < 0 || ny < 0 || (uint32_t)nx >= bw || (uint32_t)ny >= bh) continue;
                int o = owner[(uint32_t)ny * bw + (uint32_t)nx];
                if (o >= 0 && o != smallest_i) border_count[o]++;
            }
        }

        if (border_count.empty()) {
            // Fully isolated speck (shouldn't normally happen) -- drop it
            components.erase(components.begin() + smallest_i);
            continue;
        }

        int merge_into = std::max_element(border_count.begin(), border_count.end(),
            [](auto& a, auto& b) { return a.second < b.second; })->first;

        auto& dst = components[merge_into];
        auto& src = components[smallest_i];
        dst.pixels.insert(dst.pixels.end(), src.pixels.begin(), src.pixels.end());
        dst.x0 = std::min(dst.x0, src.x0);
        dst.y0 = std::min(dst.y0, src.y0);
        dst.x1 = std::max(dst.x1, src.x1);
        dst.y1 = std::max(dst.y1, src.y1);
        components.erase(components.begin() + smallest_i);
    }
}

/**
* cone shape from contact point to find the closest point on other side to form the crack
*/
static bool nearest_point_in_cone(
    const std::vector<glm::vec2>& poly, glm::vec2 origin, int skip_edge,
    float base_angle, float cone_half_angle, PolyHit& out)
{
    constexpr float PI = 3.14159265f;
    int   n       = (int)poly.size();
    float best_d2 = std::numeric_limits<float>::max();
    bool  found   = false;

    for (int i = 0; i < n; i++) {
        if (i == skip_edge) continue;
        glm::vec2 a = poly[i], b = poly[(i + 1) % n];
        glm::vec2 ab = b - a;
        float len2 = glm::dot(ab, ab);
        float t = (len2 > 1e-12f) ? std::clamp(glm::dot(origin - a, ab) / len2, 0.f, 1.f) : 0.f;
        glm::vec2 q = a + t * ab;

        glm::vec2 to_q = q - origin;
        float d2 = glm::dot(to_q, to_q);
        if (d2 < 1.f) continue;  // too close to the entry point itself to matter

        float diff = std::atan2(to_q.y, to_q.x) - base_angle;
        diff = std::fmod(diff + PI, 2.f * PI);
        if (diff < 0.f) diff += 2.f * PI;
        diff -= PI;  // wrapped to [-pi, pi]
        if (std::abs(diff) > cone_half_angle) continue;

        if (d2 < best_d2) { best_d2 = d2; out = { i, t, q }; found = true; }
    }
    return found;
}

/**
 * split the shape along a given line
 * uses a jagged edge iwth noise
 */
static std::vector<PixelComponent> partition_along_stroke(
    const RigidBody& rb, uint32_t bw, uint32_t bh,
    glm::vec2 P0, glm::vec2 P1, uint32_t& rng)
{
    auto frand = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return (float)(rng >> 8) / (float)(1u << 24);
    };

    glm::vec2 cut = P1 - P0;
    float cut_len = glm::length(cut);
    if (cut_len < 1.f) return {};  // degenerate, too short to matter
    glm::vec2 cut_dir    = cut / cut_len;
    glm::vec2 cut_normal = { -cut_dir.y, cut_dir.x };

    // jagged barrier mask method
    // finds closest edge and stops to not cut possible further edges in concave shape
    constexpr float JAG_AMPLITUDE_1  = 4.0f;
    constexpr float JAG_FREQUENCY_1  = 1.f / 7.f;
    constexpr float JAG_AMPLITUDE_2  = 2.0f;
    constexpr float JAG_FREQUENCY_2  = 1.f / 3.f;
    constexpr float STROKE_RADIUS    = 0.75f; 
    glm::vec2 phase{ frand() * 1000.f, frand() * 1000.f };

    std::vector<uint8_t> barrier(bw * bh, 0);
    auto mark_pixel = [&](int px, int py) {
        if (px < 0 || py < 0 || px >= (int)bw || py >= (int)bh) return;
        barrier[(uint32_t)py * bw + (uint32_t)px] = 1;
    };
    // create barrier
    auto mark_disc = [&](float cx, float cy, float radius) {
        int r_ceil = (int)std::ceil(radius);
        int icx = (int)std::floor(cx), icy = (int)std::floor(cy);
        for (int oy = -r_ceil; oy <= r_ceil; oy++) {
            for (int ox = -r_ceil; ox <= r_ceil; ox++) {
                int px = icx + ox, py = icy + oy;
                float dx = (px + 0.5f) - cx, dy = (py + 0.5f) - cy;
                if (dx * dx + dy * dy <= radius * radius) mark_pixel(px, py);
            }
        }
    };

    int n_samples = std::max(1, (int)std::ceil(cut_len / 0.5f));
    for (int s = 0; s <= n_samples; s++) {
        float t = (float)s / n_samples;
        glm::vec2 base = P0 + cut * t;
        float n1 = value_noise(base * JAG_FREQUENCY_1 + phase, g_fracture_seed) - 0.5f;
        float n2 = value_noise(base * JAG_FREQUENCY_2 + phase * 1.7f, g_fracture_seed + 1u) - 0.5f;
        float offset = n1 * 2.f * JAG_AMPLITUDE_1 + n2 * 2.f * JAG_AMPLITUDE_2;
        glm::vec2 p = base + cut_normal * offset;
        mark_disc(p.x, p.y, STROKE_RADIUS);
    }

    // apply barrier and retrieve new components
    // also check is valid size
    auto components = components_from_barrier(rb, barrier, bw, bh);
    if (components.size() < 2) return {};
    for (auto& c : components)
        if ((float)c.pixels.size() < MIN_VIABLE_FRAGMENT_AREA) return {};
    return components;
}

/**
 * create a jagged crack and cast it across the shape to create separated objects
 */
static std::vector<PixelComponent> line_crack_partition(
    const RigidBody& rb, uint32_t bw, uint32_t bh,
    glm::vec2 contact_local, uint32_t& rng)
{
    if (rb.shape.size() < 3) return {};

    float total_area = rb.mass / rb.density;
    if (total_area < MIN_BODY_AREA_TO_FRACTURE) return {};

    // get an anchor point then find nearest polygon point to get direction of crack
    PolyHit p0 = nearest_point_on_polygon(rb.shape, contact_local);
    glm::vec2 edge_a   = rb.shape[p0.edge];
    glm::vec2 edge_b   = rb.shape[(p0.edge + 1) % rb.shape.size()];
    glm::vec2 edge_dir = edge_b - edge_a;
    glm::vec2 normal{ -edge_dir.y, edge_dir.x };
    if (glm::dot(normal, normal) > 1e-9f) normal = glm::normalize(normal);
    // rb.shape is stored COM-relative (origin at the centroid), so whichever
    // perpendicular points back toward the origin is the inward one.
    if (glm::dot(normal, -p0.pos) < 0.f) normal = -normal;
    float base_angle = std::atan2(normal.y, normal.x);

    constexpr float CONE_HALF_ANGLE = 1.0f;  // rad
    PolyHit p1;
    if (!nearest_point_in_cone(rb.shape, p0.pos, p0.edge, base_angle, CONE_HALF_ANGLE, p1))
        return {};

    // Reject cheaply (no pixel work at all) if this cut clearly wouldn't
    // leave both sides big enough, before paying for the full flood fill.
    if (!cut_area_is_viable(rb.shape, p0, p1, total_area)) return {};

    // Convert both cut points from COM-relative shape space to body-pixel
    // space (same frame as the sprite).
    glm::vec2 half(bw * 0.5f, bh * 0.5f);
    return partition_along_stroke(rb, bw, bh,
        p0.pos + rb.com_local + half, p1.pos + rb.com_local + half, rng);
}

/**
 * fallback case for small partitions from contact point crack
 * instead builds a crack from the thickest point in the sdf to the closest edge
 */
static std::vector<PixelComponent> thickest_point_partition(
    const RigidBody& rb, uint32_t bw, uint32_t bh, uint32_t& rng)
{
    if (rb.shape.size() < 3) return {};

    float total_area = rb.mass / rb.density;
    if (total_area < MIN_BODY_AREA_TO_FRACTURE) return {};

    glm::vec2 half(bw * 0.5f, bh * 0.5f);
    glm::vec2 origin_local = rb.thickest_px - rb.com_local - half;

    // First edge: the single closest point on the boundary to the thickest
    // point, in any direction.
    PolyHit p0 = nearest_point_on_polygon(rb.shape, origin_local);

    // Second edge: closest point within a cone around the direction that
    // continues on through the thickest point and out the far side, rather
    // than just finding another nearby point back toward p0.
    glm::vec2 away_dir = origin_local - p0.pos;
    if (glm::dot(away_dir, away_dir) < 1e-9f) return {};
    float base_angle = std::atan2(away_dir.y, away_dir.x);

    constexpr float CONE_HALF_ANGLE = 1.0f;  // rad
    PolyHit p1;
    if (!nearest_point_in_cone(rb.shape, origin_local, p0.edge, base_angle, CONE_HALF_ANGLE, p1))
        return {};

    if (!cut_area_is_viable(rb.shape, p0, p1, total_area)) return {};

    return partition_along_stroke(rb, bw, bh,
        p0.pos + rb.com_local + half, p1.pos + rb.com_local + half, rng);
}

/**
 * anchor points need to be created or adapted
 */
static uint32_t remap_anchor(
    glm::vec2&                          r_local,
    uint32_t                            old_id,
    const std::vector<PixelComponent>&  components,
    const std::vector<int>&             comp_map,
    const std::vector<uint32_t>&        new_ids,
    const std::vector<glm::vec2>&       new_com_local_arr,
    glm::vec2 orig_com, uint32_t bw, uint32_t bh)
{
    // Convert anchor from old body-local frame -> original image pixel space
    float img_x = r_local.x + orig_com.x + bw * 0.5f - 0.5f;
    float img_y = r_local.y + orig_com.y + bh * 0.5f - 0.5f;
    int   ix    = std::clamp((int)std::round(img_x), 0, (int)bw - 1);
    int   iy    = std::clamp((int)std::round(img_y), 0, (int)bh - 1);

    int ci = comp_map[iy * bw + ix];
    if (ci < 0) {
        // Anchor landed on an erased pixel -- pick nearest component by centroid
        float best_d2 = 1e30f;
        ci = 0;
        for (int k = 0; k < (int)components.size(); k++) {
            float cx = (components[k].x0 + components[k].x1) * 0.5f;
            float cy = (components[k].y0 + components[k].y1) * 0.5f;
            float d2 = (cx - img_x) * (cx - img_x) + (cy - img_y) * (cy - img_y);
            if (d2 < best_d2) { best_d2 = d2; ci = k; }
        }
    }

    // Convert anchor from original image space -> new fragment's body-local frame
    auto& comp = components[ci];
    int   nw   = comp.x1 - comp.x0 + 1;
    int   nh   = comp.y1 - comp.y0 + 1;
    r_local = {
        img_x - comp.x0 - nw * 0.5f + 0.5f - new_com_local_arr[ci].x,
        img_y - comp.y0 - nh * 0.5f + 0.5f - new_com_local_arr[ci].y,
    };
    return new_ids[ci];
}

/**
 * fragment a body
 * called if we have "components" representing fragmented parts
 * builds new rigidbodies from old one
 */
static void split_body_into_fragments(
    uint32_t old_id, std::vector<PixelComponent>& components, PhysicsEngine& engine,
    float separation_impulse = 800.f)
{
    RigidBody& rb  = engine.world.bodies[old_id];
    auto&      bdi = engine.body_draw_info[old_id];
    uint32_t   bw  = bdi.body_w;
    uint32_t   bh  = bdi.body_h;

    // Save everything we need before add_body may reallocate GPU buffers
    const glm::vec2 orig_pos   = glm::vec2(rb.position);
    const float     orig_angle = rb.position.z;
    const glm::vec2 orig_vel   = glm::vec2(rb.velocity);
    const float     orig_omega = rb.velocity.z;
    const glm::vec2 orig_com   = rb.com_local;
    const bool      is_static  = (rb.inv_mass == 0.f);
    const float     density    = rb.density;
    const uint32_t  col_layer  = rb.collision_layer;
    const uint32_t  col_mask   = rb.collision_mask;
    const float     ang_damp    = rb.angular_damping;
    const float     fric        = rb.friction;
    const float     frac_thresh = rb.fracture_threshold;
    const float     orig_mass   = rb.mass;
    const std::vector<glm::vec4> orig_pixels = rb.sprite.pixels;

    // Build pixel -> component index lookup for joint remapping
    std::vector<int> comp_map(bw * bh, -1);
    for (int ci = 0; ci < (int)components.size(); ci++)
        for (auto& p : components[ci].pixels)
            comp_map[p.y * bw + p.x] = ci;

    // Create one new body per component
    std::vector<uint32_t>   new_ids(components.size());
    std::vector<glm::vec2>  new_com_local_arr(components.size());

    for (int ci = 0; ci < (int)components.size(); ci++) {
        auto& comp = components[ci];
        int nw = comp.x1 - comp.x0 + 1;
        int nh = comp.y1 - comp.y0 + 1;

        LoadedBodyImage new_sprite;
        new_sprite.width  = (uint32_t)nw;
        new_sprite.height = (uint32_t)nh;
        new_sprite.pixels.assign(nw * nh, glm::vec4(0.f));
        for (auto& p : comp.pixels)
            new_sprite.pixels[(p.y - comp.y0) * nw + (p.x - comp.x0)]
                = orig_pixels[p.y * bw + p.x];

        RigidBody new_rb;
        new_rb.sprite          = std::move(new_sprite);
        new_rb.density         = density;
        new_rb.collision_layer = col_layer;
        new_rb.collision_mask  = col_mask;
        new_rb.angular_damping = ang_damp;
        new_rb.friction        = fric;

        new_rb.compute_mass_properties(density);

        // Scale threshold by mass ratio: a fragment at half the mass generates
        // half the contact force, so give it half the threshold to match.
        new_rb.fracture_threshold = (frac_thresh > 0.f && orig_mass > 0.f)
            ? frac_thresh * (new_rb.mass / orig_mass)
            : frac_thresh;
        new_rb.generate_shape();
        new_rb.generate_sdf();
        new_com_local_arr[ci] = new_rb.com_local;

        // Offset of new fragment COM from original COM, in original local frame
        glm::vec2 offset_local = {
            comp.x0 + nw * 0.5f - 0.5f + new_rb.com_local.x - (bw * 0.5f - 0.5f) - orig_com.x,
            comp.y0 + nh * 0.5f - 0.5f + new_rb.com_local.y - (bh * 0.5f - 0.5f) - orig_com.y,
        };
        glm::vec2 offset_world = glm::rotate(offset_local, orig_angle);
        glm::vec2 new_world    = orig_pos + offset_world;

        new_rb.position = glm::vec3(new_world, orig_angle);

        if (is_static) {
            new_rb.mass = new_rb.inertia = new_rb.inv_mass = new_rb.inv_inertia = 0.f;
        } else {
            glm::vec2 v_frag = orig_vel + glm::vec2(-orig_omega * offset_world.y,
                                                      orig_omega * offset_world.x);

            // separation force
            // TODO: improve
            if (glm::dot(offset_world, offset_world) > 1e-6f && new_rb.mass > 0.f)
                v_frag += glm::normalize(offset_world) * (separation_impulse / new_rb.mass);

            new_rb.velocity = glm::vec3(v_frag, orig_omega);
        }

        new_ids[ci] = engine.add_body(engine.engine_ref, std::move(new_rb));
    }

    // Extract forces that reference the old body so remove_body doesn't delete them
    std::vector<std::unique_ptr<Force>> to_reassign;
    {
        std::vector<std::unique_ptr<Force>> remaining;
        for (auto& f : engine.world.forces) {
            if (f->bodyA == old_id || f->bodyB == old_id)
                to_reassign.push_back(std::move(f));
            else
                remaining.push_back(std::move(f));
        }
        engine.world.forces = std::move(remaining);
    }

    engine.remove_body(old_id);

    // Remap each extracted force to the correct new fragment and re-add
    for (auto& f : to_reassign) {
        if (auto* dj = dynamic_cast<DistanceJoint*>(f.get())) {
            if (dj->bodyA == old_id)
                dj->bodyA = remap_anchor(dj->rA_local, old_id, components, comp_map,
                                         new_ids, new_com_local_arr, orig_com, bw, bh);
            if (dj->bodyB == old_id)
                dj->bodyB = remap_anchor(dj->rB_local, old_id, components, comp_map,
                                         new_ids, new_com_local_arr, orig_com, bw, bh);
        } else if (auto* sp = dynamic_cast<Spring*>(f.get())) {
            if (sp->bodyA == old_id)
                sp->bodyA = remap_anchor(sp->rA_local, old_id, components, comp_map,
                                         new_ids, new_com_local_arr, orig_com, bw, bh);
            if (sp->bodyB == old_id)
                sp->bodyB = remap_anchor(sp->rB_local, old_id, components, comp_map,
                                         new_ids, new_com_local_arr, orig_com, bw, bh);
        } else {
            // Fallback for other force types: assign to largest fragment (index 0)
            if (f->bodyA == old_id) f->bodyA = new_ids[0];
            if (f->bodyB == old_id) f->bodyB = new_ids[0];
        }
        engine.world.add_force(std::move(f));
    }
}

/**
 * impact fracture: apply a fracture event from a contact point
 * falls back to thickest point as the origin of the crack otherwise
 * partitions the rigidbody into pixel components
 */
static void apply_line_crack_fracture(uint32_t body_id, glm::vec2 contact_world,
                                       float fn, PhysicsEngine& engine)
{
    if (body_id >= engine.world.bodies.size() || !engine.world.active[body_id]) return;

    RigidBody& rb  = engine.world.bodies[body_id];
    auto&      bdi = engine.body_draw_info[body_id];
    uint32_t   bw  = bdi.body_w;
    uint32_t   bh  = bdi.body_h;
    if (bw == 0 || bh == 0) return;

    // World -> body-local (COM-relative) space -- the same frame rb.shape is
    // stored in, since line_crack_partition works directly against the mesh.
    float cos_a = std::cos(-rb.position.z);
    float sin_a = std::sin(-rb.position.z);
    glm::vec2 dw = contact_world - glm::vec2(rb.position);
    glm::vec2 contact_local = {
        cos_a * dw.x - sin_a * dw.y,
        sin_a * dw.x + cos_a * dw.y,
    };

    uint32_t rng = g_fracture_seed ^ (body_id * 2246822519u) ^ (uint32_t)(fn * 73856093.f);
    g_fracture_seed = rng * 1664525u + 1013904223u;

    auto components = line_crack_partition(rb, bw, bh, contact_local, rng);
   
    // fallback to thickest point partition
    if (components.size() < 2)
        components = thickest_point_partition(rb, bw, bh, rng);
    if (components.size() < 2) return;  // still nothing viable -- leave the body alone

    // Harder impacts throw fragments apart more forcefully.
    float separation_impulse = std::clamp(fn * 2.f, 200.f, 2000.f);
    split_body_into_fragments(body_id, components, engine, separation_impulse);
}

/**
 * Apply fracture logic to all marked bodies. 
 * main logic body for fractures
 */
void PhysicsEngine::fracture_dirty_bodies()
{
    // Apply physics-driven fractures from last frame's solve() pass.
    // Deduplicate per body: only apply the strongest impact per body per frame.
    if (!world.pending_fractures.empty()) {
        std::unordered_map<uint32_t, FractureEvent*> strongest;
        for (auto& ev : world.pending_fractures) {
            auto it = strongest.find(ev.body_id);
            if (it == strongest.end() || ev.impulse > it->second->impulse)
                strongest[ev.body_id] = &ev;
        }
        for (auto& [bid, ev] : strongest)
            apply_line_crack_fracture(bid, ev->contact_world, ev->impulse, *this);
        world.pending_fractures.clear();
    }

    if (dirty_bodies_.empty()) return;

    for (auto& [old_id, di] : dirty_bodies_) {
        if (old_id >= world.bodies.size() || !world.active[old_id]) continue;

        RigidBody& rb  = world.bodies[old_id];
        BodyGPUInfo& bdi = body_draw_info[old_id];
        uint32_t bw = bdi.body_w;
        uint32_t bh = bdi.body_h;
        if (bw == 0 || bh == 0) continue;

        // Remove immediately if no solid pixels remain (early-exit on first found)
        {
            bool any = false;
            for (const auto& px : rb.sprite.pixels)
                if (px.a > 0.f) { any = true; break; }
            if (!any) { remove_body(old_id); continue; }
        }

        auto components = find_components(
            rb.sprite.pixels.data(), bw, bh,
            di.px_min, di.py_min, di.px_max, di.py_max);

        // -- All pixels erased --------------------------------------------------
        if (components.empty()) {
            remove_body(old_id);
            continue;
        }

        // this function might need more explanation
        // basically it will prevent weird edge cases, particularly for in terms of the size of the fractured bodies
        enforce_fragment_limits(components, bw, bh, MAX_FRAGMENTS);

        // -- No split: update mass incrementally, regenerate shape + patch SDF --
        if (components.size() == 1) {
            // Incremental mass/inertia update (skip for static bodies).
            if (rb.inv_mass != 0.f && di.del_count > 0.f) {
                float old_N = rb.mass / rb.density;
                float new_N = old_N - di.del_count;
                if (new_N <= 0.f) {
                    remove_body(old_id);
                    continue;
                }
                // Recover absolute-pixel-space sums from existing mass properties.
                glm::vec2 old_centroid = rb.com_local + glm::vec2(bw * 0.5f, bh * 0.5f);
                float     old_r2_sum   = rb.inertia / rb.density + old_N * glm::dot(old_centroid, old_centroid);
                glm::vec2 old_pos_sum  = old_N * old_centroid;

                // Subtract deleted pixels' contributions.
                glm::vec2 new_pos_sum  = old_pos_sum - di.del_pos_sum;
                float     new_r2_sum   = old_r2_sum  - di.del_r2_sum;
                glm::vec2 new_centroid = new_pos_sum / new_N;

                rb.mass      = rb.density * new_N;
                rb.inv_mass  = 1.f / rb.mass;
                rb.com_local = new_centroid - glm::vec2(bw * 0.5f, bh * 0.5f);

                float new_I    = rb.density * (new_r2_sum - new_N * glm::dot(new_centroid, new_centroid));
                rb.inertia     = std::max(new_I, 1.f);
                rb.inv_inertia = 1.f / rb.inertia;
            }

            // upload new shape
            rb.generate_shape();
            if (!rb.shape.empty()) {
                auto* vert_dst = static_cast<glm::vec2*>(body_verts_buf.info.pMappedData);
                uint32_t new_count = (uint32_t)rb.shape.size();
                uint32_t upload_count = std::min(new_count, bdi.edge_count);
                std::memcpy(vert_dst + bdi.edge_offset, rb.shape.data(),
                            upload_count * sizeof(glm::vec2));
                bdi.edge_count = upload_count;
            }

            // patch sdf
            {
                int max_dim = std::max(di.px_max - di.px_min + 1,
                                       di.py_max - di.py_min + 1);
                int margin  = max_dim + 2;       // sprite-pixel margin
                uint32_t s  = SDF_SCALE;
                uint32_t sdf_w = bw * s;
                uint32_t sdf_h = bh * s;

                uint32_t sdf_x0 = (uint32_t)std::max(0, di.px_min * (int)s - margin * (int)s);
                uint32_t sdf_y0 = (uint32_t)std::max(0, di.py_min * (int)s - margin * (int)s);
                uint32_t sdf_x1 = std::min(sdf_w - 1u,
                    (uint32_t)(di.px_max * (int)s + (int)s - 1 + margin * (int)s));
                uint32_t sdf_y1 = std::min(sdf_h - 1u,
                    (uint32_t)(di.py_max * (int)s + (int)s - 1 + margin * (int)s));

                patch_sdf_region(rb.sprite, rb.shape, rb.com_local, s,
                                 sdf_x0, sdf_y0, sdf_x1, sdf_y1, rb.sdf);

                // Upload only the patched rows to the GPU buffer.
                auto* sdf_dst = static_cast<float*>(sdf_data_buf.info.pMappedData);
                for (uint32_t py = sdf_y0; py <= sdf_y1; py++)
                    std::memcpy(sdf_dst + bdi.sdf_offset + py * sdf_w + sdf_x0,
                                rb.sdf.data()             + py * sdf_w + sdf_x0,
                                (sdf_x1 - sdf_x0 + 1) * sizeof(float));
            }
            continue;
        }

        // do the split!
        split_body_into_fragments(old_id, components, *this);
    }

    dirty_bodies_.clear();
}
