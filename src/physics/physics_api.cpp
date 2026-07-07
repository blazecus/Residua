#include "physics_engine.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <cmath>

// ── Helpers ────────────────────────────────────────────────────────────────────

// Works for convex and concave polygons: tests each edge, returns nearest hit.
static bool ray_vs_polygon(glm::vec2 origin, glm::vec2 dir,
                            const std::vector<glm::vec2>& verts,
                            float max_dist,
                            float& t_out, glm::vec2& normal_out)
{
    bool      found   = false;
    float     best_t  = max_dist;
    glm::vec2 best_n  {0.f};

    for (size_t i = 0; i < verts.size(); i++) {
        glm::vec2 a    = verts[i];
        glm::vec2 b    = verts[(i + 1) % verts.size()];
        glm::vec2 edge = b - a;

        // 2D cross: cross(D, E) = D.x*E.y - D.y*E.x
        float denom = dir.x * edge.y - dir.y * edge.x;
        if (std::abs(denom) < 1e-7f) continue; // parallel

        glm::vec2 ao = a - origin;
        float t = (ao.x * edge.y - ao.y * edge.x) / denom;
        float u = (ao.x * dir.y  - ao.y * dir.x)  / denom;

        if (t < 0.f || t > best_t || u < 0.f || u > 1.f) continue;

        // Edge normal facing against the ray direction.
        glm::vec2 norm = glm::vec2(edge.y, -edge.x);
        if (glm::dot(norm, dir) > 0.f) norm = -norm;

        best_t = t;
        best_n = glm::normalize(norm);
        found  = true;
    }

    if (found) { t_out = best_t; normal_out = best_n; }
    return found;
}

static std::vector<glm::vec2> world_verts(const RigidBody& rb)
{
    std::vector<glm::vec2> out;
    out.reserve(rb.shape.size());
    glm::vec2 pos   = glm::vec2(rb.position);
    float     angle = rb.position.z;
    for (auto& v : rb.shape)
        out.push_back(pos + glm::rotate(v, angle));
    return out;
}

// ── Body state ─────────────────────────────────────────────────────────────────

bool PhysicsEngine::body_valid(uint32_t id) const
{
    return id < world.bodies.size() && world.active[id];
}

bool PhysicsEngine::is_static(uint32_t id) const
{
    return body_valid(id) && world.bodies[id].inv_mass == 0.f;
}

glm::vec2 PhysicsEngine::get_position(uint32_t id) const
{
    return glm::vec2(world.bodies[id].position);
}

float PhysicsEngine::get_rotation(uint32_t id) const
{
    return world.bodies[id].position.z;
}

glm::vec2 PhysicsEngine::get_velocity(uint32_t id) const
{
    return glm::vec2(world.bodies[id].velocity);
}

float PhysicsEngine::get_angular_velocity(uint32_t id) const
{
    return world.bodies[id].velocity.z;
}

void PhysicsEngine::set_position(uint32_t id, glm::vec2 pos)
{
    world.bodies[id].position.x = pos.x;
    world.bodies[id].position.y = pos.y;
}

void PhysicsEngine::set_rotation(uint32_t id, float angle)
{
    world.bodies[id].position.z = angle;
}

void PhysicsEngine::set_velocity(uint32_t id, glm::vec2 vel)
{
    world.bodies[id].velocity.x = vel.x;
    world.bodies[id].velocity.y = vel.y;
}

void PhysicsEngine::set_angular_velocity(uint32_t id, float omega)
{
    world.bodies[id].velocity.z = omega;
}

// ── Forces & impulses ──────────────────────────────────────────────────────────

void PhysicsEngine::apply_force(uint32_t id, glm::vec2 force)
{
    world.bodies[id].ext_force += force;
}

void PhysicsEngine::apply_force_at(uint32_t id, glm::vec2 force, glm::vec2 world_point)
{
    RigidBody& rb  = world.bodies[id];
    glm::vec2  com = glm::vec2(rb.position) + glm::rotate(rb.com_local, rb.position.z);
    glm::vec2  r   = world_point - com;
    rb.ext_force  += force;
    rb.ext_torque += r.x * force.y - r.y * force.x;
}

void PhysicsEngine::apply_torque(uint32_t id, float torque)
{
    world.bodies[id].ext_torque += torque;
}

void PhysicsEngine::apply_impulse(uint32_t id, glm::vec2 impulse)
{
    RigidBody& rb   = world.bodies[id];
    rb.velocity.x  += impulse.x * rb.inv_mass;
    rb.velocity.y  += impulse.y * rb.inv_mass;
}

void PhysicsEngine::apply_impulse_at(uint32_t id, glm::vec2 impulse, glm::vec2 world_point)
{
    RigidBody& rb  = world.bodies[id];
    glm::vec2  com = glm::vec2(rb.position) + glm::rotate(rb.com_local, rb.position.z);
    glm::vec2  r   = world_point - com;
    rb.velocity.x += impulse.x * rb.inv_mass;
    rb.velocity.y += impulse.y * rb.inv_mass;
    rb.velocity.z += (r.x * impulse.y - r.y * impulse.x) * rb.inv_inertia;
}

// ── Collision queries ──────────────────────────────────────────────────────────

std::vector<ContactInfo> PhysicsEngine::get_contacts(uint32_t id) const
{
    std::vector<ContactInfo> result;
    for (auto& [key, points] : world.precomputed_contacts) {
        if (points.empty()) continue;
        uint32_t a = (uint32_t)(key >> 32);
        uint32_t b = (uint32_t)(key & 0xFFFFFFFF);
        if (a != id && b != id) continue;

        uint32_t other = (a == id) ? b : a;
        float    flip  = (a == id) ? 1.f : -1.f;

        // Pick the deepest point as the representative for this pair.
        const ManifoldPoint* best = &points[0];
        for (auto& mp : points)
            if (mp.depth > best->depth) best = &mp;

        result.push_back({
            .other_id = other,
            .point    = best->position,
            .normal   = best->normal * flip,
            .depth    = best->depth,
        });
    }
    return result;
}

bool PhysicsEngine::is_touching(uint32_t a, uint32_t b) const
{
    uint64_t key_ab = ((uint64_t)a << 32) | b;
    uint64_t key_ba = ((uint64_t)b << 32) | a;
    return world.precomputed_contacts.count(key_ab) ||
           world.precomputed_contacts.count(key_ba);
}

// ── Raycasts ───────────────────────────────────────────────────────────────────

std::optional<RaycastHit> PhysicsEngine::raycast(glm::vec2 origin, glm::vec2 dir,
                                                   float max_dist, uint32_t mask) const
{
    std::vector<int> candidates;
    lbvh_query_ray(world.lbvh, origin, dir, max_dist, candidates);

    std::optional<RaycastHit> best;
    for (int box_idx : candidates) {
        uint32_t i = world.lbvh_index_map[box_idx];
        if (!(world.bodies[i].collision_layer & mask)) continue;

        auto verts = world_verts(world.bodies[i]);
        float t; glm::vec2 n;
        if (ray_vs_polygon(origin, dir, verts, max_dist, t, n))
            if (!best || t < best->distance)
                best = RaycastHit{ i, origin + dir * t, n, t };
    }
    return best;
}

std::vector<RaycastHit> PhysicsEngine::raycast_all(glm::vec2 origin, glm::vec2 dir,
                                                    float max_dist, uint32_t mask) const
{
    std::vector<int> candidates;
    lbvh_query_ray(world.lbvh, origin, dir, max_dist, candidates);

    std::vector<RaycastHit> hits;
    for (int box_idx : candidates) {
        uint32_t i = world.lbvh_index_map[box_idx];
        if (!(world.bodies[i].collision_layer & mask)) continue;

        auto verts = world_verts(world.bodies[i]);
        float t; glm::vec2 n;
        if (ray_vs_polygon(origin, dir, verts, max_dist, t, n))
            hits.push_back({ i, origin + dir * t, n, t });
    }
    std::sort(hits.begin(), hits.end(),
              [](const RaycastHit& a, const RaycastHit& b) { return a.distance < b.distance; });
    return hits;
}

void PhysicsEngine::destroy_pixels_in_radius(glm::vec2 world_pos, float radius)
{
    auto* px_buf = static_cast<glm::vec4*>(pixel_colors_buf.info.pMappedData);
    float r2 = radius * radius;

    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (!world.active[i]) continue;
        auto& bdi = body_draw_info[i];
        if (bdi.body_w == 0 || bdi.body_h == 0) continue;

        auto& rb = world.bodies[i];
        float c = std::cos(rb.position.z);
        float s = std::sin(rb.position.z);

        // Transform world_pos into body-local space (inverse rotation)
        glm::vec2 r_world = world_pos - glm::vec2(rb.position);
        float local_x =  c * r_world.x + s * r_world.y;
        float local_y = -s * r_world.x + c * r_world.y;

        // Invert the shader formula: local_pos = vec2(lx,ly) - vec2(bw,bh)*0.5 + 0.5 - com_local
        float px_cx = local_x + rb.com_local.x + bdi.body_w * 0.5f - 0.5f;
        float py_cy = local_y + rb.com_local.y + bdi.body_h * 0.5f - 0.5f;

        int px_min = std::max(0, (int)std::floor(px_cx - radius));
        int px_max = std::min((int)bdi.body_w - 1, (int)std::ceil(px_cx + radius));
        int py_min = std::max(0, (int)std::floor(py_cy - radius));
        int py_max = std::min((int)bdi.body_h - 1, (int)std::ceil(py_cy + radius));

        DirtyBodyInfo* di = nullptr;
        for (int py = py_min; py <= py_max; py++) {
            for (int px = px_min; px <= px_max; px++) {
                float dx = px - px_cx;
                float dy = py - py_cy;
                if (dx * dx + dy * dy > r2) continue;
                uint32_t sprite_idx = (uint32_t)py * bdi.body_w + (uint32_t)px;
                if (rb.sprite.pixels[sprite_idx].a <= 0.f) continue;
                rb.sprite.pixels[sprite_idx].a = 0.f;
                px_buf[bdi.pixel_index         + sprite_idx].a = 0.f;
                px_buf[bdi.flipped_pixel_index + py * bdi.body_w + (bdi.body_w - 1 - px)].a = 0.f;

                if (!di) di = &dirty_bodies_[i];
                glm::vec2 p{ (float)px + 0.5f, (float)py + 0.5f };
                di->del_count   += 1.f;
                di->del_pos_sum += p;
                di->del_r2_sum  += glm::dot(p, p);
                di->px_min = std::min(di->px_min, px);
                di->py_min = std::min(di->py_min, py);
                di->px_max = std::max(di->px_max, px);
                di->py_max = std::max(di->py_max, py);
            }
        }
    }
}
