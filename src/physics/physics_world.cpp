#include "physics_world.h"
#include "manifold.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <execution>
#include <numeric>
#include <mutex>
#include <unordered_set>

static AABB compute_world_aabb(const RigidBody2& rb) {
    glm::vec2 mn( 1e20f,  1e20f);
    glm::vec2 mx(-1e20f, -1e20f);
    for (const glm::vec2& v : rb.shape) {
        glm::vec2 w = glm::vec2(rb.position) + glm::rotate(v, rb.position.z);
        mn = glm::min(mn, w);
        mx = glm::max(mx, w);
    }
    return { mn, mx };
}


Force::Force(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB)
    : world(world), bodyA(bodyA), bodyB(bodyB)
{
    for (int i = 0; i < AVBD_MAX_ROWS; i++) {
        C[i]         = 0.f;
        fmin[i]      = -std::numeric_limits<float>::infinity();
        fmax[i]      =  std::numeric_limits<float>::infinity();
        stiffness[i] =  std::numeric_limits<float>::infinity();
        fracture[i]  =  std::numeric_limits<float>::infinity();
        penalty[i]   = 0.f;
        lambda[i]    = 0.f;
    }
}

uint32_t PhysicsWorld::add_body(RigidBody2& rb) {
    if (!open_slots.empty()) {
        uint32_t slot = open_slots.back();
        open_slots.pop_back();
        bodies[slot] = rb;
        active[slot] = true;
        return slot;
    }
    bodies.push_back(rb);
    active.push_back(true);
    return (uint32_t)bodies.size() - 1;
}

void PhysicsWorld::remove_body(uint32_t index) {
    active[index] = false;
    open_slots.push_back(index);
    // Remove any forces referencing this body.
    forces.erase(
        std::remove_if(forces.begin(), forces.end(),
            [index](const std::unique_ptr<Force>& f) {
                return f->bodyA == index || f->bodyB == index;
            }),
        forces.end());
}

uint32_t PhysicsWorld::add_static_rect(glm::vec2 center, float w, float h) {
    RigidBody2 rb;
    rb.position    = glm::vec3(center, 0.f);
    rb.inv_mass    = 0.f;
    rb.inv_inertia = 0.f;
    rb.mass        = 0.f;
    rb.inertia     = 0.f;

    float hw = w * 0.5f, hh = h * 0.5f;
    rb.shape = {
        { -hw, -hh },
        {  hw, -hh },
        {  hw,  hh },
        { -hw,  hh },
    };

    uint32_t iw = (uint32_t)std::max(1.f, std::round(w));
    uint32_t ih = (uint32_t)std::max(1.f, std::round(h));
    LoadedBodyImage img;
    img.width  = iw;
    img.height = ih;
    img.pixels.assign(iw * ih, glm::vec4(1.f));  

    rb.sprite = img;
    rb.sdf    = generate_sdf(img, rb.shape, glm::vec2(0.f));  
    rb.sdf_w  = iw;
    rb.sdf_h  = ih;

    return add_body(rb);
}

RigidBody2& PhysicsWorld::get_body(uint32_t index) {
    return bodies[index];
}

void PhysicsWorld::prepare() {
    auto start = std::chrono::system_clock::now();

    precomputed_contacts.clear();

    stats.num_bodies = 0;
    for (uint32_t i = 0; i < (uint32_t)active.size(); i++)
        if (active[i]) stats.num_bodies++;

    std::vector<AABB>     boxes;
    std::vector<uint32_t> index_map;
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i] || bodies[i].shape.empty()) continue;
        boxes.push_back(compute_world_aabb(bodies[i]));
        index_map.push_back(i);
    }
    lbvh_build(lbvh, boxes);

    auto t1 = std::chrono::system_clock::now();
    stats.lbvh_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count() / 1000.f;

    // Build O(1) lookup for already-existing manifold pairs.
    std::unordered_set<uint64_t> existing;
    existing.reserve(forces.size() * 2);
    for (const auto& f : forces) {
        uint32_t a = std::min(f->bodyA, f->bodyB);
        uint32_t b = std::max(f->bodyA, f->bodyB);
        existing.insert(((uint64_t)a << 32) | b);
    }

    // Parallel AABB queries — each thread owns its local candidates vector.
    // The lj > li guard ensures pair (i,j) is emitted by exactly one thread
    std::vector<std::pair<uint32_t,uint32_t>> new_pairs;
    std::mutex new_pairs_mtx;

    std::vector<uint32_t> li_range((uint32_t)index_map.size());
    std::iota(li_range.begin(), li_range.end(), 0u);

    std::for_each(std::execution::par, li_range.begin(), li_range.end(),
        [&](uint32_t li) {
            std::vector<int> local_cands;
            lbvh_query_AABB(lbvh, boxes[li], local_cands);

            std::vector<std::pair<uint32_t,uint32_t>> local_new;
            for (int lj : local_cands) {
                if (lj <= (int)li) continue;
                uint32_t i = index_map[li];
                uint32_t j = index_map[lj];
                uint32_t a = std::min(i, j), b = std::max(i, j);
                if (!existing.count(((uint64_t)a << 32) | b))
                    local_new.push_back({i, j});
            }

            if (!local_new.empty()) {
                std::lock_guard lock(new_pairs_mtx);
                for (auto p : local_new) new_pairs.push_back(p);
            }
        });

    for (auto [i, j] : new_pairs)
        forces.push_back(std::make_unique<Manifold>(this, i, j));

    stats.num_forces = (uint32_t)forces.size();

    auto t2 = std::chrono::system_clock::now();
    stats.broadphase_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;
}

void PhysicsWorld::solve(float dt) {
    auto start = std::chrono::system_clock::now();

    // Warm-start forces (initializes contacts, Jacobians, dual variables).
    std::for_each(std::execution::par, forces.begin(), forces.end(),
        [&](std::unique_ptr<Force>& f) {
            if (!f->initialize()) { f.reset(); return; }
            for (int i = 0; i < f->rows(); i++) {
                if (postStabilize) {
                    f->penalty[i] = std::clamp(f->penalty[i] * gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
                } else {
                    f->lambda[i]  = f->lambda[i] * alpha * gamma;
                    f->penalty[i] = std::clamp(f->penalty[i] * gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
                }
                f->penalty[i] = std::min(f->penalty[i], f->stiffness[i]);
            }
        });
    forces.erase(std::remove_if(forces.begin(), forces.end(),
        [](const std::unique_ptr<Force>& f) { return !f; }), forces.end());

    // Build per-body adjacency list so the solver doesn't scan all forces per body.
    std::vector<std::vector<Force*>> body_forces(bodies.size());
    for (auto& f : forces) {
        body_forces[f->bodyA].push_back(f.get());
        body_forces[f->bodyB].push_back(f.get());
    }

    auto t1 = std::chrono::system_clock::now();
    stats.warmstart_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count() / 1000.f;

    // Predict positions (primal warm-start).
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        RigidBody2& rb = bodies[i];

        rb.velocity.z = std::clamp(rb.velocity.z, -50.f, 50.f);

        rb.inertial = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.inertial += glm::vec3(0.f, gravity, 0.f) * (dt * dt);

        glm::vec3 accel      = (rb.velocity - rb.prev_velocity) / dt;
        float     accelExt   = accel.y * (gravity < 0.f ? -1.f : 1.f);
        float     accelWeight = std::clamp(accelExt / std::abs(gravity), 0.f, 1.f);
        if (!std::isfinite(accelWeight)) accelWeight = 0.f;

        rb.initial  = rb.position;
        rb.position = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.position += glm::vec3(0.f, gravity, 0.f) * (accelWeight * dt * dt);
    }

    auto t2 = std::chrono::system_clock::now();
    stats.predict_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;

    // Pre-group dynamic bodies by color. Falls back to a single group
    uint32_t numColors = colors.empty() ? 1u : (max_color + 1u);
    std::vector<std::vector<uint32_t>> color_groups(numColors);
    for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
        if (!active[bi] || bodies[bi].inv_mass == 0.f) continue;
        uint32_t c = colors.empty() ? 0u : colors[bi];
        if (c < numColors)
            color_groups[c].push_back(bi);
    }

    std::vector<glm::vec3> x_new(bodies.size());

    // Main solver loop.
    int totalIterations = iterations + (postStabilize ? 1 : 0);

    for (int it = 0; it < totalIterations; it++) {
        float currentAlpha = alpha;
        if (postStabilize)
            currentAlpha = (it < iterations) ? 1.f : 0.f;

        // Steps 8-25: color loop — all vertices in a color update simultaneously.
        for (uint32_t c = 0; c < numColors; c++) {
            // Steps 9-20: compute x_new for each body in this color 
            std::for_each(std::execution::par_unseq,
                color_groups[c].begin(), color_groups[c].end(),
                [&](uint32_t bi) {
                    RigidBody2& rb = bodies[bi];

                    Mat3      lhs = mat3_diag(rb.mass, rb.mass, rb.inertia) / (dt * dt);
                    glm::vec3 rhs = lhs * (rb.position - rb.inertial);

                    for (Force* f : body_forces[bi]) {
                        f->computeConstraint(currentAlpha);
                        f->computeDerivatives(bi);

                        for (int i = 0; i < f->rows(); i++) {
                            float lam       = std::isinf(f->stiffness[i]) ? f->lambda[i] : 0.f;
                            float force_val = std::clamp(f->penalty[i] * f->C[i] + lam,
                                                         f->fmin[i], f->fmax[i]);

                            Mat3 G = mat3_diag(glm::length(f->H[i].col(0)),
                                               glm::length(f->H[i].col(1)),
                                               glm::length(f->H[i].col(2))) * std::abs(force_val);

                            rhs += f->J[i] * force_val;
                            lhs += mat3_outer(f->J[i], f->J[i] * f->penalty[i]) + G;
                        }
                    }

                    x_new[bi] = rb.position - mat3_solve(lhs, rhs);
                });

            // Steps 22-24: apply buffered positions for this color 
            std::for_each(std::execution::par_unseq,
                color_groups[c].begin(), color_groups[c].end(),
                [&](uint32_t bi) {
                    bodies[bi].position = x_new[bi];
                });
        }

        // Steps 26-35: lambda and penalty update over all forces 
        if (it < iterations) {
            std::for_each(std::execution::par_unseq,
                forces.begin(), forces.end(),
                [&](const std::unique_ptr<Force>& f) {
                    f->computeConstraint(currentAlpha);
                    for (int i = 0; i < f->rows(); i++) {
                        float lam    = std::isinf(f->stiffness[i]) ? f->lambda[i] : 0.f;
                        f->lambda[i] = std::clamp(f->penalty[i] * f->C[i] + lam,
                                                  f->fmin[i], f->fmax[i]);

                        if (f->lambda[i] > f->fmin[i] && f->lambda[i] < f->fmax[i])
                            f->penalty[i] = std::min(f->penalty[i] + beta * std::abs(f->C[i]),
                                                     std::min(AVBD_PENALTY_MAX, f->stiffness[i]));
                    }
                });
        }
    }

    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        RigidBody2& rb = bodies[i];
        rb.prev_velocity = rb.velocity;
        if (rb.inv_mass > 0.f)
            rb.velocity = (rb.position - rb.initial) / dt;
    }

    auto t3 = std::chrono::system_clock::now();
    stats.solver_ms = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.f;
}

void PhysicsWorld::step(float dt) {
    prepare();
    solve(dt);
}
