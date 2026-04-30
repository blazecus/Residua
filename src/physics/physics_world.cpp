#include "physics_world.h"
#include "manifold.h"
#include "shape_gen.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <execution>
#include <numeric>
#include <mutex>
#include <unordered_set>

static AABB compute_world_aabb(const RigidBody& rb) {
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

uint32_t PhysicsWorld::add_body(RigidBody& rb) {
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

    // Remove index from every body it was joint-connected to.
    for (uint32_t nb : bodies[index].joint_connected) {
        if (nb >= bodies.size()) continue;
        auto& jc = bodies[nb].joint_connected;
        jc.erase(std::remove(jc.begin(), jc.end(), index), jc.end());
    }
    bodies[index].joint_connected.clear();

    // Remove any forces referencing this body.
    forces.erase(
        std::remove_if(forces.begin(), forces.end(),
            [index](const std::unique_ptr<Force>& f) {
                return f->bodyA == index || f->bodyB == index;
            }),
        forces.end());
}

void PhysicsWorld::add_force(std::unique_ptr<Force> f) {
    if (!f->is_contact()) {
        uint32_t a = f->bodyA, b = f->bodyB;
        if (a < bodies.size() && b < bodies.size()) {
            bodies[a].joint_connected.push_back(b);
            bodies[b].joint_connected.push_back(a);
        }
    }
    forces.push_back(std::move(f));
}

void PhysicsWorld::remove_force(Force* f) {
    auto it = std::find_if(forces.begin(), forces.end(),
        [f](const std::unique_ptr<Force>& p) { return p.get() == f; });
    if (it != forces.end()) forces.erase(it);
}

uint32_t PhysicsWorld::add_static_rect(glm::vec2 center, float w, float h) {
    RigidBody rb;
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
    rb.sdf    = generate_sdf(img, rb.shape, glm::vec2(0.f), SDF_SCALE);
    rb.sdf_w  = iw * SDF_SCALE;
    rb.sdf_h  = ih * SDF_SCALE;

    return add_body(rb);
}

RigidBody& PhysicsWorld::get_body(uint32_t index) {
    return bodies[index];
}

void PhysicsWorld::prepare() {
    precomputed_contacts.clear();

    stats.num_bodies = 0;
    for (uint32_t i = 0; i < (uint32_t)active.size(); i++)
        if (active[i]) stats.num_bodies++;

    std::vector<AABB>     boxes;
    std::vector<uint32_t> index_map;

    auto t0 = std::chrono::system_clock::now();
    build_lbvh(boxes, index_map);
    auto t1 = std::chrono::system_clock::now();
    broadphase(boxes, index_map);
    auto t2 = std::chrono::system_clock::now();

    stats.lbvh_ms       = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    stats.broadphase_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;
}

void PhysicsWorld::build_lbvh(std::vector<AABB>& boxes, std::vector<uint32_t>& index_map) {
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i] || bodies[i].shape.empty()) continue;
        boxes.push_back(compute_world_aabb(bodies[i]));
        index_map.push_back(i);
    }
    lbvh_build(lbvh, boxes);
    lbvh_index_map = index_map;
}

void PhysicsWorld::broadphase(const std::vector<AABB>& boxes, const std::vector<uint32_t>& index_map) {
    forces.erase(std::remove_if(forces.begin(), forces.end(),
        [&](const std::unique_ptr<Force>& f) -> bool {
            if (!f->is_contact()) return false;
            uint32_t i = f->bodyA, j = f->bodyB;
            if (i >= bodies.size() || !active[i]) return true;
            if (j >= bodies.size() || !active[j]) return true;
            return !AABB_intersects(compute_world_aabb(bodies[i]),
                                    compute_world_aabb(bodies[j]));
        }),
        forces.end());

    std::unordered_set<uint64_t> existing;
    existing.reserve(forces.size() * 2);
    for (const auto& f : forces) {
        if (!f->is_contact()) continue;
        uint32_t a = std::min(f->bodyA, f->bodyB);
        uint32_t b = std::max(f->bodyA, f->bodyB);
        existing.insert(((uint64_t)a << 32) | b);
    }

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
                const RigidBody& bi = bodies[i];
                const RigidBody& bj = bodies[j];
                if (!(bi.collision_layer & bj.collision_mask)) continue;
                if (!(bj.collision_layer & bi.collision_mask)) continue;
                {
                    bool joint_skip = false;
                    for (uint32_t c : bi.joint_connected) if (c == j) { joint_skip = true; break; }
                    if (joint_skip) continue;
                }
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
}

void PhysicsWorld::solve(float dt) {
    last_dt = dt;

    auto t0 = std::chrono::system_clock::now();
    warm_start(dt);
    auto t1 = std::chrono::system_clock::now();
    predict(dt);
    auto t2 = std::chrono::system_clock::now();
    iterate(dt);
    auto t3 = std::chrono::system_clock::now();

    stats.warmstart_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    stats.predict_ms   = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;
    stats.solver_ms    = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.f;
}

void PhysicsWorld::warm_start(float dt) {
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
}

void PhysicsWorld::predict(float dt) {
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        RigidBody& rb = bodies[i];

        rb.velocity.z = std::clamp(rb.velocity.z, -50.f, 50.f);
        if (rb.inv_mass > 0.f) {
            rb.velocity.z *= rb.angular_damping;
            rb.velocity.x += rb.ext_force.x * rb.inv_mass   * dt;
            rb.velocity.y += rb.ext_force.y * rb.inv_mass   * dt;
            rb.velocity.z += rb.ext_torque  * rb.inv_inertia * dt;
            rb.ext_force  = {0.f, 0.f};
            rb.ext_torque = 0.f;
        }

        rb.inertial = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.inertial += glm::vec3(0.f, gravity, 0.f) * (dt * dt);

        glm::vec3 accel       = (rb.velocity - rb.prev_velocity) / dt;
        float     accelExt    = accel.y * (gravity < 0.f ? -1.f : 1.f);
        float     accelWeight = std::clamp(accelExt / std::abs(gravity), 0.f, 1.f);
        if (!std::isfinite(accelWeight)) accelWeight = 0.f;

        rb.initial  = rb.position;
        rb.position = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.position += glm::vec3(0.f, gravity, 0.f) * (accelWeight * dt * dt);
    }
}

void PhysicsWorld::iterate(float dt) {
    std::vector<std::vector<Force*>> body_forces(bodies.size());
    for (auto& f : forces) {
        body_forces[f->bodyA].push_back(f.get());
        if (f->bodyB < bodies.size())
            body_forces[f->bodyB].push_back(f.get());
    }

    uint32_t numColors = colors.empty() ? 1u : (max_color + 1u);
    std::vector<std::vector<uint32_t>> color_groups(numColors);
    for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
        if (!active[bi] || bodies[bi].inv_mass == 0.f) continue;
        uint32_t c = colors.empty() ? 0u : colors[bi];
        if (c < numColors)
            color_groups[c].push_back(bi);
    }

    std::vector<glm::vec3> x_new(bodies.size());
    int totalIterations = iterations + (postStabilize ? 1 : 0);

    for (int it = 0; it < totalIterations; it++) {
        float currentAlpha = alpha;
        if (postStabilize)
            currentAlpha = (it < iterations) ? 1.f : 0.f;

        for (uint32_t c = 0; c < numColors; c++) {
            std::for_each(std::execution::par_unseq,
                color_groups[c].begin(), color_groups[c].end(),
                [&](uint32_t bi) {
                    RigidBody& rb = bodies[bi];

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

            std::for_each(std::execution::par_unseq,
                color_groups[c].begin(), color_groups[c].end(),
                [&](uint32_t bi) {
                    bodies[bi].position = x_new[bi];
                });
        }

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

        if (it == iterations - 1) {
            for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
                if (!active[i]) continue;
                RigidBody& rb = bodies[i];
                rb.prev_velocity = rb.velocity;
                if (rb.inv_mass > 0.f)
                    rb.velocity = (rb.position - rb.initial) / dt;
            }
        }
    }
}

void PhysicsWorld::step(float dt) {
    prepare();
    solve(dt);
}
