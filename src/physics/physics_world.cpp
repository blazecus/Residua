#include "physics_world.h"
#include "manifold.h"
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

// Returns true if a Manifold already exists between bodyA and bodyB.
static bool manifold_exists(const std::vector<std::unique_ptr<Force>>& forces,
                             uint32_t a, uint32_t b)
{
    for (const auto& f : forces)
        if ((f->bodyA == a && f->bodyB == b) || (f->bodyA == b && f->bodyB == a))
            if (dynamic_cast<Manifold*>(f.get())) return true;
    return false;
}

// Force constructor registers itself — here we just own it.
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

// ─── Body management ──────────────────────────────────────────────────────────

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
    rb.triangles = {
        { 0, 1, 2 },
        { 0, 2, 3 },
    };
    return add_body(rb);
}

RigidBody2& PhysicsWorld::get_body(uint32_t index) {
    return bodies[index];
}

// ─── Step ─────────────────────────────────────────────────────────────────────

void PhysicsWorld::step(float dt) {
    std::chrono::microseconds elapsed{ 0 };
    auto start = std::chrono::system_clock::now();

    stats.num_bodies = 0;
    for (uint32_t i = 0; i < (uint32_t)active.size(); i++)
        if (active[i]) stats.num_bodies++;

    // 1. Build LBVH from world-space AABBs of active bodies.
    std::vector<AABB>     boxes;
    std::vector<uint32_t> index_map;

    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i] || bodies[i].shape.empty()) continue;
        boxes.push_back(compute_world_aabb(bodies[i]));
        index_map.push_back(i);
    }
    lbvh_build(lbvh, boxes);

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now () - start);
	stats.lbvh_ms = elapsed.count() / 1000.f;
    start = std::chrono::system_clock::now(); 

    // 2. Broad phase: create a Manifold for each new overlapping body pair.
    std::vector<int> candidates;
    for (uint32_t li = 0; li < (uint32_t)index_map.size(); li++) {
        lbvh_query_AABB(lbvh, boxes[li], candidates);
        for (int lj : candidates) {
            if (lj <= (int)li) continue;
            uint32_t i = index_map[li];
            uint32_t j = index_map[lj];
            if (!manifold_exists(forces, i, j))
                forces.push_back(std::make_unique<Manifold>(this, i, j));
        }
    }


    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now () - start);
	stats.broadphase_ms = elapsed.count() / 1000.f;
    start = std::chrono::system_clock::now();

    // 3. Warm-start forces (dual variables: lambda, penalty).
    for (auto& f : forces) {
        if (!f->initialize()) {
            f.reset();
            continue;
        }
        for (int i = 0; i < f->rows(); i++) {
            if (postStabilize) {
                f->penalty[i] = std::clamp(f->penalty[i] * gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
            } else {
                f->lambda[i]  = f->lambda[i] * alpha * gamma;
                f->penalty[i] = std::clamp(f->penalty[i] * gamma, AVBD_PENALTY_MIN, AVBD_PENALTY_MAX);
            }
            f->penalty[i] = std::min(f->penalty[i], f->stiffness[i]);
        }
    }
    forces.erase(std::remove_if(forces.begin(), forces.end(),
        [](const std::unique_ptr<Force>& f) { return !f; }), forces.end());

    // 4. Warm-start bodies (primal variables: predict position).
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        RigidBody2& rb = bodies[i];

        rb.velocity.z = std::clamp(rb.velocity.z, -50.f, 50.f);

        rb.inertial = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.inertial += glm::vec3(0.f, gravity, 0.f) * (dt * dt);

        glm::vec3 accel = (rb.velocity - rb.prev_velocity) / dt;
        float accelExt  = accel.y * (gravity < 0.f ? -1.f : 1.f);
        float accelWeight = std::clamp(accelExt / std::abs(gravity), 0.f, 1.f);
        if (!std::isfinite(accelWeight)) accelWeight = 0.f;

        rb.initial  = rb.position;
        rb.position = rb.position + rb.velocity * dt;
        if (rb.inv_mass > 0.f)
            rb.position += glm::vec3(0.f, gravity, 0.f) * (accelWeight * dt * dt);
    }

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now () - start);
	stats.warmstart_ms = elapsed.count() / 1000.f;
    start = std::chrono::system_clock::now();

    // 5. Main solver loop.
    int totalIterations = iterations + (postStabilize ? 1 : 0);

    for (int it = 0; it < totalIterations; it++) {
        float currentAlpha = alpha;
        if (postStabilize)
            currentAlpha = (it < iterations) ? 1.f : 0.f;

        // ── Primal update ──────────────────────────────────────────────────
        for (uint32_t bi = 0; bi < (uint32_t)bodies.size(); bi++) {
            if (!active[bi] || bodies[bi].inv_mass == 0.f) continue;
            RigidBody2& rb = bodies[bi];

            Mat3 lhs = mat3_diag(rb.mass, rb.mass, rb.inertia) / (dt * dt);
            glm::vec3 rhs = lhs * (rb.position - rb.inertial);

            for (auto& f : forces) {
                if (f->bodyA != bi && f->bodyB != bi) continue;

                f->computeConstraint(currentAlpha);
                f->computeDerivatives(bi);

                for (int i = 0; i < f->rows(); i++) {
                    float lam = std::isinf(f->stiffness[i]) ? f->lambda[i] : 0.f;
                    float force_val = std::clamp(f->penalty[i] * f->C[i] + lam,
                                                 f->fmin[i], f->fmax[i]);

                    Mat3 G = mat3_diag(glm::length(f->H[i].col(0)),
                                       glm::length(f->H[i].col(1)),
                                       glm::length(f->H[i].col(2))) * std::abs(force_val);

                    rhs += f->J[i] * force_val;
                    lhs += mat3_outer(f->J[i], f->J[i] * f->penalty[i]) + G;
                }
            }

            rb.position -= mat3_solve(lhs, rhs);
        }

        // ── Dual update ───────────────
        if (it < iterations) {
            for (auto& f : forces) {
                f->computeConstraint(currentAlpha);
                for (int i = 0; i < f->rows(); i++) {
                    float lam = std::isinf(f->stiffness[i]) ? f->lambda[i] : 0.f;
                    f->lambda[i] = std::clamp(f->penalty[i] * f->C[i] + lam,
                                              f->fmin[i], f->fmax[i]);

                    if (f->lambda[i] > f->fmin[i] && f->lambda[i] < f->fmax[i])
                        f->penalty[i] = std::min(f->penalty[i] + beta * std::abs(f->C[i]),
                                                 std::min(AVBD_PENALTY_MAX, f->stiffness[i]));
                }
            }
        }

        // ── Velocity update ────────────────
        if (it == iterations - 1) {
            for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
                if (!active[i]) continue;
                RigidBody2& rb = bodies[i];
                rb.prev_velocity = rb.velocity;
                if (rb.inv_mass > 0.f)
                    rb.velocity = (rb.position - rb.initial) / dt;
            }
        }
    }

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now () - start);
	stats.solver_ms = elapsed.count() / 1000.f;
}
