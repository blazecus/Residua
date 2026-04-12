#include "../physics/physics_world.h"
#include <glm/gtx/rotate_vector.hpp>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static AABB compute_world_aabb(const RigidBody2& rb) {
    glm::vec2 mn( 1e20f,  1e20f);
    glm::vec2 mx(-1e20f, -1e20f);
    for (const glm::vec2& v : rb.shape) {
        glm::vec2 w = rb.position + glm::rotate(v, rb.rotation);
        mn = glm::min(mn, w);
        mx = glm::max(mx, w);
    }
    return { mn, mx };
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
}

RigidBody2& PhysicsWorld::get_body(uint32_t index) {
    return bodies[index];
}

// ─── Step ─────────────────────────────────────────────────────────────────────
// TODO: refactor
void PhysicsWorld::step(float dt) {
    // 1. Build LBVH from world-space AABBs of all active bodies.
    std::vector<AABB>     boxes;
    std::vector<uint32_t> index_map;  // boxes[i] → bodies[index_map[i]]

    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        boxes.push_back(compute_world_aabb(bodies[i]));
        index_map.push_back(i);
    }

    lbvh_build(lbvh, boxes);

    // 2. Broad phase: query each body's AABB against the LBVH,
    //    then narrow phase: build + reduce contact manifold for each candidate pair.
    std::vector<CollisionPair> prev_pairs = std::move(collision_pairs);
    collision_pairs.clear();
    std::vector<int> candidates;

    for (uint32_t li = 0; li < (uint32_t)index_map.size(); li++) {
        lbvh_query_AABB(lbvh, boxes[li], candidates);

        for (int lj : candidates) {
            if (lj <= (int)li) continue;  

            uint32_t i = index_map[li];
            uint32_t j = index_map[lj];

            auto contacts = build_manifold(bodies[i], bodies[j]);
            if (contacts.empty()) continue;

            contacts = reduce_manifold(std::move(contacts), 4);

            CollisionPair pair;
            pair.a         = i;
            pair.b         = j;
            pair.contacts  = std::move(contacts);
            pair.stiffness = 1e6f;

            // Match contacts to previous frame by proximity and copy raw lambdas.
            // Scaling happens later in the adaptive init step, after prediction.
            pair.constraints.resize(pair.contacts.size(), { {}, 0.f, 0.f });
            const CollisionPair* prev = nullptr;
            for (const CollisionPair& p : prev_pairs)
                if (p.a == i && p.b == j) { prev = &p; break; }

            for (size_t ci = 0; ci < pair.contacts.size(); ci++) {
                glm::vec2 pos = pair.contacts[ci].position;
                pair.constraints[ci].position = pos;

                if (!prev) continue;

                float best_dist = std::numeric_limits<float>::max();
                const ContactConstraint* best = nullptr;
                for (const ContactConstraint& pc : prev->constraints) {
                    glm::vec2 d = pos - pc.position;
                    float dist  = glm::dot(d, d);
                    if (dist < best_dist) { best_dist = dist; best = &pc; }
                }

                if (best) {
                    pair.constraints[ci].lambda_n = best->lambda_n;
                    pair.constraints[ci].lambda_t = best->lambda_t;
                }
            }

            collision_pairs.push_back(std::move(pair));
        }
    }

    // 3. Prediction: save previous state, integrate external forces into position.
    // TODO: refactor into external forces function for prediction
    for (uint32_t i = 0; i < (uint32_t)bodies.size(); i++) {
        if (!active[i]) continue;
        RigidBody2& rb = bodies[i];

        rb.position_prev = rb.position;
        rb.rotation_prev = rb.rotation;

        // v += f_ext * inv_mass * dt  (e.g. gravity)
        rb.velocity         += gravity * rb.inv_mass * dt;
        rb.position         += rb.velocity * dt;
        rb.rotation         += rb.angular_velocity * dt;
    }

    // 4. Warm start + adaptive initialization (both happen post-prediction).
    //    Scale the carried-over lambdas, then shift toward resolving current depth.
    for (CollisionPair& pair : collision_pairs) {
        RigidBody2& ba = bodies[pair.a];
        RigidBody2& bb = bodies[pair.b];

        float alpha_tilde = 1.f / (pair.stiffness * dt * dt);

        for (size_t ci = 0; ci < pair.contacts.size(); ci++) {
            const ManifoldPoint& mp = pair.contacts[ci];
            ContactConstraint&   cc = pair.constraints[ci];

            glm::vec2 ra = mp.position - ba.position;
            glm::vec2 rb = mp.position - bb.position;

            float ra_x_n = ra.x * mp.normal.y - ra.y * mp.normal.x;
            float rb_x_n = rb.x * mp.normal.y - rb.y * mp.normal.x;

            float w = ba.inv_mass + bb.inv_mass
                    + ra_x_n * ra_x_n * ba.inv_inertia
                    + rb_x_n * rb_x_n * bb.inv_inertia;

            // Warm start scale, then adaptive correction from current penetration.
            float lambda_warm = cc.lambda_n * WARM_START_SCALE;
            cc.lambda_n = std::max(0.f,
                lambda_warm + mp.depth / (w + alpha_tilde));

            cc.lambda_t *= WARM_START_SCALE;
        }
    }

    // 5. AVBD constraint solve (TODO)
    // 6. Velocity update: v = (position - position_prev) / dt  (TODO)
}
