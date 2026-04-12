#pragma once

#include "../physics/lbvh.h"
#include "../physics/rigid_body.h"
#include "../physics/collision.h"

static constexpr float WARM_START_SCALE = 0.99f;

struct ContactConstraint {
    glm::vec2 position;  // world-space position, used to match against next frame
    float     lambda_n;  // normal Lagrange multiplier
    float     lambda_t;  // tangential Lagrange multiplier (friction)
};

struct CollisionPair {
    uint32_t a, b;
    std::vector<ManifoldPoint>     contacts;
    std::vector<ContactConstraint> constraints;  // parallel to contacts, warm start state
    float stiffness;                             // shared across all contacts in this pair
};

struct PhysicsWorld {
    std::vector<RigidBody2> bodies;
    std::vector<bool>       active;
    std::vector<uint32_t>   open_slots;

    LBVH                       lbvh;
    std::vector<CollisionPair> collision_pairs;

    glm::vec2 gravity{0.f, -9.8f};

    uint32_t add_body(RigidBody2& new_body);
    void     remove_body(uint32_t index);
    RigidBody2& get_body(uint32_t index);

    void step(float dt);
};
