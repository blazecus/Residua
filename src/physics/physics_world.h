#pragma once

#include "lbvh.h"
#include "rigid_body.h"
#include "force.h"
#include <memory>
#include <vector>
#include <chrono>

struct PhysicsStats {
    float    lbvh_ms       { 0.f };
    float    broadphase_ms { 0.f };
    float    warmstart_ms  { 0.f };
    float    predict_ms    { 0.f };
    float    solver_ms     { 0.f };
    float    total_ms      { 0.f };
    uint32_t num_bodies    { 0 };
};

struct PhysicsWorld {
    // Simulation parameters
    float gravity   { 100.f };
    float alpha     { AVBD_ALPHA };
    float beta      { AVBD_BETA  };
    float gamma     { AVBD_GAMMA };
    int   iterations{ 10 };
    bool  postStabilize{ true };

    std::vector<RigidBody2>  bodies;
    std::vector<bool>        active;
    std::vector<uint32_t>    open_slots;

    std::vector<std::unique_ptr<Force>> forces;

    LBVH lbvh;

    float        last_dt{ 1.f / 60.f };
    PhysicsStats stats;

    uint32_t    add_body(RigidBody2& rb);
    // Add an immovable rectangular obstacle (no sprite, not drawn).
    // center is in world-space pixels; w/h are the full dimensions.
    uint32_t    add_static_rect(glm::vec2 center, float w, float h);
    void        remove_body(uint32_t index);
    RigidBody2& get_body(uint32_t index);

    void step(float dt);
};
