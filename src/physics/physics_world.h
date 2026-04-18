#pragma once

#include "lbvh.h"
#include "rigid_body.h"
#include "force.h"
#include "collision.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>

struct PhysicsStats {
    float    lbvh_ms       { 0.f };
    float    broadphase_ms { 0.f };
    float    warmstart_ms  { 0.f };
    float    predict_ms    { 0.f };
    float    solver_ms     { 0.f };
    float    total_ms      { 0.f };
    uint32_t num_bodies    { 0 };
    uint32_t num_contacts  { 0 };
};

struct PhysicsWorld {
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

    float        last_dt{ 1.f / 144.f };
    PhysicsStats stats;

    std::unordered_map<uint64_t, std::vector<ManifoldPoint>> precomputed_contacts;

    std::vector<uint32_t> colors;
    uint32_t              max_color{0};

    uint32_t    add_body(RigidBody2& rb);
    uint32_t    add_static_rect(glm::vec2 center, float w, float h);
    void        remove_body(uint32_t index);
    RigidBody2& get_body(uint32_t index);

    void prepare();
    void solve(float dt);
    void step(float dt);
};
