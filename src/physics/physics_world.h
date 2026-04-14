#pragma once

#include "lbvh.h"
#include "rigid_body.h"
#include "force.h"
#include <memory>
#include <vector>

struct PhysicsWorld {
    // Simulation parameters
    float gravity   { -10.f };
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

    uint32_t    add_body(RigidBody2& rb);
    void        remove_body(uint32_t index);
    RigidBody2& get_body(uint32_t index);

    void step(float dt);
};
