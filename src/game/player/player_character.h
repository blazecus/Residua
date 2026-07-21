#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/body_image.h>

class ResiduaEngine;

struct PlayerCharacter {
    static constexpr uint32_t INVALID = ~0u;

    uint32_t        body_id { INVALID };
    LoadedBodyImage sprite;

    float thrust_force { 50000.f };
    float rotate_speed  { 3.f };   

    float thrust_input { 0.f };   
    float rotate_input { 0.f };   

    void load_assets(const char* path = "../assets/player/triangle.png");
    void spawn   (PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 position);
    void despawn (PhysicsEngine& pe);
    void reset_to(PhysicsEngine& pe, glm::vec2 pos);

    void apply_inputs(float thrust, float rotate);
    void update(PhysicsEngine& pe, float dt);

    bool      is_valid()                  const { return body_id != INVALID; }
    glm::vec2 position(PhysicsEngine& pe) const;
    glm::vec2 velocity(PhysicsEngine& pe) const;
    float     rotation(PhysicsEngine& pe) const;
};
