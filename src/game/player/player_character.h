#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/body_image.h>
#include <src/physics/joint.h>
#include <array>

class ResiduaEngine;

// ── Body part indices ──────────────────────────────────────────────────────────

enum class Limb : uint32_t {
    Torso = 0,
    Head,
    UpperArmL, ForearmL, HandL,
    UpperArmR, ForearmR, HandR,
    ThighL,    LowerLegL, FootL,
    ThighR,    LowerLegR, FootR,
    Count
};

// ── Joint indices ──────────────────────────────────────────────────────────────

enum class JointID : uint32_t {
    Neck = 0,     // Torso     → Head
    ShoulderL,    // Torso     → UpperArmL
    ElbowL,       // UpperArmL → ForearmL
    WristL,       // ForearmL  → HandL
    ShoulderR,    // Torso     → UpperArmR
    ElbowR,       // UpperArmR → ForearmR
    WristR,       // ForearmR  → HandR
    HipL,         // Torso     → ThighL
    KneeL,        // ThighL    → LowerLegL
    AnkleL,       // LowerLegL → FootL
    HipR,         // Torso     → ThighR
    KneeR,        // ThighR    → LowerLegR
    AnkleR,       // LowerLegR → FootR
    Count
};

struct PlayerJoint {
    DistanceJoint* joint         { nullptr }; 
    float          desired_angle { 0.f };
    float          stiffness     {10.f };
};

struct PlayerCharacter {
    static constexpr uint32_t INVALID      = ~0u;
    static constexpr uint32_t PLAYER_LAYER = 0x00000002u;
    static constexpr uint32_t PLAYER_MASK  = ~PLAYER_LAYER;

    std::array<uint32_t,    (size_t)Limb::Count>    limbs;
    std::array<PlayerJoint, (size_t)JointID::Count> joints {};

    PlayerCharacter() { limbs.fill(INVALID); }

    uint32_t hitbox_id { INVALID };

    LoadedBodyImage img_hitbox;
    LoadedBodyImage img_torso;
    LoadedBodyImage img_head;
    LoadedBodyImage img_upper_arm;
    LoadedBodyImage img_forearm;
    LoadedBodyImage img_hand;
    LoadedBodyImage img_thigh;
    LoadedBodyImage img_lower_leg;
    LoadedBodyImage img_foot;

    void load_assets();

    void spawn  (PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 position);
    void despawn(PhysicsEngine& pe);

    // ── Movement (set each frame by the caller) ────────────────────────────────
    float move_dir          { 0.f };   // -1 = left, 0 = idle, 1 = right
    float max_speed         {180.f };
    float accel             {3000.f };
    float stride_length     { 10.f };
    float upright_stiffness  { 12.f };
    float upright_damping    { 20.f };
    float max_leg_angle_speed{ 9.8f };
    float step_time          { 0.475f };
    float step_length_test   { 51.f };

    glm::vec2 foot_target_l { 0.f };
    glm::vec2 foot_target_r { 0.f };

    void set_desired_angle(JointID jid, float angle);
    void update(PhysicsEngine& pe, float dt);

    bool      is_valid()             const { return limbs[(size_t)Limb::Torso] != INVALID; }
    uint32_t  limb_id(Limb l)       const { return limbs[(size_t)l]; }
    glm::vec2 position(PhysicsEngine& pe) const;
    glm::vec2 velocity(PhysicsEngine& pe) const;
    float rotation(PhysicsEngine& pe) const;

private:
    // running animation counter
    float stride_counter {0.0f};
    glm::vec2 right_step_goal{}, left_step_goal{};
    glm::vec2 right_step_target{}, left_step_target{};
    bool right_airborne{ false };
    bool left_airborne{ false };
    glm::vec2 airborn_foot_offset{ glm::vec2(0.0f, 30.0f) };

    uint32_t spawn_limb(PhysicsEngine& pe, ResiduaEngine& re,
                        const LoadedBodyImage& img, glm::vec2 world_pos);

    void connect(PhysicsEngine& pe,
                 JointID jid,
                 Limb parent, glm::vec2 anchor_parent_local,
                 Limb child,  glm::vec2 anchor_child_local,
                 float stiffness, float bend_stiffness);

    void animate(PhysicsEngine& pe, float dt);

    std::optional<glm::vec2> select_next_step(PhysicsEngine& pe, bool right);
};
