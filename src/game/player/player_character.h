#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/body_image.h>
#include <array>

class ResiduaEngine;

enum class Limb : uint32_t {
    Torso = 0,
    Head,
    UpperArmL, ForearmL, HandL,
    UpperArmR, ForearmR, HandR,
    ThighL,    LowerLegL, FootL,
    ThighR,    LowerLegR, FootR,
    Count
};

enum class AnimationLegState : uint32_t {
    Stationary = 0,
    Walking,
    Running
};

struct PlayerCharacter {
    static constexpr uint32_t INVALID      = ~0u;
    static constexpr uint32_t PLAYER_LAYER = 0x00000002u;
    static constexpr uint32_t PLAYER_MASK  = ~PLAYER_LAYER;

    std::array<uint32_t, (size_t)Limb::Count> limbs;
    uint32_t hitbox_id { INVALID };

    PlayerCharacter() { limbs.fill(INVALID); }

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
    float move_dir            { 0.f };
    bool  walking             { false };
    bool  jump                { false };
    float max_speed           { 280.f };
    float accel               {3000.f };
    float upright_stiffness   {  6.f };
    float upright_damping     { 20.f };
    float forward_lean        { 0.018f };
    float backward_lean       { 0.06f };
    float max_leg_angle_speed {  9.8f };
    float stride_height       {  5.f };
    float step_time           { 0.46f };
    float step_length_test    { 45.f };
    float step_interp_k       {  0.5f };
    float walk_speed          {100.0f };
    float stationary_speed    { 10.0f };
    float arm_swing_amplitude    {  0.4f };
    float ground_spring_k           {5000.f };
    float ground_spring_damping     { 20.f };
    float ground_spring_rest_length { 30.f };
    float ground_spring_max_dist    { 60.f };
    float jump_impulse              { 200.f };
    float air_accel                 { 600.f };
    float jump_spring_disable_time  {  0.3f };
    float ground_check_dist         { 50.f };
    float jump_airborne_dist        { 90.f };
    float jump_foot_x               {  6.f };
    float jump_foot_y               {  8.f };
    float jump_foot_y_low           { 25.f };
    AnimationLegState animation_leg_state { AnimationLegState::Stationary };

    glm::vec2 aim_pos {};  // physics-space mouse position, set each frame by the caller

    void apply_inputs(float move_dir, bool walking, bool jump, glm::vec2 aim_pos);
    void handle_controls(PhysicsEngine& pe, float dt);
    void animate(PhysicsEngine& pe, float dt);
    void update(PhysicsEngine& pe, float dt);

    bool      is_valid()                   const { return hitbox_id != INVALID; }
    uint32_t  limb_id(Limb l)             const { return limbs[(size_t)l]; }
    glm::vec2 position(PhysicsEngine& pe) const;
    glm::vec2 velocity(PhysicsEngine& pe) const;
    float     rotation(PhysicsEngine& pe) const;

private:
    float     stride_counter     { 0.f };
    glm::vec2 right_step_goal    {};
    glm::vec2 left_step_goal     {};
    glm::vec2 right_step_target  {};
    glm::vec2 left_step_target   {};
    glm::vec2 right_step_normal         { 0.f, -1.f };
    glm::vec2 left_step_normal          { 0.f, -1.f };
    glm::vec2 right_step_normal_target  { 0.f, -1.f };
    glm::vec2 left_step_normal_target   { 0.f, -1.f };
    bool      right_airborne     { false };
    bool      left_airborne      { false };
    glm::vec2 airborn_foot_offset{ -2.f, 5.f };

    bool  grounded      { false };
    bool  was_grounded  { false };
    float jump_timer    { 0.f };

    float facing_dir    { 1.f };  // +1 = right, -1 = left

    // Rate-limited world-space leg angles
    float thigh_angle_l { 0.f };
    float lower_angle_l { 0.f };
    float thigh_angle_r { 0.f };
    float lower_angle_r { 0.f };

    // Spawn a static visual body — no mass, no collision response
    uint32_t spawn_limb(PhysicsEngine& pe, ResiduaEngine& re,
                        const LoadedBodyImage& img, glm::vec2 world_pos);

    void place(PhysicsEngine& pe, Limb limb, glm::vec2 center, float angle);

    std::optional<RaycastHit> select_next_step(PhysicsEngine& pe, bool right, float x_offset = 0.0f);
    std::optional<RaycastHit> get_standing_step(PhysicsEngine& pe, bool right);

    std::pair<float,float> solve_arm_ik(glm::vec2 shoulder, glm::vec2 target) const;

    void update_grounded(PhysicsEngine& pe, float dt, glm::vec2 air_normal);
    void update_airborne(PhysicsEngine& pe, glm::vec2 air_normal);
    void interpolate_steps(float dt);

    void animate_leg_fk(PhysicsEngine& pe, Limb thigh, Limb lower_leg, Limb foot,
                        glm::vec2 hip, float thigh_angle, float lower_angle, float foot_angle);
    void animate_legs(PhysicsEngine& pe, float dt, glm::vec2 torso_pos, float hbox_angle);
    void animate_arms(PhysicsEngine& pe, glm::vec2 torso_pos, float hbox_angle);
};
