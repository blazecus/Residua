#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/joint.h>
#include <src/physics/body_image.h>
#include <array>
#include <limits>
#include <unordered_map>
#include "movement_state.h"
#include "character_config.h"
#include "animation_clip.h"

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

enum class Joint : uint32_t {
    Neck = 0,
    ShoulderL, ElbowL, WristL,
    ShoulderR, ElbowR, WristR,
    HipL, KneeL, AnkleL,
    HipR, KneeR, AnkleR,
    Count
};

enum class AnimationLegState : uint32_t {
    Stationary = 0,
    Walking,
    Running
};

enum class PlayerAnimState : uint32_t {
    Standing = 0,
    Running,
    Jumping,
    Airborn
};

struct PlayerCharacter {
    static constexpr uint32_t INVALID      = ~0u;
    static constexpr uint32_t PLAYER_LAYER = 0x00000002u;
    static constexpr uint32_t PLAYER_MASK  = ~PLAYER_LAYER;

    std::array<uint32_t,       (size_t)Limb::Count>  limbs;
    std::array<DistanceJoint*, (size_t)Joint::Count> joint_ptrs;

    PlayerCharacter() { limbs.fill(INVALID); joint_ptrs.fill(nullptr); }

    CharacterConfig char_config;
    std::unordered_map<std::string, LoadedBodyImage> limb_images;
    std::array<bool,  JOINT_COUNT> joint_animatable{};
    std::array<bool,  JOINT_COUNT> joint_limited{};   
    std::array<float, JOINT_COUNT> joint_angle_min{};
    std::array<float, JOINT_COUNT> joint_angle_max{};

    AnimationClip anim_standing;

    PlayerAnimState anim_state { PlayerAnimState::Standing };
    float           anim_frame { 0.f };

    void load_assets(const char* config_path = "../assets/animations/player.json");
    void spawn   (PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 position);
    void despawn (PhysicsEngine& pe);
    void reset_to(PhysicsEngine& pe, glm::vec2 pos);

    // ── Movement (set each frame by the caller) ────────────────────────────────
    float move_dir            { 0.f };
    bool  walking             { false };
    bool  jump                { false };
    float max_speed           { 280.f };
    float accel               {3000.f };
    float max_leg_angle_speed {  9.8f };
    float step_time           { 0.46f };
    float step_length_test    { 45.f };
    float walk_speed          {100.0f };

    float air_accel                 { 600.f };
    float ground_check_dist         { 50.f };
    float jump_airborne_dist        { 90.f };
    float jump_foot_x               {  6.f };
    float jump_foot_y               {  8.f };
    float jump_foot_y_low           { 25.f };
    float max_joint_torque          { 1e8f };  
    std::array<float, JOINT_COUNT> joint_goal_angle{};
    float wall_check_dist           { 40.f };
    float arm_angle_speed           { 10.f };  
    AnimationLegState animation_leg_state { AnimationLegState::Stationary };

    glm::vec2 aim_pos {};

    float    joint_bend_stiffness { 250.f };

    float    torso_upright_stiffness { 8000.f }; 
    float    torso_upright_damping   {  600.f }; 

    float    reward_anim_match     {  10.0f   };
    float    reward_alive          {  0.01f  }; 
    float    standing_height       { 25.f   }; 

    float    reward_height         { 15.0f   }; 
    float    reward_upright        { 10.0f   };
    float    reward_feet_below     { 3.0f   };
    float    reward_feet_near_x    { 2.0f   }; 
    float    feet_near_x_threshold { 20.f   };
    float    penalty_velocity      { 0.00005f}; 
    float    penalty_torso_spin   { 1.0f    }; 
    float    penalty_action_rate   { 0.05f  }; 
    float    penalty_joint_limit   { 0.1f   }; 
    float    joint_limit_threshold { 1.5f   }; 

    float    low_speed_threshold       { 2.0f };
    float    upright_angle_threshold   { 0.3f };
    float    height_threshold          { 10.f };

    MovementStateBuffer state_history;

    void  capture_state(PhysicsEngine& pe);
    float compute_reward(float dt) const;

    void get_anim_goals(std::array<float, JOINT_COUNT>& current_out,
                        std::array<float, JOINT_COUNT>& next_out) const;

    void animate_left_arm(PhysicsEngine& pe);

    float ankle_bend_stiffness { 3.f };

    struct Action {
        std::array<float, (size_t)Joint::Count> goal_angle_change{};
    };
    Action current_action{};
    Action previous_action{};
    void apply_action(PhysicsEngine& pe, const Action& action);

    void apply_inputs(float move_dir, bool walking, bool jump, glm::vec2 aim_pos);
    void handle_controls(PhysicsEngine& pe, float dt);
    void apply_joint_goals(PhysicsEngine& pe);
    void animate(PhysicsEngine& pe, float dt);
    void test_animate(PhysicsEngine& pe);
    void update(PhysicsEngine& pe, float dt, bool apply_controls = true);

    bool      is_valid()                   const { return limbs[(size_t)Limb::Torso] != INVALID; }
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
    float upright_timer_   { 0.f };
    float height_timer_    { 0.f };
    float low_speed_timer_ { 0.f };

    float facing_dir    { 1.f };

    float arm_shoulder_rest { 0.f };
    float arm_elbow_rest    { 0.f };
    float arm_wrist_rest    { 0.f };


    float thigh_angle_l { 0.f };
    float lower_angle_l { 0.f };
    float thigh_angle_r { 0.f };
    float lower_angle_r { 0.f };

    uint32_t spawn_limb(PhysicsEngine& pe, ResiduaEngine& re,
                        const LoadedBodyImage& img, glm::vec2 world_pos,
                        float density = 0.f);

    void add_joint(PhysicsEngine& pe, Joint jnt, Limb parent, Limb child,
                   glm::vec2 rA_local, glm::vec2 rB_local,
                   float bend_stiffness = -1.f,
                   float max_torque     = std::numeric_limits<float>::infinity());

    void set_rest_angle(Joint jnt, float parent_angle, float child_angle);

    std::optional<RaycastHit> select_next_step(PhysicsEngine& pe, bool right, float x_offset = 0.0f);
    std::optional<RaycastHit> get_standing_step(PhysicsEngine& pe, bool right);

    std::pair<float,float> solve_arm_ik(glm::vec2 shoulder, glm::vec2 target) const;

    void update_grounded(PhysicsEngine& pe, float dt, glm::vec2 air_normal);
    void update_airborne(PhysicsEngine& pe, glm::vec2 air_normal);
    void interpolate_steps(float dt);

    void animate_leg_fk(Joint hip, Joint knee, Joint ankle,
                        float parent_angle,
                        float thigh_angle, float lower_angle, float foot_angle);
    void animate_legs(PhysicsEngine& pe, float dt, glm::vec2 torso_pos, float torso_angle);
    void animate_arms(PhysicsEngine& pe, glm::vec2 torso_pos, float torso_angle);
};
