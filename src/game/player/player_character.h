#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/joint.h>
#include <src/physics/body_image.h>
#include <array>
#include <optional>
#include <limits>
#include <unordered_map>
#include "character_config.h"

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

static constexpr uint32_t JOINT_COUNT = (uint32_t)Joint::Count;

enum class AnimationState : uint32_t {
    Standing = 0,
    Walking,
    Running,
    JumpSquat,
    Jumping,
    Airborne,
    Landing
};

struct LegPose {
    float thigh { 0.f };
    float lower { 0.f };
};

enum class RunPhase : uint32_t {
    Drive = 0,
    Flight,
    Reach
};

struct PlayerCharacter {
    static constexpr uint32_t INVALID      = ~0u;
    static constexpr uint32_t PLAYER_LAYER = 0x00000002u;
    static constexpr uint32_t PLAYER_MASK  = ~PLAYER_LAYER;

    std::array<uint32_t,       (size_t)Limb::Count>  limbs;
    std::array<DistanceJoint*, (size_t)Joint::Count> joint_ptrs;

    PhysicsEngine* pe { nullptr };

    AngleAnchor* torso_anchor { nullptr };

    bool      foot_anchored_l      { false };
    bool      foot_anchored_r      { false };
    bool      foot_pinned_l        { false };
    bool      foot_pinned_r        { false };
    float     foot_dyn_mass_l{}, foot_dyn_inertia_l{};
    float     foot_dyn_mass_r{}, foot_dyn_inertia_r{};
    float     foot_friction_l{}, foot_friction_r{};
    float     foot_pin_break_force { 5e8f };
    float     foot_pin_snap_dist   { 2.f };
    float     foot_ground_dist     { 0.5f };

    void anchor_foot   (bool left);
    void release_anchor(bool left);
    void sync_foot_pin  (bool left);
    bool snap_foot_to_ground(bool left);
    bool foot_grounded(bool left) const;
    std::optional<float> sole_clearance(bool left, float max_dist) const;

    PlayerCharacter() { limbs.fill(INVALID); joint_ptrs.fill(nullptr); }

    CharacterConfig char_config;
    std::unordered_map<std::string, LoadedBodyImage> limb_images;
    std::array<bool,  JOINT_COUNT> joint_animatable{};
    std::array<bool,  JOINT_COUNT> joint_limited{};   
    std::array<float, JOINT_COUNT> joint_angle_min{};
    std::array<float, JOINT_COUNT> joint_angle_max{};

    void load_assets(const char* config_path = "../assets/animations/player.json");
    void spawn   (PhysicsEngine& physics, ResiduaEngine& re, glm::vec2 position);
    void despawn ();

    // ── Movement ────────────────────────────────
    float move_dir            { 0.f };
    bool  walking             { false };
    bool  jump                { false };
    float max_leg_angle_speed { 13.0f };
    float walk_leg_angle_speed {  5.0f };  

    float step_period         { 0.6f };
    float backward_step_period { 0.4f };   
    float backward_run_step_period     { 0.25f };  
    float backward_run_leg_angle_speed { 7.5f };  

    // ── Gait cycle tuning ──────
    float feet_gap                  { 5.0f };  
    float swing_search_offset_x     { 5.f };
    float swing_search_back         { 3.f };   
    float stance_lean_angle         { 1.1f };  
    float recovery_bend_angle       { 1.0f };
    float swing_thigh_forward_angle { 1.0f };  
    float standing_lower_lean_angle { 0.5236f }; 
    float standing_thigh_bend_angle { -0.8f };  
    float foot_land_dist            { 4.f };   
    float standing_origin_y         { 3.f };  
    float standing_settle_speed     { 5.f };
    float settle_step_dist          { 3.f };
    float trailing_kick_angle       { 1.0f };
    float backward_knee_bend        { 0.3f };  
    float backward_stance_lean_angle{ 0.8f };  
    float backward_search_offset_x  { 8.f };   
    float backward_swing_knee_bend  { 2.5f };  
    float backward_swing_lift_angle { 0.4f };  
    float run_speed_target          { 150.f };
    float run_speed_kp              { 10.f };
    float run_speed_kd              { 0.5f };
    float run_max_accel             { 600.f };
    float run_release_angle         { 0.45f };
    float run_drive_lead            { 3.f };   
    float run_drive_lift            { 1.f };  
    float run_swing_thigh_angle     { 1.745f };
    float run_tuck_knee_bend        { 2.0f };
    float run_lift_speed            { 0.0f };
    float run_flight_delay          { 0.075f };
    float run_step_ahead            { 10.f };
    float run_stop_speed            { 15.f };
    float run_stop_decel            { 300.f };
    float run_airborne_delay        { 0.5f };
    float run_foot_point_angle      { 0.7f };  
    float foot_angle_speed          { 40.f };    
    float jump_impulse              { 100000.f };
    float jump_squat_thigh_angle    { 1.0f };
    float jump_squat_knee_bend      { 2.0f };
    float jump_squat_sink_speed     { 30.f };  
    float jump_squat_min_hip_height { 10.f }; 
    float jump_rise_thigh_l         { 1.75f };
    float jump_rise_thigh_r         { 2.1f };
    float jump_rise_knee_bend       { 1.775f };
    float jump_fall_thigh_l         { 0.3f };
    float jump_fall_thigh_r         { 0.6f };
    float jump_fall_knee_bend       { 0.8f };   
    float air_leg_angle_speed       { 4.f };
    float jump_fall_delay           { 0.5f };  
    std::array<float, JOINT_COUNT> joint_goal_angle{};
    float arm_angle_speed           { 30.f };
    float support_arm_offset_angle  { 1.309f };
    float turn_deadzone             { 1.f };   

    glm::vec2 aim_pos {};

    float          facing()          const { return facing_dir; }


    float    joint_bend_hz            { 1500.f }; 
    float    joint_bend_damping_ratio { 1.f };   
    float    joint_bend_max_accel     { 1e8f };

    void animate_left_arm();
    void animate_right_arm();

    void apply_inputs(float move_dir, bool walking, bool jump, glm::vec2 aim_pos);
    void apply_joint_goals();
    void update(float dt, bool apply_controls = true);

    bool      is_valid()                   const { return limbs[(size_t)Limb::Torso] != INVALID; }
    glm::vec2 position() const;
    glm::vec2 velocity() const;
    float     rotation() const;

private:
    glm::vec2 right_step_target  {};
    glm::vec2 left_step_target   {};

    float facing_dir    { 1.f };

    float arm_shoulder_rest { 0.f };
    float arm_elbow_rest    { 0.f };
    float arm_wrist_rest    { 0.f };
    float arm_shoulder_rest_r { 0.f };
    float arm_elbow_rest_r    { 0.f };
    float arm_wrist_rest_r    { 0.f };

    LegPose leg_angle_l {};
    LegPose leg_angle_r {};
    float   foot_angle_l {};  
    float   foot_angle_r {};
    LegPose leg_target_l {};  
    LegPose leg_target_r {};

    bool grounded_l { false };  
    bool grounded_r { false };

    float gait_timer   { 0.f };
    bool  stance_left  { true };

    LegPose stance_phase_start {};
    LegPose swing_phase_start  {};

    float jump_timer { 0.0f };
    float jump_dir_timer { 0.0f };
    bool will_jump { false };
    bool jump_launched { false };

    AnimationState anim_state { AnimationState::Standing };
    float          state_time { 0.f };
    float          air_time   { 0.f };
    float          fall_time  { 0.f };   

    RunPhase run_phase      { RunPhase::Drive };
    float    run_phase_time { 0.f };
    float    run_prev_vel_x { 0.f };
    float    squat_vel_x    { 0.f };   
    bool     squat_moving   { false }; 
    bool     squat_left     { true };  

    bool settling { false };
    float standing_center_x { 0.f };
    float arm_elbow_side { 1.f };

    // ── Left/right accessors ──
    uint32_t   foot_id(bool left)       const { return limbs[(size_t)(left ? Limb::FootL : Limb::FootR)]; }
    bool       foot_anchored(bool left) const { return left ? foot_anchored_l : foot_anchored_r; }
    bool       foot_pinned(bool left)   const { return left ? foot_pinned_l   : foot_pinned_r;   }
    bool       grounded(bool left)      const { return left ? grounded_l : grounded_r; }
    glm::vec2& step_target(bool left)         { return left ? left_step_target : right_step_target; }
    LegPose&   leg_angle(bool left)           { return left ? leg_angle_l : leg_angle_r; }
    float&     foot_angle(bool left)          { return left ? foot_angle_l : foot_angle_r; }
    LegPose&   leg_target(bool left)          { return left ? leg_target_l : leg_target_r; }

    float     side_x(bool left) const { return (left ? -1.f : 1.f) * facing_dir; }
    glm::vec2 hip_position(bool left) const;
    glm::vec2 ankle_position(bool left) const;   // world-space ankle joint anchor on the foot
    bool      walking_backward() const { return move_dir * facing_dir < 0.f; }
    bool      running_backward() const { return walking_backward() && !walking; }
    bool      wants_turn()       const { return (aim_pos.x - position().x) * facing_dir < -turn_deadzone; }
    bool      run_stopping()     const { return anim_state == AnimationState::Running &&
                                                (move_dir * facing_dir <= 0.f || wants_turn() ||
                                                 velocity().x * facing_dir < -run_stop_speed); }
    float     run_travel_dir()   const;
    float     run_stop_scale()   const;

    void update_facing();
    float com_velocity_x() const;
    void  add_velocity_x(float dv);
    void set_facing(float dir);

    // ── animation helpers ──
    bool    scan_step_target(bool left, float x_from, float x_to);
    bool    foot_near_step_target(bool left);
    bool    try_land_foot(bool left);
    bool    settle_foot(bool left, float target_x);
    LegPose reach_step_target(glm::vec2 hip, bool left);
    LegPose standing_planted_pose() const;
    LegPose walking_stance_pose(float t) const;
    LegPose walking_swing_pose(bool swing_left, float t);
    LegPose backward_bent_pose(float thigh, float knee_bend) const;
    LegPose backward_stance_pose(float t) const;
    LegPose backward_swing_pose(bool swing_left, float t);
    LegPose crouch_pose() const;
    LegPose air_pose(bool left) const;
    void    begin_gait_phase();

    // ── update() stages ──
    AnimationState next_animation_state() const;
    void update_animation_state(float dt);
    void enter_animation_state(AnimationState next);
    void set_animation_goals(float dt);
    void anchor_grounded_feet();
    void update_standing_anchors();
    void update_settling();
    void update_standing();
    void update_walking(float dt);
    void begin_run_cycle();
    void update_running(float dt);
    void enter_run_phase(RunPhase phase);
    void run_speed_control(float dt);
    float leg_over_foot_angle(bool left, float dir) const;
    void drive_legs(float dt);
    void handle_jump(float dt);
    void launch_jump();

    uint32_t spawn_limb(ResiduaEngine& re,
                        const LoadedBodyImage& img, glm::vec2 world_pos,
                        float density = 0.f);

    void add_joint(Joint jnt, Limb parent, Limb child,
                   glm::vec2 rA_local, glm::vec2 rB_local,
                   float bend_stiffness = -1.f,
                   float max_torque     = std::numeric_limits<float>::infinity());

    void set_rest_angle(Joint jnt, float parent_angle, float child_angle);

    std::pair<float,float> solve_arm_ik(glm::vec2 shoulder, glm::vec2 target, float& elbow_side);

    void animate_leg_fk(Joint hip, Joint knee, Joint ankle,
                        float parent_angle,
                        float thigh_angle, float lower_angle, float foot_angle);
};
