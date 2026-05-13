#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <src/game/debug_draw.h>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>

static constexpr float TORSO_ANG_DAMPING = 0.99f;
static constexpr float TORSO_HALF_H      = 10.f;

// ── Limb skeleton ---------------------------------────────────────────────────
static constexpr float THIGH_LEN        = 10.f;
static constexpr float THIGH_MID        =  5.f;
static constexpr float LOWER_LEN        = 10.f;
static constexpr float LOWER_MID        =  5.f;
static constexpr float LOWER_TO_ANKLE   = 10.f;
static constexpr float HIP_X            =  3.f;
static constexpr float HIP_Y            =  8.f;
static constexpr float ANKLE_RAISE      =  3.f;
static constexpr float FOOT_OFF_X       =  1.f;
static constexpr float FOOT_OFF_Y       =  1.f;

// ── Head / neck ───────────────────────────────────────────────────────────────
static constexpr float NECK_Y           = -10.f;
static constexpr float HEAD_Y           =  -4.f;

// ── Arm skeleton ──────────────────────────────────────────────────────────────
static constexpr float SHOULDER_X       =  4.f;
static constexpr float SHOULDER_Y       = -8.f;
static constexpr float UPPER_ARM_LEN    = 9.f;
static constexpr float UPPER_ARM_MID    =  5.f;
static constexpr float FOREARM_LEN      = 9.f;
static constexpr float FOREARM_MID      =  5.f;
static constexpr float GRAB_BLEND       =  0.5f;

// ── Animation ─────────────────────────────────────────────────────────────────
static constexpr float AIR_NORMAL_TILT   =  2.0f;
static constexpr float INTERP_RUNNING    = 12.f;
static constexpr float INTERP_OTHER      =  7.f;
static constexpr float INTERP_AIR_MULT   =  3.f;
static constexpr float AIRBORNE_RAY_DIST = 500.f;
static constexpr float RUN_STEP_BASE_X   =  5.f;
static constexpr float RUN_STEP_LEFT_OFF = 10.f;
static constexpr float RUN_ARC_X         = 16.f;
static constexpr float RUN_ARC_Y         =  5.f;
static constexpr float WALK_STRIDE_RATE  =  1.5f;

// ── Step detection ────────────────────────────────────────────────────────────
static constexpr float STEP_RAY_DIST      =  80.f;
static constexpr float STEP_HEIGHT_THRESH =  20.f;
static constexpr float STEP_HEIGHT_NUDGE  =  -3.f;
static constexpr float STEP_VEL_MIN       =   0.1f;
static constexpr float STEP_WALK_MULT     =   1.25f;
static constexpr float STEP_RIGHT_EXTRA   =   5.f;
static constexpr float STANDING_STEP_X    =  12.f;
static constexpr float STANDING_ORIGIN_Y  =   3.f;
static constexpr float SPRING_MIN_DIST    =   0.001f;

// ─────────────────────────────────────────────────────────────────────────────

void PlayerCharacter::load_assets()
{
    img_torso     = load_body_image("../assets/player/torso.png");
    img_head      = load_body_image("../assets/player/head.png");
    img_upper_arm = load_body_image("../assets/player/upper_arm.png");
    img_forearm   = load_body_image("../assets/player/forearm.png");
    img_hand      = load_body_image("../assets/player/hand.png");
    img_thigh     = load_body_image("../assets/player/thigh.png");
    img_lower_leg = load_body_image("../assets/player/lower_leg.png");
    img_foot      = load_body_image("../assets/player/foot.png");
}

uint32_t PlayerCharacter::spawn_limb(PhysicsEngine& pe, ResiduaEngine& re,
                                      const LoadedBodyImage& img, glm::vec2 world_pos)
{
    RigidBody rb;
    rb.sprite          = img;
    rb.compute_mass_properties();
    rb.generate_shape();
    rb.generate_sdf();
    rb.position        = glm::vec3(world_pos, 0.f);
    rb.collision_layer = PLAYER_LAYER;
    rb.collision_mask  = PLAYER_MASK;
    return pe.add_body(&re, std::move(rb));
}

void PlayerCharacter::add_joint(PhysicsEngine& pe, Joint jnt, Limb parent, Limb child,
                                 glm::vec2 rA_local, glm::vec2 rB_local)
{
    auto j = std::make_unique<DistanceJoint>(
        &pe.world,
        limbs[(size_t)parent], limbs[(size_t)child],
        rA_local, rB_local);
    joint_ptrs[(size_t)jnt] = j.get();
    pe.world.add_force(std::move(j));
}

void PlayerCharacter::set_rest_angle(Joint jnt, float parent_angle, float child_angle)
{
    if (auto* j = joint_ptrs[(size_t)jnt])
        j->rest_angle = parent_angle - child_angle;
}

void PlayerCharacter::spawn(PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 pos)
{
    limbs.fill(INVALID);
    joint_ptrs.fill(nullptr);

    // torso
    {
        RigidBody rb;
        rb.sprite          = img_torso;
        rb.compute_mass_properties();
        rb.generate_shape();
        rb.generate_sdf();
        rb.position        = glm::vec3(pos, 0.f);
        rb.collision_layer = PLAYER_LAYER;
        rb.collision_mask  = PLAYER_MASK;
        rb.angular_damping = TORSO_ANG_DAMPING;
        limbs[(size_t)Limb::Torso] = pe.add_body(&re, std::move(rb));
    }

    limbs[(size_t)Limb::Head]      = spawn_limb(pe, re, img_head,      pos + glm::vec2(0.f,        NECK_Y + HEAD_Y));
    limbs[(size_t)Limb::UpperArmL] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2(-SHOULDER_X, SHOULDER_Y + UPPER_ARM_MID));
    limbs[(size_t)Limb::ForearmL]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2(-SHOULDER_X, SHOULDER_Y + UPPER_ARM_LEN + FOREARM_MID));
    limbs[(size_t)Limb::HandL]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2(-SHOULDER_X, SHOULDER_Y + UPPER_ARM_LEN + FOREARM_LEN));
    limbs[(size_t)Limb::UpperArmR] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2( SHOULDER_X, SHOULDER_Y + UPPER_ARM_MID));
    limbs[(size_t)Limb::ForearmR]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2( SHOULDER_X, SHOULDER_Y + UPPER_ARM_LEN + FOREARM_MID));
    limbs[(size_t)Limb::HandR]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2( SHOULDER_X, SHOULDER_Y + UPPER_ARM_LEN + FOREARM_LEN));
    limbs[(size_t)Limb::ThighL]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2(-HIP_X,      HIP_Y + THIGH_MID));
    limbs[(size_t)Limb::LowerLegL] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2(-HIP_X,      HIP_Y + THIGH_LEN + LOWER_MID));
    limbs[(size_t)Limb::FootL]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2(-HIP_X,      HIP_Y + THIGH_LEN + LOWER_LEN));
    limbs[(size_t)Limb::ThighR]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2( HIP_X,      HIP_Y + THIGH_MID));
    limbs[(size_t)Limb::LowerLegR] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2( HIP_X,      HIP_Y + THIGH_LEN + LOWER_MID));
    limbs[(size_t)Limb::FootR]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2( HIP_X,      HIP_Y + THIGH_LEN + LOWER_LEN));

    // Neck 
    add_joint(pe, Joint::Neck,
              Limb::Torso, Limb::Head,
              {0.f, -TORSO_HALF_H}, {0.f, 4.f});

    // Left arm
    add_joint(pe, Joint::ShoulderL,
              Limb::Torso,     Limb::UpperArmL,
              {-SHOULDER_X, SHOULDER_Y}, {0.f, -UPPER_ARM_MID});
    add_joint(pe, Joint::ElbowL,
              Limb::UpperArmL, Limb::ForearmL,
              {0.f, UPPER_ARM_MID}, {0.f, -FOREARM_MID});
    add_joint(pe, Joint::WristL,
              Limb::ForearmL,  Limb::HandL,
              {0.f, FOREARM_MID}, {0.f, 0.f});

    // Right arm
    add_joint(pe, Joint::ShoulderR,
              Limb::Torso,     Limb::UpperArmR,
              {SHOULDER_X, SHOULDER_Y}, {0.f, -UPPER_ARM_MID});
    add_joint(pe, Joint::ElbowR,
              Limb::UpperArmR, Limb::ForearmR,
              {0.f, UPPER_ARM_MID}, {0.f, -FOREARM_MID});
    add_joint(pe, Joint::WristR,
              Limb::ForearmR,  Limb::HandR,
              {0.f, FOREARM_MID}, {0.f, 0.f});

    // Left leg
    add_joint(pe, Joint::HipL,
              Limb::Torso,    Limb::ThighL,
              {-HIP_X, HIP_Y}, {0.f, -THIGH_MID});
    add_joint(pe, Joint::KneeL,
              Limb::ThighL,    Limb::LowerLegL,
              {0.f, THIGH_MID}, {0.f, -LOWER_MID});
    add_joint(pe, Joint::AnkleL,
              Limb::LowerLegL, Limb::FootL,
              {0.f, LOWER_MID}, {0.f, 0.f});

    // Right leg
    add_joint(pe, Joint::HipR,
              Limb::Torso,    Limb::ThighR,
              {HIP_X, HIP_Y}, {0.f, -THIGH_MID});
    add_joint(pe, Joint::KneeR,
              Limb::ThighR,    Limb::LowerLegR,
              {0.f, THIGH_MID}, {0.f, -LOWER_MID});
    add_joint(pe, Joint::AnkleR,
              Limb::LowerLegR, Limb::FootR,
              {0.f, LOWER_MID}, {0.f, 0.f});

    // render layers
    for (Limb l : { Limb::ThighR, Limb::LowerLegR, Limb::FootR,
                    Limb::UpperArmR, Limb::ForearmR, Limb::HandR })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 1;
    for (Limb l : { Limb::Torso, Limb::Head })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 2;
    for (Limb l : { Limb::ThighL, Limb::LowerLegL, Limb::FootL,
                    Limb::UpperArmL, Limb::ForearmL, Limb::HandL })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 3;
}

void PlayerCharacter::despawn(PhysicsEngine& pe)
{
    if (!is_valid()) return;

    for (auto* j : joint_ptrs)
        if (j) pe.world.remove_force(j);
    joint_ptrs.fill(nullptr);

    for (uint32_t id : limbs)
        if (id != INVALID) pe.remove_body(id);
    limbs.fill(INVALID);
}

static std::pair<float,float> solve_leg_ik(glm::vec2 hip, glm::vec2 ankle_target,
                                            float knee_dir = 1.f)
{
    glm::vec2 diff  = ankle_target - hip;
    float dist      = std::clamp(glm::length(diff),
                                 std::abs(THIGH_LEN - LOWER_LEN) + 0.01f,
                                 THIGH_LEN + LOWER_LEN - 0.01f);
    float cos_a     = (THIGH_LEN*THIGH_LEN + dist*dist - LOWER_LEN*LOWER_LEN) / (2.f * THIGH_LEN * dist);
    float alpha     = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float chain_phi = std::atan2(-diff.x, diff.y);
    float thigh_angle = chain_phi - knee_dir * alpha;

    glm::vec2 knee = hip + glm::vec2(-THIGH_LEN * std::sin(thigh_angle),
                                      THIGH_LEN * std::cos(thigh_angle));
    glm::vec2 ld   = ankle_target - knee;
    return { thigh_angle, std::atan2(-ld.x, ld.y) };
}

void PlayerCharacter::apply_inputs(float md, bool walk, bool jmp, glm::vec2 aim)
{
    move_dir = md;
    walking  = walk;
    jump     = jmp;
    aim_pos  = aim;
}

void PlayerCharacter::handle_controls(PhysicsEngine& pe, float dt)
{
    if (!is_valid()) return;

    uint32_t  torso_id  = limbs[(size_t)Limb::Torso];
    glm::vec2 torso_pos = pe.get_position(torso_id);
    glm::vec2 torso_vel = pe.get_velocity(torso_id);

    {
        auto hit = pe.raycast(torso_pos, glm::vec2(0.f, 1.f), ground_check_dist, PLAYER_MASK);
        grounded = hit.has_value();
    }

    if (grounded) {
        auto apply_ground_spring = [&](glm::vec2 contact, bool airborne) {
            if (airborne) return;
            glm::vec2 delta = contact - torso_pos;
            float dist = glm::length(delta);
            if (dist > SPRING_MIN_DIST && dist < ground_spring_max_dist) {
                glm::vec2 dir     = delta / dist;
                float     rel_vel = glm::dot(torso_vel, dir);
                pe.apply_force(torso_id,
                    (ground_spring_k * (dist - ground_spring_rest_length)
                     - ground_spring_damping * rel_vel) * dir);
            }
        };
        apply_ground_spring(left_step_target,  left_airborne);
        apply_ground_spring(right_step_target, right_airborne);
    }

    float target_vx     = move_dir * (walking ? walk_speed : max_speed);
    float current_accel = grounded ? accel : air_accel;
    pe.apply_force(torso_id, { (target_vx - torso_vel.x) * current_accel, 0.f });
}

void PlayerCharacter::update(PhysicsEngine& pe, float dt)
{
    handle_controls(pe, dt);
    //animate(pe, dt);
    capture_state(pe);
}

// Joint enum order → parent/child limb pairs (must match Joint enum in player_character.h)
static constexpr struct { Limb parent; Limb child; } JOINT_LIMBS[(size_t)Joint::Count] = {
    { Limb::Torso,     Limb::Head      },  // Neck
    { Limb::Torso,     Limb::UpperArmL },  // ShoulderL
    { Limb::UpperArmL, Limb::ForearmL  },  // ElbowL
    { Limb::ForearmL,  Limb::HandL     },  // WristL
    { Limb::Torso,     Limb::UpperArmR },  // ShoulderR
    { Limb::UpperArmR, Limb::ForearmR  },  // ElbowR
    { Limb::ForearmR,  Limb::HandR     },  // WristR
    { Limb::Torso,     Limb::ThighL    },  // HipL
    { Limb::ThighL,    Limb::LowerLegL },  // KneeL
    { Limb::LowerLegL, Limb::FootL     },  // AnkleL
    { Limb::Torso,     Limb::ThighR    },  // HipR
    { Limb::ThighR,    Limb::LowerLegR },  // KneeR
    { Limb::LowerLegR, Limb::FootR     },  // AnkleR
};
static_assert((size_t)Joint::Count == JOINT_COUNT, "JOINT_COUNT out of sync with Joint enum");

void PlayerCharacter::apply_action(PhysicsEngine& pe, const Action& action)
{
    if (!is_valid()) return;

    for (size_t i = 0; i < (size_t)Limb::Count; ++i)
        if (action.torques[i] != 0.f)
            pe.apply_torque(limbs[i], action.torques[i]);
}

void PlayerCharacter::capture_state(PhysicsEngine& pe)
{
    if (!is_valid()) return;

    uint32_t torso_id = limbs[(size_t)Limb::Torso];

    MovementState s;
    s.move_dir               = move_dir;
    s.walking                = walking;
    s.jump                   = jump;
    s.aim_pos                = aim_pos;
    s.torso_velocity         = pe.get_velocity(torso_id);
    s.torso_angle            = pe.get_rotation(torso_id);
    s.torso_angular_velocity = pe.get_angular_velocity(torso_id);

    for (size_t i = 0; i < (size_t)Joint::Count; ++i) {
        float pa  = pe.get_rotation(limbs[(size_t)JOINT_LIMBS[i].parent]);
        float ca  = pe.get_rotation(limbs[(size_t)JOINT_LIMBS[i].child]);
        float rel = ca - pa;
        s.joint_angles[i] = std::atan2(std::sin(rel), std::cos(rel));
    }

    s.grounded = grounded;

    glm::vec2 torso_pos = pe.get_position(torso_id);
    float     torso_angle = pe.get_rotation(torso_id);
    s.torso_pos = torso_pos;

    auto wall_hit   = pe.raycast(torso_pos, glm::vec2(facing_dir, 0.f), wall_check_dist, PLAYER_MASK);
    s.wall_ahead    = wall_hit.has_value();

    auto ground_hit = pe.raycast(torso_pos, glm::vec2(0.f, 1.f), 300.f, PLAYER_MASK);
    s.ground_dist   = ground_hit.has_value() ? ground_hit.value().distance : 9999.f;

    s.left_shoulder_pos = torso_pos + glm::rotate(glm::vec2(-SHOULDER_X, SHOULDER_Y), torso_angle);
    s.left_arm_angle    = pe.get_rotation(limbs[(size_t)Limb::UpperArmL]);

    state_history.push(s);
}

float PlayerCharacter::compute_reward(float dt) const
{
    if (state_history.count == 0) return 0.f;
    const MovementState& cur = state_history.at(0);

    float reward = reward_alive;

    // forward velocity in the intended direction
    reward += reward_velocity * cur.torso_velocity.x * cur.move_dir;

    // penalise tipping and spinning
    reward -= reward_upright     * std::abs(cur.torso_angle);
    reward -= reward_angular_vel * std::abs(cur.torso_angular_velocity);

    // bonus for staying grounded while moving
    if (cur.grounded && cur.move_dir != 0.f)
        reward += reward_grounded_move;

    // reward upward velocity arriving jump_frame_delay frames after a jump input
    if (state_history.count > jump_frame_delay) {
        const MovementState& past = state_history.at(jump_frame_delay);
        if (past.jump)
            reward += reward_jump_vel * (-cur.torso_velocity.y); // y-down, so negate for upward
    }

    // penalise deviation from the target standing height
    if (cur.ground_dist < 9999.f)
        reward -= reward_standing_height * std::abs(cur.ground_dist - standing_height);

    // reward left arm pointing toward aim_pos (cosine similarity, peak = 1 when aligned)
    {
        glm::vec2 to_aim = cur.aim_pos - cur.left_shoulder_pos;
        if (glm::length(to_aim) > 0.001f) {
            // game angle convention: atan2(-x, y) gives 0 pointing down, matching body rotations
            float desired = std::atan2(-to_aim.x, to_aim.y);
            float err     = cur.left_arm_angle - desired;
            err = std::atan2(std::sin(err), std::cos(err)); // normalise to [-pi, pi]
            reward += reward_arm_aim * std::cos(err);
        }
    }

    return reward;
}

void PlayerCharacter::animate(PhysicsEngine& pe, float dt)
{
    if (!is_valid()) return;

    glm::vec2 air_normal = glm::normalize(glm::vec2(facing_dir * AIR_NORMAL_TILT, -1.0f));

    if (grounded && !was_grounded) {
        auto snap = [&](bool right) {
            auto hit = get_standing_step(pe, right);
            if (!hit.has_value()) return;
            glm::vec2 pt = hit.value().point;
            glm::vec2 n  = hit.value().normal;
            if (right) {
                right_step_target = pt; right_step_goal = pt;
                right_step_normal = n;  right_step_normal_target = n;
                right_airborne = false;
            } else {
                left_step_target = pt; left_step_goal = pt;
                left_step_normal = n;  left_step_normal_target = n;
                left_airborne = false;
            }
        };
        snap(false);
        snap(true);
    }
    was_grounded = grounded;

    if (grounded)
        update_grounded(pe, dt, air_normal);
    else
        update_airborne(pe, air_normal);

    interpolate_steps(dt);

    glm::vec2 torso_pos   = pe.get_position(limbs[(size_t)Limb::Torso]);
    float     torso_angle = pe.get_rotation(limbs[(size_t)Limb::Torso]);

    facing_dir = (aim_pos.x > torso_pos.x) ? 1.f : -1.f;
    for (uint32_t id : limbs)
        if (id != INVALID) pe.world.bodies[id].flip_h = !(facing_dir > 0.f);

    set_rest_angle(Joint::Neck, torso_angle, torso_angle);

    animate_legs(pe, dt, torso_pos, torso_angle);
    animate_arms(pe, torso_pos, torso_angle);

    DebugDraw::get().point(left_step_goal,  4.f, 0x00FF00FF);
    DebugDraw::get().point(right_step_goal, 4.f, 0xFF0000FF);
}

void PlayerCharacter::update_grounded(PhysicsEngine& pe, float dt, glm::vec2 air_normal)
{
    float hvel        = velocity(pe).x;
    float step_offset = move_dir < 0.0f ? RUN_STEP_LEFT_OFF : 0.0f;

    if (!walking && move_dir != 0.f) {
        animation_leg_state = AnimationLegState::Running;
        if (stride_counter < step_time) {
            if (right_airborne) {
                auto hit = select_next_step(pe, true, step_offset);
                if (hit.has_value()) { right_step_target = hit.value().point; right_step_normal_target = hit.value().normal; right_airborne = false; }
            }
            left_step_target = position(pe) + airborn_foot_offset * glm::vec2(move_dir, 1.f)
                               + glm::vec2(RUN_STEP_BASE_X * move_dir, 0.0f)
                               + glm::vec2(RUN_ARC_X * move_dir, RUN_ARC_Y) * stride_counter / step_time;
            left_step_normal_target = air_normal;
            left_airborne = true;
        } else if (stride_counter < step_time * 2.f) {
            if (left_airborne) {
                auto hit = select_next_step(pe, false, step_offset);
                if (hit.has_value()) { left_step_target = hit.value().point; left_step_normal = hit.value().normal; left_airborne = false; }
            }
            right_step_target = position(pe) + airborn_foot_offset * glm::vec2(move_dir, 1.f)
                                + glm::vec2(RUN_ARC_X * move_dir, RUN_ARC_Y) * stride_counter / step_time;
            right_step_normal_target = air_normal;
            right_airborne = true;
        }
        stride_counter += dt * (glm::abs(hvel) / max_speed + 0.5f);
        if (stride_counter > step_time * 2.f) stride_counter = 0.f;
    } else if (walking && move_dir != 0.f) {
        animation_leg_state = AnimationLegState::Walking;
        if (stride_counter < step_time) {
            if (right_airborne) {
                auto hit = select_next_step(pe, true);
                if (hit.has_value()) { right_step_target = hit.value().point; right_step_normal_target = hit.value().normal; right_airborne = false; }
            }
            left_airborne = true;
        } else if (stride_counter < step_time * 2.f) {
            if (left_airborne) {
                auto hit = select_next_step(pe, false);
                if (hit.has_value()) { left_step_target = hit.value().point; left_step_normal = hit.value().normal; left_airborne = false; }
            }
            right_airborne = true;
        }
        stride_counter += dt * WALK_STRIDE_RATE;
        if (stride_counter > step_time * 2.f) stride_counter = 0.f;
    }
    if (move_dir == 0.f) {
        animation_leg_state = AnimationLegState::Stationary;
        auto hit = get_standing_step(pe, false);
        if (hit.has_value()) { left_step_target  = hit.value().point; left_step_normal_target  = hit.value().normal; left_airborne  = false; }
        hit = get_standing_step(pe, true);
        if (hit.has_value()) { right_step_target = hit.value().point; right_step_normal_target = hit.value().normal; right_airborne = false; }
        stride_counter = 0.f;
    }
}

void PlayerCharacter::update_airborne(PhysicsEngine& pe, glm::vec2 air_normal)
{
    float vy     = velocity(pe).y;
    bool  falling = vy > 0.f;

    auto  hit              = pe.raycast(position(pe), glm::vec2(0.f, 1.f), AIRBORNE_RAY_DIST, PLAYER_MASK);
    float dist_from_ground = hit.has_value() ? hit.value().distance : 999.f;

    bool  right_high = facing_dir > 0.f;
    float left_y_lo  = right_high ? jump_foot_y_low : jump_foot_y;
    float right_y_lo = right_high ? jump_foot_y     : jump_foot_y_low;

    if (jump_timer > 0.0f) {
        left_step_target  = position(pe) + glm::vec2(-jump_foot_x, left_y_lo);
        right_step_target = position(pe) + glm::vec2( jump_foot_x, right_y_lo);
        left_step_normal_target  = air_normal;
        right_step_normal_target = air_normal;
    } else if (!falling || dist_from_ground > jump_airborne_dist) {
        left_step_target  = position(pe) + glm::vec2(-jump_foot_x, jump_foot_y);
        right_step_target = position(pe) + glm::vec2( jump_foot_x, jump_foot_y);
        left_step_normal_target  = air_normal;
        right_step_normal_target = air_normal;
    } else if (hit.has_value()) {
        if (move_dir < 0.0f) {
            left_step_target         = hit.value().point + glm::vec2(-jump_foot_x, right_high ? 0.f : jump_foot_y);
            right_step_target        = position(pe)      + glm::vec2( jump_foot_x, jump_foot_y);
            left_step_normal_target  = hit.value().normal;
            right_step_normal_target = air_normal;
        } else {
            right_step_target        = hit.value().point + glm::vec2( jump_foot_x, right_high ? 0.f : jump_foot_y);
            right_step_normal_target = hit.value().normal;
            left_step_target         = position(pe)      + glm::vec2(-jump_foot_x, jump_foot_y);
            left_step_normal_target  = air_normal;
        }
    }
    left_airborne  = false;
    right_airborne = false;
    stride_counter = 0.f;
}

void PlayerCharacter::interpolate_steps(float dt)
{
    float interp      = animation_leg_state == AnimationLegState::Running ? INTERP_RUNNING : INTERP_OTHER;
    float goal_interp = grounded ? interp : interp * INTERP_AIR_MULT;
    float t = 1.f - std::exp(-goal_interp * dt);
    left_step_goal  = glm::mix(left_step_goal,  left_step_target, t);
    right_step_goal = glm::mix(right_step_goal, right_step_target, t);
    left_step_normal  = glm::normalize(glm::mix(left_step_normal,  left_step_normal_target,  t));
    right_step_normal = glm::normalize(glm::mix(right_step_normal, right_step_normal_target, t));
}

std::pair<float,float> PlayerCharacter::solve_arm_ik(glm::vec2 shoulder, glm::vec2 target) const
{
    glm::vec2 diff = target - shoulder;
    float dist = std::clamp(glm::length(diff),
                            std::abs(UPPER_ARM_LEN - FOREARM_LEN) + 0.01f,
                            UPPER_ARM_LEN + FOREARM_LEN - 0.01f);
    float cos_a = (UPPER_ARM_LEN*UPPER_ARM_LEN + dist*dist - FOREARM_LEN*FOREARM_LEN)
                  / (2.f * UPPER_ARM_LEN * dist);
    float alpha = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float phi   = std::atan2(-diff.x, diff.y);
    for (int s : {1, -1}) {
        float ua_a  = phi + s * alpha;
        glm::vec2 elbow = shoulder + glm::vec2(-UPPER_ARM_LEN * std::sin(ua_a),
                                                UPPER_ARM_LEN * std::cos(ua_a));
        if (elbow.y >= shoulder.y) {
            glm::vec2 ld = target - elbow;
            return { ua_a, std::atan2(-ld.x, ld.y) };
        }
    }
    float ua_a  = phi - alpha;
    glm::vec2 elbow = shoulder + glm::vec2(-UPPER_ARM_LEN * std::sin(ua_a),
                                            UPPER_ARM_LEN * std::cos(ua_a));
    glm::vec2 ld    = target - elbow;
    return { ua_a, std::atan2(-ld.x, ld.y) };
}

void PlayerCharacter::animate_leg_fk(Joint hip, Joint knee, Joint ankle,
                                      float parent_angle,
                                      float thigh_angle, float lower_angle, float foot_angle)
{
    set_rest_angle(hip,   parent_angle, thigh_angle);
    set_rest_angle(knee,  thigh_angle,  lower_angle);
    set_rest_angle(ankle, lower_angle,  foot_angle);
}

void PlayerCharacter::animate_legs(PhysicsEngine& pe, float dt, glm::vec2 torso_pos, float torso_angle)
{
    bool facing_right = facing_dir > 0.f;

    glm::vec2 hip_l = torso_pos + glm::rotate(glm::vec2(-HIP_X, HIP_Y), torso_angle);
    glm::vec2 hip_r = torso_pos + glm::rotate(glm::vec2( HIP_X, HIP_Y), torso_angle);

    glm::vec2 ankle_tgt_l = (facing_right ? left_step_goal  : right_step_goal) + glm::vec2(0.f, ANKLE_RAISE);
    glm::vec2 ankle_tgt_r = (facing_right ? right_step_goal : left_step_goal)  + glm::vec2(0.f, ANKLE_RAISE);

    auto [tl, ll] = solve_leg_ik(hip_l, ankle_tgt_l, facing_dir);
    auto [tr, lr] = solve_leg_ik(hip_r, ankle_tgt_r, facing_dir);

    float max_d = max_leg_angle_speed * dt;
    auto approach = [&](float cur, float tgt) {
        float d = std::atan2(std::sin(tgt - cur), std::cos(tgt - cur));
        return cur + std::clamp(d, -max_d, max_d);
    };
    thigh_angle_l = approach(thigh_angle_l, tl);
    lower_angle_l = approach(lower_angle_l, ll);
    thigh_angle_r = approach(thigh_angle_r, tr);
    lower_angle_r = approach(lower_angle_r, lr);

    auto resolve_foot_angle = [this](glm::vec2 normal) {
        constexpr float PI = 3.14159265f;
        float a = std::atan2(normal.y, normal.x) + PI * 0.5f;
        if (a >  PI) a -= 2.f * PI;
        if (-std::sin(a) * facing_dir < 0.f) { a += PI; if (a > PI) a -= 2.f * PI; }
        return a;
    };
    float foot_angle_l = resolve_foot_angle(facing_right ? left_step_normal  : right_step_normal);
    float foot_angle_r = resolve_foot_angle(facing_right ? right_step_normal : left_step_normal);

    animate_leg_fk(Joint::HipL, Joint::KneeL, Joint::AnkleL, torso_angle, thigh_angle_l, lower_angle_l, foot_angle_l);
    animate_leg_fk(Joint::HipR, Joint::KneeR, Joint::AnkleR, torso_angle, thigh_angle_r, lower_angle_r, foot_angle_r);
}

void PlayerCharacter::animate_arms(PhysicsEngine& pe, glm::vec2 torso_pos, float torso_angle)
{
    glm::vec2 shoulder_r = torso_pos + glm::rotate(glm::vec2( SHOULDER_X, SHOULDER_Y), torso_angle);
    glm::vec2 shoulder_l = torso_pos + glm::rotate(glm::vec2(-SHOULDER_X, SHOULDER_Y), torso_angle);

    auto [ua_angle_r, fa_angle_r] = solve_arm_ik(shoulder_r, aim_pos);
    set_rest_angle(Joint::ShoulderR, torso_angle, ua_angle_r);
    set_rest_angle(Joint::ElbowR,    ua_angle_r,  fa_angle_r);
    set_rest_angle(Joint::WristR,    fa_angle_r,  fa_angle_r);

    glm::vec2 elbow_r = shoulder_r + glm::rotate(glm::vec2(0.f, UPPER_ARM_LEN), ua_angle_r);
    glm::vec2 fa_r    = elbow_r    + glm::rotate(glm::vec2(0.f, FOREARM_MID),   fa_angle_r);
    glm::vec2 grab_r  = glm::mix(elbow_r, fa_r, GRAB_BLEND);

    auto [ua_angle_l, fa_angle_l] = solve_arm_ik(shoulder_l, grab_r);
    set_rest_angle(Joint::ShoulderL, torso_angle, ua_angle_l);
    set_rest_angle(Joint::ElbowL,    ua_angle_l,  fa_angle_l);
    set_rest_angle(Joint::WristL,    fa_angle_l,  fa_angle_r);
}

std::optional<RaycastHit> PlayerCharacter::select_next_step(PhysicsEngine& pe, bool right, float x_offset)
{
    float step_offset = (right ? STEP_RIGHT_EXTRA : 0.0f) + x_offset;
    glm::vec2 origin = position(pe) + glm::vec2(
        move_dir * (step_offset + (glm::abs(velocity(pe).x) / max_speed + STEP_VEL_MIN)
        * step_length_test * (animation_leg_state == AnimationLegState::Walking ? STEP_WALK_MULT : 1.0f)),
        0.0f);
    auto hit = pe.raycast(origin, glm::vec2(0.f, 1.f), STEP_RAY_DIST, PLAYER_MASK);
    if (hit.has_value() && hit.value().point.y > position(pe).y + STEP_HEIGHT_THRESH)
        hit.value().point += glm::vec2(0.0f, STEP_HEIGHT_NUDGE);
    return hit;
}

std::optional<RaycastHit> PlayerCharacter::get_standing_step(PhysicsEngine& pe, bool right)
{
    float step_offset = right ? 0.0f : -STANDING_STEP_X * facing_dir;
    glm::vec2 origin  = position(pe) + glm::vec2(step_offset, STANDING_ORIGIN_Y);
    auto hit = pe.raycast(origin, glm::vec2(0.f, 1.f), STEP_RAY_DIST, PLAYER_MASK);
    if (hit.has_value() && hit.value().point.y > position(pe).y + STEP_HEIGHT_THRESH)
        hit.value().point += glm::vec2(0.0f, STEP_HEIGHT_NUDGE);
    return hit;
}

glm::vec2 PlayerCharacter::position(PhysicsEngine& pe) const { return pe.get_position(limbs[(size_t)Limb::Torso]); }
glm::vec2 PlayerCharacter::velocity(PhysicsEngine& pe) const { return pe.get_velocity(limbs[(size_t)Limb::Torso]); }
float     PlayerCharacter::rotation(PhysicsEngine& pe) const { return pe.get_rotation(limbs[(size_t)Limb::Torso]); }
