#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <src/game/debug_draw.h>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>

// ── Hitbox ────────────────────────────────────────────────────────────────────
static constexpr float HITBOX_ANG_DAMPING = 0.99f;
static constexpr float TORSO_ABOVE_HITBOX = 17.f;

// ── Limb initial spawn positions ──────────────────────────────────────────────
static constexpr float SPAWN_SIDE_X      =  8.f;
static constexpr float SPAWN_HEAD_Y      = -24.f;
static constexpr float SPAWN_FOREARM_Y   =  14.f;
static constexpr float SPAWN_HAND_Y      =  26.f;
static constexpr float SPAWN_THIGH_Y     =  24.f;
static constexpr float SPAWN_LOWER_LEG_Y =  40.f;
static constexpr float SPAWN_FOOT_Y      =  56.f;

// ── Leg skeleton ──────────────────────────────────────────────────────────────
static constexpr float THIGH_LEN      = 15.f;  // hip → knee
static constexpr float THIGH_MID      =  8.f;  // thigh sprite centre from hip
static constexpr float LOWER_LEN      = 20.f;  // knee → ankle (IK)
static constexpr float LOWER_MID      =  8.f;  // lower-leg sprite centre from knee
static constexpr float LOWER_TO_ANKLE = 10.f;  // knee → ankle (FK)
static constexpr float HIP_X          =  3.f;  // hip lateral offset from torso centre
static constexpr float HIP_Y          = 14.f;  // hip vertical offset from torso centre
static constexpr float ANKLE_RAISE    =  6.f;  // ankle target raised above step goal
static constexpr float FOOT_OFF_X     =  3.f;  // foot centre x from ankle (facing-scaled)
static constexpr float FOOT_OFF_Y     =  4.f;  // foot centre y from ankle

// ── Head / neck ───────────────────────────────────────────────────────────────
static constexpr float NECK_Y = -10.f;
static constexpr float HEAD_Y =  -8.f;

// ── Arm skeleton ──────────────────────────────────────────────────────────────
static constexpr float SHOULDER_X     =  4.f;   // shoulder x from torso centre (facing-scaled)
static constexpr float SHOULDER_Y     = -11.f;  // shoulder y from torso centre
static constexpr float UPPER_ARM_LEN  = 15.f;   // shoulder → elbow
static constexpr float UPPER_ARM_MID  =  8.f;   // upper-arm sprite centre from shoulder
static constexpr float FOREARM_LEN    = 17.f;   // elbow → hand tip (IK)
static constexpr float FOREARM_MID    =  8.f;   // forearm sprite centre from elbow
static constexpr float ELBOW_TO_WRIST =  9.f;
static constexpr float WRIST_TO_HAND  =  8.f;
static constexpr float GRAB_BLEND     =  0.5f;  // blend between elbow and forearm for left-hand grab

// ── Animation ─────────────────────────────────────────────────────────────────
static constexpr float AIR_NORMAL_TILT   = 2.0f;
static constexpr float INTERP_RUNNING    = 12.f;
static constexpr float INTERP_OTHER      =  7.f;
static constexpr float INTERP_AIR_MULT   =  3.f;
static constexpr float AIRBORNE_RAY_DIST = 500.f;
static constexpr float RUN_STEP_BASE_X   =  5.f;   // extra forward offset while running
static constexpr float RUN_STEP_LEFT_OFF = 10.f;   // step offset when move_dir < 0
static constexpr float RUN_ARC_X         = 16.f;   // step arc horizontal extent
static constexpr float RUN_ARC_Y         =  5.f;   // step arc vertical drop
static constexpr float WALK_STRIDE_RATE  =  1.5f;

// ── Step detection ────────────────────────────────────────────────────────────
static constexpr float STEP_RAY_DIST      = 80.f;
static constexpr float STEP_HEIGHT_THRESH = 20.f;
static constexpr float STEP_HEIGHT_NUDGE  = -3.f;
static constexpr float STEP_VEL_MIN       =  0.1f;
static constexpr float STEP_WALK_MULT     =  1.25f;
static constexpr float STEP_RIGHT_EXTRA   =  5.f;
static constexpr float STANDING_STEP_X    = 18.f;
static constexpr float STANDING_ORIGIN_Y  =  3.f;
static constexpr float SPRING_MIN_DIST    =  0.001f;

// ─────────────────────────────────────────────────────────────────────────────

void PlayerCharacter::load_assets()
{
    img_hitbox    = load_body_image("../assets/player/hitbox.png");
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
    rb.collision_mask  = 0u;
    uint32_t id = pe.add_body(&re, std::move(rb));
    pe.world.bodies[id].inv_mass    = 0.f;
    pe.world.bodies[id].inv_inertia = 0.f;
    return id;
}

void PlayerCharacter::place(PhysicsEngine& pe, Limb limb, glm::vec2 center, float angle)
{
    uint32_t id = limbs[(size_t)limb];
    if (id == INVALID) return;
    pe.set_position(id, center);
    pe.set_rotation(id, angle);
}

void PlayerCharacter::spawn(PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 pos)
{
    limbs.fill(INVALID);

    // physics body
    {
        RigidBody rb;
        rb.sprite          = img_hitbox;
        rb.compute_mass_properties();
        rb.generate_shape();
        rb.generate_sdf();
        rb.position        = glm::vec3(pos, 0.f);
        rb.collision_layer = PLAYER_LAYER;
        rb.collision_mask  = PLAYER_MASK;
        rb.angular_damping = HITBOX_ANG_DAMPING;
        hitbox_id = pe.add_body(&re, std::move(rb));
        pe.world.bodies[hitbox_id].visible = false;
    }

    limbs[(size_t)Limb::Torso]      = spawn_limb(pe, re, img_torso,     pos);
    limbs[(size_t)Limb::Head]       = spawn_limb(pe, re, img_head,      pos + glm::vec2(              0, SPAWN_HEAD_Y));
    limbs[(size_t)Limb::UpperArmL]  = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2(-SPAWN_SIDE_X,             0));
    limbs[(size_t)Limb::ForearmL]   = spawn_limb(pe, re, img_forearm,   pos + glm::vec2(-SPAWN_SIDE_X,  SPAWN_FOREARM_Y));
    limbs[(size_t)Limb::HandL]      = spawn_limb(pe, re, img_hand,      pos + glm::vec2(-SPAWN_SIDE_X,  SPAWN_HAND_Y));
    limbs[(size_t)Limb::UpperArmR]  = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2( SPAWN_SIDE_X,             0));
    limbs[(size_t)Limb::ForearmR]   = spawn_limb(pe, re, img_forearm,   pos + glm::vec2( SPAWN_SIDE_X,  SPAWN_FOREARM_Y));
    limbs[(size_t)Limb::HandR]      = spawn_limb(pe, re, img_hand,      pos + glm::vec2( SPAWN_SIDE_X,  SPAWN_HAND_Y));
    limbs[(size_t)Limb::ThighL]     = spawn_limb(pe, re, img_thigh,     pos + glm::vec2(-SPAWN_SIDE_X,  SPAWN_THIGH_Y));
    limbs[(size_t)Limb::LowerLegL]  = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2(-SPAWN_SIDE_X,  SPAWN_LOWER_LEG_Y));
    limbs[(size_t)Limb::FootL]      = spawn_limb(pe, re, img_foot,      pos + glm::vec2(-SPAWN_SIDE_X,  SPAWN_FOOT_Y));
    limbs[(size_t)Limb::ThighR]     = spawn_limb(pe, re, img_thigh,     pos + glm::vec2( SPAWN_SIDE_X,  SPAWN_THIGH_Y));
    limbs[(size_t)Limb::LowerLegR]  = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2( SPAWN_SIDE_X,  SPAWN_LOWER_LEG_Y));
    limbs[(size_t)Limb::FootR]      = spawn_limb(pe, re, img_foot,      pos + glm::vec2( SPAWN_SIDE_X,  SPAWN_FOOT_Y));

    // render layers
    for (Limb l : { Limb::ThighR, Limb::LowerLegR, Limb::FootR,
                    Limb::UpperArmR, Limb::ForearmR, Limb::HandR })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 0;
    for (Limb l : { Limb::Torso, Limb::Head })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 1;
    for (Limb l : { Limb::ThighL, Limb::LowerLegL, Limb::FootL,
                    Limb::UpperArmL, Limb::ForearmL, Limb::HandL })
        pe.world.bodies[limbs[(size_t)l]].draw_layer = 2;
}

void PlayerCharacter::despawn(PhysicsEngine& pe)
{
    if (!is_valid()) return;

    if (hitbox_id != INVALID) {
        pe.remove_body(hitbox_id);
        hitbox_id = INVALID;
    }
    for (uint32_t id : limbs)
        if (id != INVALID) pe.remove_body(id);

    limbs.fill(INVALID);
}

static std::pair<float,float> solve_leg_ik(glm::vec2 hip, glm::vec2 ankle_target,
                                            float knee_dir = 1.f)
{
    glm::vec2 diff = ankle_target - hip;
    float dist = std::clamp(glm::length(diff), std::abs(THIGH_LEN - LOWER_LEN) + 0.01f, THIGH_LEN + LOWER_LEN - 0.01f);
    float cos_a     = (THIGH_LEN*THIGH_LEN + dist*dist - LOWER_LEN*LOWER_LEN) / (2.f * THIGH_LEN * dist);
    float alpha     = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float chain_phi = std::atan2(-diff.x, diff.y);
    float thigh_angle = chain_phi - knee_dir * alpha;

    glm::vec2 knee = hip + glm::vec2(-THIGH_LEN * std::sin(thigh_angle),
                                      THIGH_LEN * std::cos(thigh_angle));
    glm::vec2 ld = ankle_target - knee;
    float lower_angle = std::atan2(-ld.x, ld.y);
    return { thigh_angle, lower_angle };
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

    glm::vec2 hbox_pos   = pe.get_position(hitbox_id);
    glm::vec2 hbox_vel   = pe.get_velocity(hitbox_id);
    float     hbox_angle = pe.get_rotation(hitbox_id);

    {
        auto hit = pe.raycast(hbox_pos, glm::vec2(0.f, 1.f), ground_check_dist, PLAYER_MASK);
        grounded = hit.has_value();
    }

    if (jump_timer > 0.f) jump_timer -= dt;
    if (jump && grounded) {
        pe.set_velocity(hitbox_id, glm::vec2(hbox_vel.x, -jump_impulse));
        jump_timer = jump_spring_disable_time;
        grounded   = false;
    }

    // keep character somewhat locked to the ground
    if (grounded && jump_timer <= 0.f) {
        auto apply_ground_spring = [&](glm::vec2 contact, bool airborne) {
            if (airborne) return;
            glm::vec2 delta = contact - hbox_pos;
            float dist = glm::length(delta);
            if (dist > SPRING_MIN_DIST && dist < ground_spring_max_dist) {
                glm::vec2 dir = delta / dist;
                float rel_vel = glm::dot(hbox_vel, dir);
                pe.apply_force(hitbox_id, (ground_spring_k * (dist - ground_spring_rest_length) - ground_spring_damping * rel_vel) * dir);
            }
        };
        apply_ground_spring(left_step_target,  left_airborne);
        apply_ground_spring(right_step_target, right_airborne);
    }

    float target_vx     = move_dir * (walking ? walk_speed : max_speed);
    float current_accel = grounded ? accel : air_accel;
    pe.apply_force(hitbox_id, { (target_vx - hbox_vel.x) * current_accel, 0.f });

    // righting force and leaning logic
    bool  moving_backward = (move_dir * facing_dir < 0.f);
    float lean_target     = move_dir * (moving_backward ? backward_lean : forward_lean);
    float angle_error  = std::atan2(std::sin(hbox_angle - lean_target),
                                    std::cos(hbox_angle - lean_target));
    float omega        = pe.get_angular_velocity(hitbox_id);
    float target_omega = -upright_stiffness * angle_error;
    pe.set_angular_velocity(hitbox_id, omega + (target_omega - omega) * upright_damping * dt);
}

void PlayerCharacter::update(PhysicsEngine& pe, float dt)
{
    handle_controls(pe, dt);
    animate(pe, dt);
}

void PlayerCharacter::animate(PhysicsEngine& pe, float dt)
{
    if (!is_valid()) return;

    glm::vec2 air_normal = glm::normalize(glm::vec2(facing_dir * AIR_NORMAL_TILT, -1.0f));

    // fix feet on landing
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

    float bounce = 0.f;
    if (grounded && animation_leg_state == AnimationLegState::Running && step_time > 0.f) {
        float phase = std::fmod(stride_counter, step_time);
        bounce = (1.f - std::abs(phase / step_time - 0.5f) * 2.f) * stride_height;
    }
    float hbox_angle    = pe.get_rotation(hitbox_id);
    glm::vec2 torso_pos = pe.get_position(hitbox_id) + glm::vec2(0.f, -TORSO_ABOVE_HITBOX - bounce);
    place(pe, Limb::Torso, torso_pos, hbox_angle);

    facing_dir = (aim_pos.x > position(pe).x) ? 1.f : -1.f;
    for (uint32_t id : limbs)
        if (id != INVALID) pe.world.bodies[id].flip_h = !(facing_dir > 0.f);

    animate_legs(pe, dt, torso_pos, hbox_angle);

    glm::vec2 neck     = torso_pos + glm::rotate(glm::vec2(0.f, NECK_Y), hbox_angle);
    glm::vec2 head_pos = neck      + glm::rotate(glm::vec2(0.f, HEAD_Y), hbox_angle);
    place(pe, Limb::Head, head_pos, hbox_angle);

    animate_arms(pe, torso_pos, hbox_angle);

    DebugDraw::get().point(left_step_goal,  4.f, 0x00FF00FF);
    DebugDraw::get().point(right_step_goal, 4.f, 0xFF0000FF);
}

void PlayerCharacter::update_grounded(PhysicsEngine& pe, float dt, glm::vec2 air_normal)
{
    float hvel        = pe.get_velocity(hitbox_id).x;
    float step_offset = move_dir < 0.0f ? RUN_STEP_LEFT_OFF : 0.0f;

    if (!walking && move_dir != 0.f) {
        // running logic
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
        // walking logic
        // TODO: fix walking - maybe just use running state
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
        // standing
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
    float vy   = pe.get_velocity(hitbox_id).y;
    bool falling = vy > 0.f;

    auto hit = pe.raycast(position(pe), glm::vec2(0.f, 1.f), AIRBORNE_RAY_DIST, PLAYER_MASK);
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
    float cos_a = (UPPER_ARM_LEN*UPPER_ARM_LEN + dist*dist - FOREARM_LEN*FOREARM_LEN) / (2.f * UPPER_ARM_LEN * dist);
    float alpha = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float phi   = std::atan2(-diff.x, diff.y);
    for (int s : {1, -1}) {
        float ua_a = phi + s * alpha;
        glm::vec2 elbow = shoulder + glm::vec2(-UPPER_ARM_LEN * std::sin(ua_a), UPPER_ARM_LEN * std::cos(ua_a));
        if (elbow.y >= shoulder.y) {
            glm::vec2 ld = target - elbow;
            return { ua_a, std::atan2(-ld.x, ld.y) };
        }
    }
    float ua_a  = phi - alpha;
    glm::vec2 elbow = shoulder + glm::vec2(-UPPER_ARM_LEN * std::sin(ua_a), UPPER_ARM_LEN * std::cos(ua_a));
    glm::vec2 ld    = target - elbow;
    return { ua_a, std::atan2(-ld.x, ld.y) };
}

void PlayerCharacter::animate_leg_fk(PhysicsEngine& pe,
                                      Limb thigh_limb, Limb lower_leg_limb, Limb foot_limb,
                                      glm::vec2 hip, float thigh_angle, float lower_angle, float foot_angle)
{
    glm::vec2 thigh = hip   + glm::rotate(glm::vec2(0.f, THIGH_MID),      thigh_angle);
    glm::vec2 knee  = hip   + glm::rotate(glm::vec2(0.f, THIGH_LEN),      thigh_angle);
    glm::vec2 lower = knee  + glm::rotate(glm::vec2(0.f, LOWER_MID),      lower_angle);
    glm::vec2 ankle = knee  + glm::rotate(glm::vec2(0.f, LOWER_TO_ANKLE), lower_angle);
    glm::vec2 foot  = ankle + glm::rotate(glm::vec2(facing_dir * FOOT_OFF_X, FOOT_OFF_Y), lower_angle);
    place(pe, thigh_limb,    thigh, thigh_angle);
    place(pe, lower_leg_limb, lower, lower_angle);
    place(pe, foot_limb,      foot,  foot_angle);
}

void PlayerCharacter::animate_legs(PhysicsEngine& pe, float dt, glm::vec2 torso_pos, float hbox_angle)
{
    bool facing_right = facing_dir > 0.f;

    glm::vec2 hip_l = torso_pos + glm::rotate(glm::vec2(-HIP_X, HIP_Y), hbox_angle);
    glm::vec2 hip_r = torso_pos + glm::rotate(glm::vec2( HIP_X, HIP_Y), hbox_angle);

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
        if (a > PI) a -= 2.f * PI;
        if (-std::sin(a) * facing_dir < 0.f) { a += PI; if (a > PI) a -= 2.f * PI; }
        return a;
    };
    float foot_angle_l = resolve_foot_angle(facing_right ? left_step_normal  : right_step_normal);
    float foot_angle_r = resolve_foot_angle(facing_right ? right_step_normal : left_step_normal);

    animate_leg_fk(pe, Limb::ThighL, Limb::LowerLegL, Limb::FootL, hip_l, thigh_angle_l, lower_angle_l, foot_angle_l);
    animate_leg_fk(pe, Limb::ThighR, Limb::LowerLegR, Limb::FootR, hip_r, thigh_angle_r, lower_angle_r, foot_angle_r);
}

void PlayerCharacter::animate_arms(PhysicsEngine& pe, glm::vec2 torso_pos, float hbox_angle)
{
    glm::vec2 shoulder_l = torso_pos + glm::rotate(glm::vec2(-facing_dir * SHOULDER_X, SHOULDER_Y), hbox_angle);
    glm::vec2 shoulder_r = torso_pos + glm::rotate(glm::vec2( facing_dir * SHOULDER_X, SHOULDER_Y), hbox_angle);

    auto [ua_angle_r, fa_angle_r] = solve_arm_ik(shoulder_r, aim_pos);

    glm::vec2 ua_r    = shoulder_r + glm::rotate(glm::vec2(0.f, UPPER_ARM_MID),  ua_angle_r);
    glm::vec2 elbow_r = shoulder_r + glm::rotate(glm::vec2(0.f, UPPER_ARM_LEN),  ua_angle_r);
    glm::vec2 fa_r    = elbow_r    + glm::rotate(glm::vec2(0.f, FOREARM_MID),    fa_angle_r);
    glm::vec2 wrist_r = elbow_r    + glm::rotate(glm::vec2(0.f, ELBOW_TO_WRIST), fa_angle_r);
    glm::vec2 hand_r  = wrist_r    + glm::rotate(glm::vec2(0.f, WRIST_TO_HAND),  fa_angle_r);
    place(pe, Limb::UpperArmR, ua_r,   ua_angle_r);
    place(pe, Limb::ForearmR,  fa_r,   fa_angle_r);
    place(pe, Limb::HandR,     hand_r, fa_angle_r);

    glm::vec2 grab_r = glm::mix(elbow_r, fa_r, GRAB_BLEND);
    auto [ua_angle_l, fa_angle_l] = solve_arm_ik(shoulder_l, grab_r);

    glm::vec2 ua_l    = shoulder_l + glm::rotate(glm::vec2(0.f, UPPER_ARM_MID),  ua_angle_l);
    glm::vec2 elbow_l = shoulder_l + glm::rotate(glm::vec2(0.f, UPPER_ARM_LEN),  ua_angle_l);
    glm::vec2 fa_l    = elbow_l    + glm::rotate(glm::vec2(0.f, FOREARM_MID),    fa_angle_l);
    glm::vec2 wrist_l = elbow_l    + glm::rotate(glm::vec2(0.f, ELBOW_TO_WRIST), fa_angle_l);
    glm::vec2 hand_l  = grab_r;
    place(pe, Limb::UpperArmL, ua_l,   ua_angle_l);
    place(pe, Limb::ForearmL,  fa_l,   fa_angle_l);
    place(pe, Limb::HandL,     hand_l, fa_angle_r);
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
    return {};
}

std::optional<RaycastHit> PlayerCharacter::get_standing_step(PhysicsEngine& pe, bool right)
{
    float step_offset = right ? 0.0f : -STANDING_STEP_X * facing_dir;
    glm::vec2 origin  = position(pe) + glm::vec2(step_offset, STANDING_ORIGIN_Y);
    auto hit = pe.raycast(origin, glm::vec2(0.f, 1.f), STEP_RAY_DIST, PLAYER_MASK);
    if (hit.has_value() && hit.value().point.y > position(pe).y + STEP_HEIGHT_THRESH)
        hit.value().point += glm::vec2(0.0f, STEP_HEIGHT_NUDGE);
        return hit;
    return {};
}

glm::vec2 PlayerCharacter::position(PhysicsEngine& pe) const { return pe.get_position(hitbox_id); }
glm::vec2 PlayerCharacter::velocity(PhysicsEngine& pe) const { return pe.get_velocity(hitbox_id); }
float     PlayerCharacter::rotation(PhysicsEngine& pe) const { return pe.get_rotation(hitbox_id); }
