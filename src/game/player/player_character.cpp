#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <src/game/debug_draw.h>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>
#include <fmt/core.h>
#include <iostream>

static const std::unordered_map<std::string, Limb> LIMB_FROM_ID = {
    { "torso",       Limb::Torso      },
    { "head",        Limb::Head       },
    { "upper_arm_l", Limb::UpperArmL  },
    { "forearm_l",   Limb::ForearmL   },
    { "hand_l",      Limb::HandL      },
    { "upper_arm_r", Limb::UpperArmR  },
    { "forearm_r",   Limb::ForearmR   },
    { "hand_r",      Limb::HandR      },
    { "thigh_l",     Limb::ThighL     },
    { "lower_leg_l", Limb::LowerLegL  },
    { "foot_l",      Limb::FootL      },
    { "thigh_r",     Limb::ThighR     },
    { "lower_leg_r", Limb::LowerLegR  },
    { "foot_r",      Limb::FootR      },
};

static const std::unordered_map<std::string, Joint> JOINT_FROM_ID = {
    { "neck",       Joint::Neck      },
    { "shoulder_l", Joint::ShoulderL },
    { "elbow_l",    Joint::ElbowL    },
    { "wrist_l",    Joint::WristL    },
    { "shoulder_r", Joint::ShoulderR },
    { "elbow_r",    Joint::ElbowR    },
    { "wrist_r",    Joint::WristR    },
    { "hip_l",      Joint::HipL      },
    { "knee_l",     Joint::KneeL     },
    { "ankle_l",    Joint::AnkleL    },
    { "hip_r",      Joint::HipR      },
    { "knee_r",     Joint::KneeR     },
    { "ankle_r",    Joint::AnkleR    },
};

// ── Limb skeleton ---------------------------------────────────────────────────
static constexpr float THIGH_LEN        =  8.f;
static constexpr float LOWER_LEN        =  8.f;
static constexpr float HIP_X            =  2.0f;
static constexpr float HIP_Y            =  8.f;
static constexpr float ANKLE_RAISE      =  1.f;

// ── Arm skeleton ──────────────────────────────────────────────────────────────
static constexpr float SHOULDER_X       =  3.f;
static constexpr float SHOULDER_Y       = -6.f;
static constexpr float UPPER_ARM_LEN    =  7.f;
static constexpr float UPPER_ARM_MID    =  3.5f;
static constexpr float FOREARM_LEN      =  7.f;

// ── Step detection ────────────────────────────────────────────────────────────
static constexpr float STEP_RAY_DIST      =  80.f;

static constexpr float MAX_JUMP_CUTOFF = 0.4f;
static constexpr float MIN_JUMP_CUTOFF = 0.2f;
static constexpr float JUMP_DELAY = 0.2f;
static constexpr glm::vec2 JUMP_RATIO = {1.0f, 1.0f};
static constexpr float JUMP_ANIMATION_LENGTH = 0.3f;
static constexpr float LANDING_ANIMATION_LENGTH = 0.15f;
static constexpr float AIRBORNE_DELAY = 0.1f;

static float wrap_angle(float a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

void PlayerCharacter::load_assets(const char* config_path)
{
    if (!char_config.load(config_path)) {
        fmt::print("[PlayerCharacter] Failed to load character config: {}\n", config_path);
        return;
    }
    limb_images.clear();
    for (const auto& [id, sprite] : char_config.limb_sprites)
        limb_images[id] = load_body_image(("../" + sprite).c_str());

    // Build animatable-joint mask and angle limits from character config
    joint_animatable.fill(false);
    joint_limited.fill(false);
    joint_angle_min.fill(-3.14159f);
    joint_angle_max.fill( 3.14159f);
    for (const auto& [jid, jc] : char_config.joints) {
        auto it = JOINT_FROM_ID.find(jid);
        if (it != JOINT_FROM_ID.end()) {
            size_t idx = (size_t)it->second;
            joint_animatable[idx] = jc.animatable;
            joint_angle_min [idx] = jc.angle_min;
            joint_angle_max [idx] = jc.angle_max;
            joint_limited   [idx] = (jc.angle_max - jc.angle_min) < 6.2831f;
        }
    }
}

uint32_t PlayerCharacter::spawn_limb(ResiduaEngine& re,
                                      const LoadedBodyImage& img, glm::vec2 world_pos,
                                      float density)
{
    RigidBody rb;
    rb.sprite = img;
    rb.compute_mass_properties(density > 0.f ? density : 1.f);
    rb.generate_shape();
    rb.generate_sdf();
    rb.position        = glm::vec3(world_pos, 0.f);
    rb.collision_layer = PLAYER_LAYER;
    rb.collision_mask  = PLAYER_MASK;
    return pe->add_body(&re, std::move(rb));
}

void PlayerCharacter::add_joint(Joint jnt, Limb parent, Limb child,
                                 glm::vec2 rA_local, glm::vec2 rB_local,
                                 float /*bend_stiffness*/, float /*max_torque*/)
{
    auto j = std::make_unique<DistanceJoint>(
        &pe->world,
        limbs[(size_t)parent], limbs[(size_t)child],
        rA_local, rB_local,
        std::numeric_limits<float>::infinity(),
        1.f, 1.f, 1.f);
    joint_ptrs[(size_t)jnt] = j.get();
    bool left_arm = (jnt == Joint::ShoulderL || jnt == Joint::ElbowL || jnt == Joint::WristL);
    if (left_arm)
        j->angular_reaction = false;
    pe->world.add_force(std::move(j));
}

void PlayerCharacter::set_rest_angle(Joint jnt, float parent_angle, float child_angle)
{
    joint_goal_angle[(size_t)jnt] = parent_angle - child_angle;
}

void PlayerCharacter::spawn(PhysicsEngine& physics, ResiduaEngine& re, glm::vec2 pos)
{
    pe = &physics;
    limbs.fill(INVALID);
    joint_ptrs.fill(nullptr);
    facing_dir     = 1.f;   
    arm_elbow_side = 1.f;

    auto offsets = char_config.compute_spawn_offsets();

    // Spawn all limbs from JSON definition
    for (const auto& [lid, sprite] : char_config.limb_sprites) {
        auto lit = LIMB_FROM_ID.find(lid);
        if (lit == LIMB_FROM_ID.end()) continue;
        glm::vec2 spawn_pos = pos + offsets.at(lid);

        auto density_it = char_config.limb_densities.find(lid);
        float density = (density_it != char_config.limb_densities.end()) ? density_it->second : 0.f;

        if (lid == "torso") {
            RigidBody rb;
            rb.sprite          = limb_images.at(lid);
            if (density > 0.f)
                rb.compute_mass_properties(density);
            else
                rb.compute_mass_properties();
            rb.generate_shape();
            rb.generate_sdf();
            rb.position        = glm::vec3(spawn_pos, 0.f);
            rb.collision_layer = PLAYER_LAYER;
            rb.collision_mask  = PLAYER_MASK;

            limbs[(size_t)lit->second] = pe->add_body(&re, std::move(rb));
        } else {
            limbs[(size_t)lit->second] = spawn_limb(re, limb_images.at(lid), spawn_pos, density);
        }
    }

    // Cache each foot's dynamic mass/inertia so sync_foot_pin() can zero them out
    // (static, pinned) and restore them (dynamic, swinging) later.
    {
        const RigidBody& fl = pe->world.bodies[limbs[(size_t)Limb::FootL]];
        foot_dyn_mass_l    = fl.mass;
        foot_dyn_inertia_l = fl.inertia;
        foot_friction_l    = fl.friction;
        const RigidBody& fr = pe->world.bodies[limbs[(size_t)Limb::FootR]];
        foot_dyn_mass_r    = fr.mass;
        foot_dyn_inertia_r = fr.inertia;
        foot_friction_r    = fr.friction;
    }

    for (const auto& [jid, jc] : char_config.joints) {
        auto jit  = JOINT_FROM_ID.find(jid);
        auto plit = LIMB_FROM_ID.find(jc.parent_limb);
        auto clit = LIMB_FROM_ID.find(jc.child_limb);
        if (jit == JOINT_FROM_ID.end() || plit == LIMB_FROM_ID.end() || clit == LIMB_FROM_ID.end())
            continue;
        add_joint(jit->second, plit->second, clit->second,
                  jc.attach_parent, jc.attach_child, -1.f, jc.max_torque);
    }

    joint_goal_angle.fill(0.f);
    for (const auto& [jid, jc] : char_config.joints) {
        auto jit = JOINT_FROM_ID.find(jid);
        if (jit != JOINT_FROM_ID.end())
            joint_goal_angle[(size_t)jit->second] = jc.default_angle;
    }

    // upright anchored torso
    {
        auto anchor = std::make_unique<AngleAnchor>(&pe->world, limbs[(size_t)Limb::Torso]);
        anchor->rest_angle = 0.f;
        torso_anchor = anchor.get();
        pe->world.add_force(std::move(anchor));
    }
    left_step_target  = pos + offsets.at("foot_l");
    right_step_target = pos + offsets.at("foot_r");

    // Assign draw layers from JSON draw_order:
    //   items before "torso" → layer 1, "torso"/"head" → layer 2, after "head" → layer 3
    // TODO: integrate with larger draw order system. not exactly sure how this will work
    int layer = 1;
    for (const auto& lid : char_config.draw_order) {
        if (lid == "torso") layer = 2;
        auto lit = LIMB_FROM_ID.find(lid);
        if (lit != LIMB_FROM_ID.end()) {
            uint32_t bid = limbs[(size_t)lit->second];
            if (bid != INVALID)
                pe->world.bodies[bid].draw_layer = layer;
        }
        if (lid == "head") layer = 3;
    }
}

void PlayerCharacter::despawn()
{
    if (!is_valid()) return;

    release_anchor(true);
    release_anchor(false);

    for (auto* j : joint_ptrs)
        if (j) pe->world.remove_force(j);
    joint_ptrs.fill(nullptr);

    if (torso_anchor) pe->world.remove_force(torso_anchor);
    torso_anchor = nullptr;

    for (uint32_t id : limbs)
        if (id != INVALID) pe->remove_body(id);
    limbs.fill(INVALID);
}

// anchoring sets an anchor goal but the actual pinning is managed by sync_foot_pin
void PlayerCharacter::anchor_foot(bool left)
{
    uint32_t foot = foot_id(left);
    if (foot == INVALID || anim_state == AnimationState::Jumping) return;

    (left ? foot_anchored_l : foot_anchored_r) = true;

    sync_foot_pin(left);
}

void PlayerCharacter::release_anchor(bool left)
{
    (left ? foot_anchored_l : foot_anchored_r) = false;
    sync_foot_pin(left);
}

// when a foot should be anchored, this will attempt to place the foot in its anchored position and pin it there
void PlayerCharacter::sync_foot_pin(bool left)
{
    uint32_t foot = foot_id(left);
    if (foot == INVALID) return;

    bool       anchored = left ? foot_anchored_l : foot_anchored_r;
    bool&      pinned    = left ? foot_pinned_l   : foot_pinned_r;
    RigidBody& rb        = pe->world.bodies[foot];

    auto unpin = [&] {
        pinned         = false;
        rb.mass        = left ? foot_dyn_mass_l    : foot_dyn_mass_r;
        rb.inertia     = left ? foot_dyn_inertia_l : foot_dyn_inertia_r;
        rb.inv_mass    = rb.mass    > 0.f ? 1.f / rb.mass    : 0.f;
        rb.inv_inertia = rb.inertia > 0.f ? 1.f / rb.inertia : 0.f;
    };

    // Break under a strong enough impact
    if (pinned) {
        DistanceJoint* ankle = joint_ptrs[(size_t)(left ? Joint::AnkleL : Joint::AnkleR)];
        if (ankle) {
            float force = glm::length(glm::vec2(ankle->lambda[0], ankle->lambda[1]));
            if (force > foot_pin_break_force) {
                unpin();
                (left ? foot_anchored_l : foot_anchored_r) = false;
                return;
            }
        }
    }

    if (!anchored) {
        if (pinned) unpin();
        return;
    }

    if (pinned || !snap_foot_to_ground(left)) return;

    // set to static body
    pinned         = true;
    rb.mass        = 0.f;
    rb.inv_mass    = 0.f;
    rb.inertia     = 0.f;
    rb.inv_inertia = 0.f;
    rb.velocity    = glm::vec3(0.f);
}

void PlayerCharacter::update_facing()
{
    float dx = aim_pos.x - position().x;
    if      (dx < -turn_deadzone) set_facing(-1.f);
    else if (dx >  turn_deadzone) set_facing( 1.f);
}

// Turn around by mirroring the whole character about the torso's vertical axis. Bodies,
// joint anchors, solver impulses and controller state are all mirrored together so every
// constraint stays satisfied and nothing snaps
void PlayerCharacter::set_facing(float dir)
{
    if (!is_valid() || dir == facing_dir) return;
    facing_dir = dir;

    float cx       = position().x;
    auto  mirror_x = [cx](glm::vec2& p) { p.x = 2.f * cx - p.x; };
    auto  mirror_v = [cx](glm::vec3& p) { p.x = 2.f * cx - p.x; p.z = -p.z; };

    float mass = 0.f, vel_x = 0.f, prev_vel_x = 0.f;
    for (uint32_t id : limbs) {
        if (id == INVALID) continue;
        const RigidBody& rb = pe->world.bodies[id];
        mass       += rb.mass;
        vel_x      += rb.mass * rb.velocity.x;
        prev_vel_x += rb.mass * rb.prev_velocity.x;
    }
    if (mass > 0.f) { vel_x /= mass; prev_vel_x /= mass; }

    for (uint32_t id : limbs) {
        if (id == INVALID) continue;
        RigidBody& rb = pe->world.bodies[id];
        mirror_v(rb.position);
        mirror_v(rb.initial);
        mirror_v(rb.inertial);
        if (rb.inv_mass > 0.f) {
            rb.velocity.x      = 2.f * vel_x      - rb.velocity.x;
            rb.prev_velocity.x = 2.f * prev_vel_x - rb.prev_velocity.x;
        }
        rb.velocity.z      = -rb.velocity.z;
        rb.prev_velocity.z = -rb.prev_velocity.z;
        pe->mirror_body_x(id);   // sprite, COM, collision polygon and SDF
        // Swap near/far render layers: the side facing the camera changes
        if      (rb.draw_layer == 1) rb.draw_layer = 3;
        else if (rb.draw_layer == 3) rb.draw_layer = 1;
    }

    for (DistanceJoint* j : joint_ptrs) {
        if (!j) continue;
        j->rA_local.x = -j->rA_local.x;
        j->rB_local.x = -j->rB_local.x;
        j->rest_angle = -j->rest_angle;
        j->lambda[0]  = -j->lambda[0];
        j->lambda[2]  = -j->lambda[2];
    }
    if (torso_anchor) {
        torso_anchor->rest_angle = -torso_anchor->rest_angle;
        torso_anchor->lambda[0]  = -torso_anchor->lambda[0];
    }

    for (float& a : joint_goal_angle) a = -a;
    for (LegPose* p : { &leg_angle_l, &leg_angle_r, &leg_target_l, &leg_target_r,
                        &stance_phase_start, &swing_phase_start }) {
        p->thigh = -p->thigh;
        p->lower = -p->lower;
    }
    foot_angle_l = -foot_angle_l;
    foot_angle_r = -foot_angle_r;
    for (float* a : { &arm_shoulder_rest, &arm_elbow_rest, &arm_wrist_rest,
                      &arm_shoulder_rest_r, &arm_elbow_rest_r, &arm_wrist_rest_r })
        *a = -*a;
    arm_elbow_side = -arm_elbow_side;

    for (glm::vec2* p : { &left_step_target, &right_step_target })
        mirror_x(*p);
    standing_center_x = 2.f * cx - standing_center_x;
}

// gap between the foot and the ground, if the foot is within a given distance
std::optional<float> PlayerCharacter::sole_clearance(bool left, float max_dist) const
{
    uint32_t foot = foot_id(left);
    if (foot == INVALID) return std::nullopt;
    const RigidBody& rb = pe->world.bodies[foot];
    if (rb.shape.empty()) return std::nullopt;

    glm::vec2 pos = glm::vec2(rb.position);
    glm::vec2 sole(0.f, -std::numeric_limits<float>::infinity());
    for (const glm::vec2& v : rb.shape) {
        glm::vec2 w = pos + glm::rotate(v, rb.position.z);
        if (w.y > sole.y) sole = w;
    }

    constexpr float CAST_ABOVE = 3.f;
    glm::vec2 origin = sole - glm::vec2(0.f, CAST_ABOVE);
    auto hit = pe->raycast(origin, glm::vec2(0.f, 1.f), CAST_ABOVE + max_dist, PLAYER_MASK);
    if (!hit.has_value()) return std::nullopt;

    return hit.value().point.y - sole.y;
}

bool PlayerCharacter::snap_foot_to_ground(bool left)
{
    auto clearance = sole_clearance(left, foot_pin_snap_dist);
    if (!clearance.has_value()) return false;

    // rotate the foot to the right angle to be pinned at
    RigidBody& rb    = pe->world.bodies[foot_id(left)];
    glm::vec3  saved = rb.position;
    constexpr float CAST_ABOVE = 3.f;
    glm::vec2 origin = glm::vec2(rb.position) - glm::vec2(0.f, CAST_ABOVE);
    auto ground = pe->raycast(origin, glm::vec2(0.f, 1.f), CAST_ABOVE + STEP_RAY_DIST, PLAYER_MASK);
    if (ground.has_value()) {
        glm::vec2 n     = ground.value().normal;
        float     turn  = wrap_angle(std::atan2(n.x, -n.y) - rb.position.z);
        glm::vec2 ankle = ankle_position(left);
        glm::vec2 rel   = glm::rotate(glm::vec2(rb.position) - ankle, turn);
        rb.position     = glm::vec3(ankle + rel, rb.position.z + turn);
        clearance       = sole_clearance(left, foot_pin_snap_dist);
        if (!clearance.has_value()) { rb.position = saved; return false; }
    }

    rb.position.y += clearance.value();
    return true;
}

bool PlayerCharacter::foot_grounded(bool left) const
{
    return (left ? foot_pinned_l : foot_pinned_r) || sole_clearance(left, foot_ground_dist).has_value();
}


// Move an angle toward target along the shortest arc, by at most max_delta
static void step_angle(float& tracked, float target, float max_delta)
{
    float diff = std::atan2(std::sin(target - tracked), std::cos(target - tracked));
    tracked += std::clamp(diff, -max_delta, max_delta);
}

// step angle from leg angles pointing straight down
static void step_leg_angle(float& tracked, float target, float max_delta)
{
    tracked = wrap_angle(tracked);
    tracked += std::clamp(wrap_angle(target) - tracked, -max_delta, max_delta);
}

static LegPose solve_leg_ik(glm::vec2 hip, glm::vec2 ankle_target, float knee_dir = 1.f)
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

void PlayerCharacter::animate_left_arm()
{
    if (!is_valid()) return;
    glm::vec2 torso_pos   = pe->get_position(limbs[(size_t)Limb::Torso]);
    float     torso_angle = pe->get_rotation(limbs[(size_t)Limb::Torso]);

    glm::vec2 shoulder_l = torso_pos + glm::rotate(glm::vec2(-facing_dir * SHOULDER_X, SHOULDER_Y), torso_angle);

    float     right_upper_arm_angle = pe->get_rotation(limbs[(size_t)Limb::UpperArmR]);
    glm::vec2 upper_arm_r_pos       = pe->get_position(limbs[(size_t)Limb::UpperArmR]);
    glm::vec2 elbow_r = upper_arm_r_pos + glm::rotate(glm::vec2(0.f, UPPER_ARM_MID), right_upper_arm_angle);

    float     ua        = right_upper_arm_angle + facing_dir * support_arm_offset_angle;
    glm::vec2 elbow_l    = shoulder_l + glm::vec2(-UPPER_ARM_LEN * std::sin(ua), UPPER_ARM_LEN * std::cos(ua));
    glm::vec2 ld         = elbow_r - elbow_l;
    float     fa         = std::atan2(-ld.x, ld.y);

    float target_shoulder = torso_angle - ua;
    float target_elbow    = ua - fa;

    float max_delta = arm_angle_speed * pe->world.last_dt;
    auto step = [&](float& tracked, Joint jnt, float target) {
        step_angle(tracked, target, max_delta);
        joint_goal_angle[(size_t)jnt] = tracked;
    };

    step(arm_shoulder_rest, Joint::ShoulderL, target_shoulder);
    step(arm_elbow_rest,    Joint::ElbowL,    target_elbow);
    step(arm_wrist_rest,    Joint::WristL,    0.f);
}

void PlayerCharacter::animate_right_arm()
{
    if (!is_valid()) return;
    glm::vec2 torso_pos   = pe->get_position(limbs[(size_t)Limb::Torso]);
    float     torso_angle = pe->get_rotation(limbs[(size_t)Limb::Torso]);

    glm::vec2 shoulder_r = torso_pos + glm::rotate(glm::vec2(facing_dir * SHOULDER_X, SHOULDER_Y), torso_angle);

    // Project a virtual target far along the shoulder→mouse direction (avoid weird IK arm angles)
    glm::vec2 raw_dir = aim_pos - shoulder_r;
    float     raw_len = glm::length(raw_dir);
    glm::vec2 aim_dir = (raw_len > 0.001f) ? raw_dir / raw_len : glm::vec2(1.f, 0.f);
    glm::vec2 virtual_target = shoulder_r + aim_dir * 500.f;

    auto [ua, fa] = solve_arm_ik(shoulder_r, virtual_target, arm_elbow_side);

    float target_shoulder = torso_angle - ua;
    float target_elbow    = ua - fa;

    float max_delta = arm_angle_speed * pe->world.last_dt;
    auto step = [&](float& tracked, Joint jnt, float target) {
        step_angle(tracked, target, max_delta);
        joint_goal_angle[(size_t)jnt] = tracked;
    };

    step(arm_shoulder_rest_r, Joint::ShoulderR, target_shoulder);
    step(arm_elbow_rest_r,    Joint::ElbowR,    target_elbow);
    step(arm_wrist_rest_r,    Joint::WristR,    0.f);
}

void PlayerCharacter::apply_joint_goals()
{
    if (!is_valid()) return;
    for (size_t i = 0; i < (size_t)Joint::Count; ++i) {
        if (!joint_ptrs[i]) continue;
        // Limits are authored facing right; mirror them when facing left
        float lo = facing_dir > 0.f ? joint_angle_min[i] : -joint_angle_max[i];
        float hi = facing_dir > 0.f ? joint_angle_max[i] : -joint_angle_min[i];
        float goal = wrap_angle(joint_goal_angle[i]);
        if (joint_limited[i]) goal = std::clamp(goal, lo, hi);
        joint_ptrs[i]->rest_angle = goal;
        joint_ptrs[i]->configureBendMotor(joint_bend_hz, joint_bend_damping_ratio, joint_bend_max_accel);
    }

    if (torso_anchor)
        torso_anchor->configureBendMotor(joint_bend_hz, joint_bend_damping_ratio, joint_bend_max_accel);
}

std::pair<float,float> PlayerCharacter::solve_arm_ik(glm::vec2 shoulder, glm::vec2 target, float& elbow_side)
{
    glm::vec2 diff = target - shoulder;
    float dist = std::clamp(glm::length(diff),
                            std::abs(UPPER_ARM_LEN - FOREARM_LEN) + 0.01f,
                            UPPER_ARM_LEN + FOREARM_LEN - 0.01f);
    float cos_a = (UPPER_ARM_LEN*UPPER_ARM_LEN + dist*dist - FOREARM_LEN*FOREARM_LEN)
                  / (2.f * UPPER_ARM_LEN * dist);
    float alpha = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float phi   = std::atan2(-diff.x, diff.y);

    for (int s : {(int)elbow_side, -(int)elbow_side}) {
        float ua_a  = phi + s * alpha;
        glm::vec2 elbow = shoulder + glm::vec2(-UPPER_ARM_LEN * std::sin(ua_a),
                                                UPPER_ARM_LEN * std::cos(ua_a));
        if (elbow.y >= shoulder.y) {
            elbow_side = (float)s;
            glm::vec2 ld = target - elbow;
            return { ua_a, std::atan2(-ld.x, ld.y) };
        }
    }
    float ua_a  = phi - elbow_side * alpha;
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

glm::vec2 PlayerCharacter::hip_position(bool left) const
{
    return position() + glm::rotate(glm::vec2(side_x(left) * HIP_X, HIP_Y), rotation());
}

glm::vec2 PlayerCharacter::ankle_position(bool left) const
{
    const RigidBody&     foot  = pe->world.bodies[foot_id(left)];
    const DistanceJoint* ankle = joint_ptrs[(size_t)(left ? Joint::AnkleL : Joint::AnkleR)];
    glm::vec2 pos = glm::vec2(foot.position);
    return ankle ? pos + glm::rotate(ankle->rB_local, foot.position.z) : pos;
}

// Cast down-rays from x_from toward x_to (inclusive, 1 unit apart) and store the
// first ground hit as this foot's step target
bool PlayerCharacter::scan_step_target(bool left, float x_from, float x_to)
{
    float origin_y = position().y + standing_origin_y;
    float dir      = x_to < x_from ? -1.f : 1.f;
    for (float x = x_from; dir * (x_to - x) >= 0.f; x += dir) {
        glm::vec2 origin = glm::vec2(x, origin_y);
        auto hit = pe->raycast(origin, glm::vec2(0.f, 1.f), STEP_RAY_DIST, PLAYER_MASK);
        DebugDraw::get().line(origin, hit.has_value() ? hit.value().point : origin + glm::vec2(0.f, STEP_RAY_DIST),
                               hit.has_value() ? 0x00FFFFFF : 0xFF00FFFF);
        if (!hit.has_value()) continue;

        step_target(left) = hit.value().point;
        return true;
    }
    return false;
}

bool PlayerCharacter::foot_near_step_target(bool left)
{
    return glm::length(pe->get_position(foot_id(left)) - step_target(left)) <= foot_land_dist;
}

bool PlayerCharacter::try_land_foot(bool left)
{
    if (!grounded(left) || !foot_near_step_target(left)) return false;
    anchor_foot(left);
    return true;
}

// Single-ray step to target_x, IK the leg toward it, and anchor once the foot lands
bool PlayerCharacter::settle_foot(bool left, float target_x)
{
    if (!scan_step_target(left, target_x, target_x)) return false;
    leg_target(left) = reach_step_target(hip_position(left), left);
    return try_land_foot(left);
}

LegPose PlayerCharacter::reach_step_target(glm::vec2 hip, bool left)
{
    return solve_leg_ik(hip, step_target(left) + glm::vec2(0.f, ANKLE_RAISE), facing_dir);
}

LegPose PlayerCharacter::standing_planted_pose() const
{
    float lower = facing_dir * standing_lower_lean_angle;
    return { lower + facing_dir * standing_thigh_bend_angle, lower };
}

LegPose PlayerCharacter::walking_stance_pose(float t) const
{
    float   stance_end = facing_dir * stance_lean_angle;
    LegPose pose { glm::mix(stance_phase_start.thigh, stance_end, t), 0.f };

    if (t < 2.f/3.f) {
        pose.lower = glm::mix(stance_phase_start.lower, stance_end, t);
    } else {
        float lower_at_two_thirds = glm::mix(stance_phase_start.lower, stance_end, 2.f/3.f);
        float kick_end            = stance_end + facing_dir * trailing_kick_angle;
        float t3                  = (t - 2.f/3.f) / (1.f/3.f);
        pose.lower = glm::mix(lower_at_two_thirds, kick_end, t3);
    }
    return pose;
}

LegPose PlayerCharacter::walking_swing_pose(bool swing_left, float t)
{
    float thigh_bent_up = -facing_dir * swing_thigh_forward_angle; 
    float lower_bent_up =  facing_dir * recovery_bend_angle;      

    if (t < 1.f/3.f) {
        float t3 = t / (1.f/3.f);
        return { glm::mix(swing_phase_start.thigh, thigh_bent_up, t3),
                 glm::mix(swing_phase_start.lower, lower_bent_up, t3) };
    }
    if (t < 2.f/3.f) {
        float t3 = (t - 1.f/3.f) / (1.f/3.f);
        return { thigh_bent_up, glm::mix(lower_bent_up, thigh_bent_up, t3) };
    }
    float     t3   = (t - 2.f/3.f) / (1.f/3.f);
    glm::vec2 diff = step_target(swing_left) + glm::vec2(0.f, ANKLE_RAISE) - hip_position(swing_left);
    float straight_angle = std::atan2(-diff.x, diff.y); // same convention as solve_leg_ik's chain_phi
    return { glm::mix(thigh_bent_up, straight_angle, t3), glm::mix(thigh_bent_up, straight_angle, t3) };
}

// Knees stay bent throughout: the calf trails the thigh by knee_bend
LegPose PlayerCharacter::backward_bent_pose(float thigh, float knee_bend) const
{
    return { thigh, thigh + facing_dir * knee_bend };
}

// Planted leg rotates from behind the body to in front of it, pushing the body backward
LegPose PlayerCharacter::backward_stance_pose(float t) const
{
    LegPose end = backward_bent_pose(-facing_dir * backward_stance_lean_angle, backward_knee_bend);
    return { glm::mix(stance_phase_start.thigh, end.thigh, t),
             glm::mix(stance_phase_start.lower, end.lower, t) };
}

// Lifted leg raises its thigh forward a bit with the calf tucked, then swings back and
// unfolds to reach the step target
LegPose PlayerCharacter::backward_swing_pose(bool swing_left, float t)
{
    LegPose lifted = backward_bent_pose(swing_phase_start.thigh - facing_dir * backward_swing_lift_angle,
                                        backward_swing_knee_bend);

    if (t < 1.f/3.f) {
        float t3 = t / (1.f/3.f);
        return { glm::mix(swing_phase_start.thigh, lifted.thigh, t3),
                 glm::mix(swing_phase_start.lower, lifted.lower, t3) };
    }
    float     t3   = (t - 1.f/3.f) / (2.f/3.f);
    glm::vec2 diff = step_target(swing_left) + glm::vec2(0.f, ANKLE_RAISE) - hip_position(swing_left);
    LegPose   reach = backward_bent_pose(std::atan2(-diff.x, diff.y), backward_knee_bend);
    return { glm::mix(lifted.thigh, reach.thigh, t3), glm::mix(lifted.lower, reach.lower, t3) };
}

LegPose PlayerCharacter::crouch_pose() const
{
    float thigh = -facing_dir * jump_squat_thigh_angle;
    return { thigh, thigh + facing_dir * jump_squat_knee_bend };
}

// Thighs raised forward with calves tucked while rising; thighs drop and calves partly untuck
// once falling (y points down, so rising is negative velocity)
LegPose PlayerCharacter::air_pose(bool left) const
{
    bool  rising = fall_time < jump_fall_delay;   // keep the tuck for a moment after the peak
    float raise  = rising ? (left ? jump_rise_thigh_l : jump_rise_thigh_r)
                          : (left ? jump_fall_thigh_l : jump_fall_thigh_r);
    float bend   = rising ? jump_rise_knee_bend : jump_fall_knee_bend;
    float thigh  = -facing_dir * raise;
    return { thigh, thigh + facing_dir * bend };
}

void PlayerCharacter::begin_gait_phase()
{
    stance_phase_start = leg_angle(stance_left);
    swing_phase_start  = leg_angle(!stance_left);
}

// ── update() stages ───────────────────────────────────────────────────────────

void PlayerCharacter::update_standing_anchors()
{
    gait_timer = 0.f;

    if (foot_anchored_l || foot_anchored_r) {
        for (bool left : { true, false })
            if (!foot_anchored(left)) try_land_foot(left);
        return;
    }

    // Nothing anchored yet: pin grounded feet where they are
    for (bool left : { true, false })
        if (grounded(left)) stance_left = left;
    anchor_grounded_feet();
    standing_center_x = position().x;
    begin_gait_phase();
}

void PlayerCharacter::anchor_grounded_feet()
{
    for (bool left : { true, false })
        if (grounded(left) && !foot_anchored(left))
            anchor_foot(left);
}

AnimationState PlayerCharacter::next_animation_state() const
{
    bool on_ground = grounded_l || grounded_r;

    if (jump_launched)
        return AnimationState::Jumping;
    if (anim_state == AnimationState::Jumping && state_time < JUMP_ANIMATION_LENGTH)
        return AnimationState::Jumping;

    if (!on_ground) {
        float delay = anim_state == AnimationState::Running ? run_airborne_delay : AIRBORNE_DELAY;
        if (anim_state == AnimationState::Jumping || air_time >= delay)
            return AnimationState::Airborne;
        return anim_state;
    }

    if (anim_state == AnimationState::Jumping || anim_state == AnimationState::Airborne)
        return AnimationState::Landing;
    if (anim_state == AnimationState::Landing && state_time < LANDING_ANIMATION_LENGTH)
        return AnimationState::Landing;
    if (jump_timer > 0.f)
        return AnimationState::JumpSquat;
    // break stride if going from running to stopping state
    if (run_stopping()) {
        if (std::abs(velocity().x) > run_stop_speed || run_phase != RunPhase::Drive)
            return AnimationState::Running;
        return move_dir == 0.f ? AnimationState::Standing : AnimationState::Walking;
    }
    if (move_dir == 0.f)
        return AnimationState::Standing;
    return (walking || walking_backward()) ? AnimationState::Walking : AnimationState::Running;
}

void PlayerCharacter::update_animation_state(float dt)
{
    state_time += dt;

    AnimationState next = next_animation_state();
    if (next != anim_state)
        enter_animation_state(next);
}

void PlayerCharacter::enter_animation_state(AnimationState next)
{
    AnimationState prev = anim_state;
    anim_state    = next;
    state_time    = 0.f;
    jump_launched = false;

    switch (next) {
    case AnimationState::Standing:
        settling = prev == AnimationState::Walking || prev == AnimationState::Running
                || prev == AnimationState::Landing;
        break;
    case AnimationState::Walking:
        gait_timer = 0.f;
        begin_gait_phase();
        break;
    case AnimationState::Running:
        begin_run_cycle();
        break;
    case AnimationState::JumpSquat: {
        settling     = false;
        squat_vel_x  = com_velocity_x();
        squat_moving = (prev == AnimationState::Running || prev == AnimationState::Walking)
                    && std::abs(squat_vel_x) > run_stop_speed;
        if (!squat_moving) {
            anchor_grounded_feet();
            break;
        }
        // Moving: squat on the planted foot only, keeping it anchored
        if (foot_anchored_l != foot_anchored_r) squat_left = foot_anchored_l;
        else if (foot_anchored_l)               squat_left = stance_left;
        else                                    squat_left = grounded_l || !grounded_r;
        if (!foot_anchored(squat_left) && grounded(squat_left))
            anchor_foot(squat_left);
        release_anchor(!squat_left);
        break;
    }
    case AnimationState::Landing:
        anchor_grounded_feet();
        break;
    case AnimationState::Jumping:
    case AnimationState::Airborne:
        settling = false;
        release_anchor(true);
        release_anchor(false);
        break;
    }
}

void PlayerCharacter::set_animation_goals(float dt)
{
    switch (anim_state) {
    case AnimationState::Standing:
        if (settling) update_settling();
        else          update_standing();
        break;
    case AnimationState::Walking:
        update_walking(dt);
        break;
    case AnimationState::Running:
        update_running(dt);
        break;
    case AnimationState::JumpSquat: {
        if (!squat_moving) {
            anchor_grounded_feet();
            leg_target_l = leg_target_r = crouch_pose();
            break;
        }
        bool      plant = squat_left;
        glm::vec2 ankle = ankle_position(plant);
        glm::vec2 want  = hip_position(plant) + glm::vec2(squat_vel_x * dt, jump_squat_sink_speed * dt);
        want.y = std::min(want.y, ankle.y - jump_squat_min_hip_height);  
        leg_target(plant)  = solve_leg_ik(want, ankle, facing_dir);
        leg_target(!plant) = crouch_pose();
        if (!foot_pinned(plant))
            add_velocity_x(squat_vel_x - com_velocity_x());
        break;
    }
    case AnimationState::Landing:
        anchor_grounded_feet();
        leg_target_l = leg_target_r = crouch_pose();
        break;
    case AnimationState::Jumping:
        if (grounded_l || grounded_r) break;
        [[fallthrough]];
    case AnimationState::Airborne:
        leg_target_l = air_pose(true);
        leg_target_r = air_pose(false);
        break;
    }
}

// settling goes from running to standing - we  should take one step before we stop
void PlayerCharacter::update_settling()
{
    bool swing_left = !stance_left;

    if (!foot_anchored(swing_left)) {
        settle_foot(swing_left, position().x + side_x(swing_left) * settle_step_dist);
        leg_target(stance_left) = standing_planted_pose();
        return;
    }

    if (foot_anchored(stance_left))
        release_anchor(stance_left);

    float settled_x = pe->get_position(foot_id(swing_left)).x;
    if (settle_foot(stance_left, settled_x + side_x(stance_left) * feet_gap)) {
        settling = false;
        standing_center_x = position().x;
    }
    leg_target(swing_left) = standing_planted_pose();
}

void PlayerCharacter::update_standing()
{
    if (std::abs(velocity().x) > standing_settle_speed)
        standing_center_x = position().x;

    for (bool left : { true, false }) {
        if (grounded(left)) {
            leg_target(left) = standing_planted_pose();
            continue;
        }

        // Reach a lifted foot back down, searching from feet_gap out in toward center
        float side = side_x(left);
        scan_step_target(left, standing_center_x + side * feet_gap, standing_center_x);
        glm::vec2 centered_hip = glm::vec2(standing_center_x + side * HIP_X, position().y + HIP_Y);
        leg_target(left) = reach_step_target(centered_hip, left);
    }
}

void PlayerCharacter::update_walking(float dt)
{
    float period = running_backward() ? backward_run_step_period
                 : walking_backward() ? backward_step_period : step_period;
    gait_timer = std::min(gait_timer + dt, period);
    float t_pre = std::clamp(gait_timer / period, 0.f, 1.f);

    bool pre_swing_left = !stance_left;
    bool switched = t_pre >= 1.f/3.f &&
                    (grounded(pre_swing_left) || (t_pre >= 2.f/3.f && foot_near_step_target(pre_swing_left)));

    if (t_pre >= 2.f/3.f && !switched) release_anchor(stance_left);

    if (switched) {
        gait_timer  = 0.f;
        stance_left = !stance_left;
        begin_gait_phase();
    }
    bool swing_left = !stance_left;

    release_anchor(swing_left);

    // Search ahead of the torso in the direction of travel and traverse backwards to find a point
    float travel   = move_dir > 0.f ? 1.f : -1.f;
    float offset   = walking_backward() ? backward_search_offset_x : swing_search_offset_x;
    float search_x = position().x + travel * offset;
    scan_step_target(swing_left, search_x, search_x - travel * swing_search_back);

    if (switched)
        anchor_foot(stance_left);

    float t = std::clamp(gait_timer / period, 0.f, 1.f);
    if (walking_backward()) {
        leg_target(stance_left) = backward_stance_pose(t);
        leg_target(swing_left)  = backward_swing_pose(swing_left, t);
    } else {
        leg_target(stance_left) = walking_stance_pose(t);
        leg_target(swing_left)  = walking_swing_pose(swing_left, t);
    }
}

void PlayerCharacter::begin_run_cycle()
{
    bool stance = foot_anchored_l != foot_anchored_r ? foot_anchored_l : stance_left;
    stance_left = stance;
    if (!foot_anchored(stance) && grounded(stance))
        anchor_foot(stance);
    release_anchor(!stance);

    enter_run_phase(RunPhase::Drive);
    run_prev_vel_x = velocity().x;
}

void PlayerCharacter::enter_run_phase(RunPhase phase)
{
    run_phase      = phase;
    run_phase_time = 0.f;
}

// Direction the run cycle steps in: facing, except when braking after turning around mid-run,
// where the body is still sliding the old way
float PlayerCharacter::run_travel_dir() const
{
    return (run_stopping() && velocity().x * facing_dir < 0.f) ? -facing_dir : facing_dir;
}

// How far the hip has passed over the foot along dir
float PlayerCharacter::leg_over_foot_angle(bool left, float dir) const
{
    glm::vec2 foot = pe->get_position(foot_id(left));
    glm::vec2 hip  = hip_position(left);
    return std::atan2(dir * (hip.x - foot.x), foot.y - hip.y);
}

void PlayerCharacter::update_running(float dt)
{
    run_phase_time += dt;

    bool stance = stance_left;
    bool swing  = !stance;

    // Leave Drive once the leg is far enough behind or as soon as the anchor is lost (pin broke),
    if (run_phase == RunPhase::Drive &&
        (!foot_anchored(stance) || leg_over_foot_angle(stance, run_travel_dir()) >= run_release_angle)) {
        bool pushed_off = foot_pinned(stance);
        release_anchor(stance);
        if (pushed_off)
            for (uint32_t id : limbs)
                if (id != INVALID && pe->world.bodies[id].inv_mass > 0.f) {
                    // this could be used to get a bouncy run - for now set to 0
                    float& vy = pe->world.bodies[id].velocity.y;
                    vy = std::min(vy, -run_lift_speed);
                }
        enter_run_phase(RunPhase::Flight);
    }
    if (run_phase == RunPhase::Flight && run_phase_time >= run_flight_delay)
        enter_run_phase(RunPhase::Reach);
    if (run_phase == RunPhase::Reach) {
        float search_x = position().x + run_travel_dir() * run_step_ahead * std::max(run_stop_scale(), 0.3f);
        scan_step_target(swing, search_x, search_x - run_travel_dir() * swing_search_back);

        // Require real ground contact: anchoring a foot that's merely near its target lets
        // Drive run (and lift) while still airborne
        if (grounded(swing)) {
            if (!foot_near_step_target(swing))
                step_target(swing).x = pe->get_position(foot_id(swing)).x;
            anchor_foot(swing);
            stance_left = swing;
            stance      = swing;
            swing       = !swing;
            enter_run_phase(RunPhase::Drive);
        }
    }

    LegPose forward_tucked = { -facing_dir * run_swing_thigh_angle,
                               -facing_dir * run_swing_thigh_angle + facing_dir * run_tuck_knee_bend };
    float   back_thigh      = leg_angle(stance).thigh;
    LegPose trailing_tucked = { back_thigh, back_thigh + facing_dir * run_tuck_knee_bend };

    switch (run_phase) {
    case RunPhase::Drive: {
        glm::vec2 lead = glm::vec2(facing_dir * run_drive_lead, -run_drive_lift);
        if (run_stopping()) {
            float speed = std::max(std::abs(velocity().x) - run_stop_decel * dt, 0.f);
            lead = glm::vec2(run_travel_dir() * speed * dt, 0.f);
        }
        leg_target(stance) = solve_leg_ik(hip_position(stance) + lead, ankle_position(stance), facing_dir);
        leg_target(swing)  = forward_tucked;
        break;
    }
    case RunPhase::Flight:
        leg_target(swing)  = forward_tucked;
        leg_target(stance) = trailing_tucked;   
        break;
    case RunPhase::Reach: {
        leg_target(swing)  = reach_step_target(hip_position(swing), swing);
        leg_target(stance) = trailing_tucked;
        break;
    }
    }

    run_speed_control(dt);
}

// 1 while running normally; while braking, shrinks with speed so strides shorten
float PlayerCharacter::run_stop_scale() const
{
    if (!run_stopping() || run_speed_target <= 0.f) return 1.f;
    return std::clamp(std::abs(velocity().x) / run_speed_target, 0.f, 1.f);
}

void PlayerCharacter::run_speed_control(float dt)
{
    if (dt <= 0.f) return;

    float mass = 0.f, vel_x = 0.f;
    for (uint32_t id : limbs) {
        if (id == INVALID) continue;
        const RigidBody& rb = pe->world.bodies[id];
        mass  += rb.mass;
        vel_x += rb.mass * rb.velocity.x;
    }
    if (mass <= 0.f) return;
    vel_x /= mass;

    // Derivative on velocity rather than error, so switching the target to 0 when stopping doesn't kick
    float target    = run_stopping() ? 0.f : facing_dir * run_speed_target;
    float error     = target - vel_x;
    float derror    = -(vel_x - run_prev_vel_x) / dt;
    run_prev_vel_x  = vel_x;

    float max_accel = run_stopping() ? run_stop_decel : run_max_accel;
    float accel     = std::clamp(run_speed_kp * error + run_speed_kd * derror, -max_accel, max_accel);
    for (uint32_t id : limbs)
        if (id != INVALID && pe->world.bodies[id].inv_mass > 0.f)
            pe->world.bodies[id].velocity.x += accel * dt;
}

void PlayerCharacter::drive_legs(float dt)
{
    float torso_angle = rotation();
    bool  in_air      = anim_state == AnimationState::Jumping || anim_state == AnimationState::Airborne;
    float speed       = in_air                                 ? air_leg_angle_speed
                      : anim_state != AnimationState::Walking ? max_leg_angle_speed
                      : running_backward()                     ? backward_run_leg_angle_speed
                                                               : walk_leg_angle_speed;
    float max_d       = speed * dt;

    for (bool left : { true, false }) {
        LegPose&       angle  = leg_angle(left);
        const LegPose& target = leg_target(left);
        step_leg_angle(angle.thigh, target.thigh, max_d);
        step_leg_angle(angle.lower, target.lower, max_d);

        // point toes when running, flatten before landing
        float& foot = foot_angle(left);
        bool   running = anim_state == AnimationState::Running;
        if (running && foot_pinned(left)) {
            foot = wrap_angle(pe->world.bodies[foot_id(left)].position.z);
        } else {
            bool  reaching    = running && run_phase == RunPhase::Reach && left != stance_left;
            float foot_target = (running && !reaching) ? -facing_dir * run_foot_point_angle : 0.f;
            step_leg_angle(foot, foot_target, foot_angle_speed * dt);
        }

        animate_leg_fk(left ? Joint::HipL   : Joint::HipR,
                       left ? Joint::KneeL  : Joint::KneeR,
                       left ? Joint::AnkleL : Joint::AnkleR,
                       torso_angle, angle.thigh, angle.lower, foot);
    }
}

void PlayerCharacter::handle_jump(float dt)
{
    if (jump_launched || anim_state == AnimationState::Jumping) return;

    if (!grounded_l && !grounded_r) {
        will_jump      = false;
        jump_dir_timer = 0.0f;
        jump_timer     = 0.0f;
        return;
    }

    if (jump || will_jump) {
        jump_timer += dt;
        if (jump_timer >= 0.0f) jump_dir_timer += dt * move_dir;
    }
    if (!jump && jump_timer > 0.0f)       will_jump = true;
    if (jump_timer >= MAX_JUMP_CUTOFF)    will_jump = true;

    // running squat 
    if (anim_state == AnimationState::JumpSquat && squat_moving) {
        float travel = squat_vel_x >= 0.f ? 1.f : -1.f;
        if (!foot_anchored(squat_left) ||
            leg_over_foot_angle(squat_left, travel) >= run_release_angle) {
            will_jump  = true;
            jump_timer = std::max(jump_timer, MIN_JUMP_CUTOFF);
        }
    }

    if (will_jump && jump_timer >= MIN_JUMP_CUTOFF)
        launch_jump();
}

// Mass-weighted horizontal velocity of the movable limbs
float PlayerCharacter::com_velocity_x() const
{
    float mass = 0.f, vel_x = 0.f;
    for (uint32_t id : limbs) {
        if (id == INVALID || pe->world.bodies[id].inv_mass <= 0.f) continue;
        mass  += pe->world.bodies[id].mass;
        vel_x += pe->world.bodies[id].mass * pe->world.bodies[id].velocity.x;
    }
    return mass > 0.f ? vel_x / mass : 0.f;
}

// Shift every movable limb's horizontal velocity by the same amount
void PlayerCharacter::add_velocity_x(float dv)
{
    for (uint32_t id : limbs)
        if (id != INVALID && pe->world.bodies[id].inv_mass > 0.f)
            pe->world.bodies[id].velocity.x += dv;
}

void PlayerCharacter::launch_jump()
{
    glm::vec2 impulse = jump_impulse * glm::vec2(JUMP_RATIO.x * jump_dir_timer, -JUMP_RATIO.y * jump_timer);

    release_anchor(true);
    release_anchor(false);

    float total_mass = 0.f;
    for (uint32_t id : limbs)
        if (id != INVALID) total_mass += pe->world.bodies[id].mass;
    if (total_mass > 0.f) {
        glm::vec2 dv = impulse / total_mass;
        for (uint32_t id : limbs)
            if (id != INVALID) pe->set_velocity(id, pe->get_velocity(id) + dv);
    }

    will_jump      = false;
    jump_timer     = -JUMP_DELAY;
    jump_dir_timer = 0.0f;
    jump_launched  = true;
}

void PlayerCharacter::update(float dt, bool apply_controls)
{
    if (!is_valid()) return;
    if (apply_controls)
        update_facing();

    grounded_l = foot_grounded(true);
    grounded_r = foot_grounded(false);
    air_time   = (grounded_l || grounded_r) ? 0.f : air_time + dt;

    if (apply_controls)
        handle_jump(dt);

    update_animation_state(dt);

    bool in_air = anim_state == AnimationState::Jumping || anim_state == AnimationState::Airborne;
    fall_time   = (in_air && velocity().y >= 0.f) ? fall_time + dt : 0.f;   

    if (anim_state == AnimationState::Standing)
        update_standing_anchors();

    sync_foot_pin(true);
    sync_foot_pin(false);

    // While running, traction comes from pinning; a free foot with friction just snags on the ground.
    for (bool left : { true, false }) {
        bool slick = (anim_state == AnimationState::Running && !foot_pinned(left))
                  || anim_state == AnimationState::JumpSquat || anim_state == AnimationState::Jumping;
        pe->world.bodies[foot_id(left)].friction = slick ? 0.f : (left ? foot_friction_l : foot_friction_r);
    }

    leg_target_l = leg_angle_l;
    leg_target_r = leg_angle_r;

    if (apply_controls)
        set_animation_goals(dt);

    drive_legs(dt);

    animate_left_arm();
    animate_right_arm();
    apply_joint_goals();
}

glm::vec2 PlayerCharacter::position() const { return pe->get_position(limbs[(size_t)Limb::Torso]); }
glm::vec2 PlayerCharacter::velocity() const { return pe->get_velocity(limbs[(size_t)Limb::Torso]); }
float     PlayerCharacter::rotation() const { return pe->get_rotation(limbs[(size_t)Limb::Torso]); }
