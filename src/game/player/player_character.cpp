#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <src/game/debug_draw.h>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>
#include <fmt/core.h>

// ── JSON id → enum mappings ────────────────────────────────────────────────────
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

static constexpr float TORSO_HALF_H      = 10.f;

// ── Limb skeleton ---------------------------------────────────────────────────
static constexpr float THIGH_LEN        = 10.f;
static constexpr float THIGH_MID        =  5.f;
static constexpr float LOWER_LEN        = 10.f;
static constexpr float LOWER_MID        =  5.f;
static constexpr float LOWER_TO_ANKLE   = 10.f;
static constexpr float HIP_X            =  3.f;
static constexpr float HIP_Y            =  8.f;
static constexpr float ANKLE_RAISE      =  1.f;
static constexpr float FOOT_OFF_X       =  2.f;

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


// ─────────────────────────────────────────────────────────────────────────────

void PlayerCharacter::load_assets(const char* config_path)
{
    if (!char_config.load(config_path)) {
        fmt::print("[PlayerCharacter] Failed to load character config: {}\n", config_path);
        return;
    }
    limb_images.clear();
    for (const auto& [id, sprite] : char_config.limb_sprites)
        limb_images[id] = load_body_image(("../" + sprite).c_str());

    anim_standing.load("../assets/animations/standing.json");

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

uint32_t PlayerCharacter::spawn_limb(PhysicsEngine& pe, ResiduaEngine& re,
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
    return pe.add_body(&re, std::move(rb));
}

void PlayerCharacter::add_joint(PhysicsEngine& pe, Joint jnt, Limb parent, Limb child,
                                 glm::vec2 rA_local, glm::vec2 rB_local,
                                 float /*bend_stiffness*/, float /*max_torque*/)
{
    auto j = std::make_unique<DistanceJoint>(
        &pe.world,
        limbs[(size_t)parent], limbs[(size_t)child],
        rA_local, rB_local,
        std::numeric_limits<float>::infinity(),
        joint_bend_stiffness);
    joint_ptrs[(size_t)jnt] = j.get();
    bool left_arm = (jnt == Joint::ShoulderL || jnt == Joint::ElbowL || jnt == Joint::WristL);
    if (left_arm)
        j->angular_reaction = false;
    pe.world.add_force(std::move(j));
}

void PlayerCharacter::set_rest_angle(Joint jnt, float parent_angle, float child_angle)
{
    joint_goal_angle[(size_t)jnt] = parent_angle - child_angle;
}

void PlayerCharacter::spawn(PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 pos)
{
    limbs.fill(INVALID);
    joint_ptrs.fill(nullptr);

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

            limbs[(size_t)lit->second] = pe.add_body(&re, std::move(rb));
        } else {
            limbs[(size_t)lit->second] = spawn_limb(pe, re, limb_images.at(lid), spawn_pos, density);
        }
    }

    // Create joints from JSON definition
    for (const auto& [jid, jc] : char_config.joints) {
        auto jit  = JOINT_FROM_ID.find(jid);
        auto plit = LIMB_FROM_ID.find(jc.parent_limb);
        auto clit = LIMB_FROM_ID.find(jc.child_limb);
        if (jit == JOINT_FROM_ID.end() || plit == LIMB_FROM_ID.end() || clit == LIMB_FROM_ID.end())
            continue;
        add_joint(pe, jit->second, plit->second, clit->second,
                  jc.attach_parent, jc.attach_child, -1.f, jc.max_torque);
    }

    // Initialise goal angles from JSON defaults
    joint_goal_angle.fill(0.f);
    for (const auto& [jid, jc] : char_config.joints) {
        auto jit = JOINT_FROM_ID.find(jid);
        if (jit != JOINT_FROM_ID.end())
            joint_goal_angle[(size_t)jit->second] = jc.default_angle;
    }

    // Assign draw layers from JSON draw_order:
    //   items before "torso" → layer 1, "torso"/"head" → layer 2, after "head" → layer 3
    // layers are used to render certain parts of the body over others. these are fed into rendering draw layers
    // for future reference, we will need to fit this into a larger layer system to allow for backgrounds / other layers
    int layer = 1;
    for (const auto& lid : char_config.draw_order) {
        if (lid == "torso") layer = 2;
        auto lit = LIMB_FROM_ID.find(lid);
        if (lit != LIMB_FROM_ID.end()) {
            uint32_t bid = limbs[(size_t)lit->second];
            if (bid != INVALID)
                pe.world.bodies[bid].draw_layer = layer;
        }
        if (lid == "head") layer = 3;
    }
}

void PlayerCharacter::despawn(PhysicsEngine& pe)
{
    upright_timer_   = 0.f;
    height_timer_    = 0.f;
    low_speed_timer_ = 0.f;
    if (!is_valid()) return;

    for (auto* j : joint_ptrs)
        if (j) pe.world.remove_force(j);
    joint_ptrs.fill(nullptr);

    for (uint32_t id : limbs)
        if (id != INVALID) pe.remove_body(id);
    limbs.fill(INVALID);
}

void PlayerCharacter::reset_to(PhysicsEngine& pe, glm::vec2 pos)
{
    if (!is_valid()) return;

    auto offsets = char_config.compute_spawn_offsets();

    for (const auto& [lid, offset] : offsets) {
        auto lit = LIMB_FROM_ID.find(lid);
        if (lit == LIMB_FROM_ID.end()) continue;
        size_t idx = (size_t)lit->second;
        if (limbs[idx] == INVALID) continue;
        pe.set_position        (limbs[idx], pos + offset);
        pe.set_rotation        (limbs[idx], 0.f);
        pe.set_velocity        (limbs[idx], { 0.f, 0.f });
        pe.set_angular_velocity(limbs[idx], 0.f);
    }

    left_step_goal  = pos + offsets.at("foot_l");
    right_step_goal = pos + offsets.at("foot_r");
    left_step_target        = left_step_goal;
    right_step_target       = right_step_goal;
    left_step_normal        = { 0.f, -1.f };
    right_step_normal       = { 0.f, -1.f };
    left_step_normal_target = { 0.f, -1.f };
    right_step_normal_target= { 0.f, -1.f };

    left_airborne  = false;
    right_airborne = false;
    stride_counter = 0.f;
    jump_timer     = 0.f;
    grounded       = false;
    was_grounded   = false;
    facing_dir     = 1.f;
    upright_timer_   = 0.f;
    height_timer_    = 0.f;
    low_speed_timer_ = 0.f;
    state_history   = {};
    current_action  = {};
    previous_action = {};
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

    {
        auto hit = pe.raycast(torso_pos, glm::vec2(0.f, 1.f), ground_check_dist, PLAYER_MASK);
        grounded = hit.has_value();
    }

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

void PlayerCharacter::update(PhysicsEngine& pe, float dt, bool apply_controls)
{
    if (!is_valid()) return;
    if (apply_controls) {
        handle_controls(pe, dt);
        animate(pe, dt);
    }
    apply_joint_goals(pe);

    // Righting torque: PD controller drives torso angle toward 0 (upright)
    {
        uint32_t torso_id = limbs[(size_t)Limb::Torso];
        float angle = pe.get_rotation(torso_id);
        angle = std::atan2(std::sin(angle), std::cos(angle)); // wrap to [-π, π]
        float omega = pe.get_angular_velocity(torso_id);
        float torque = -torso_upright_stiffness * angle - torso_upright_damping * omega;
        pe.apply_torque(torso_id, torque);
    }

    // Advance animation frame 
    const AnimationClip* clip = nullptr;
    if (anim_state == PlayerAnimState::Standing) clip = &anim_standing;
    if (clip && !clip->empty()) {
        anim_frame += 1.f;
        if (anim_frame >= float(clip->length))
            anim_frame = 0.f;
    }

    capture_state(pe);

    if (state_history.count > 0) {
        const MovementState& cur = state_history.at(0);
        if (std::abs(cur.torso_angle) < upright_angle_threshold)
            upright_timer_ += dt;
        else
            upright_timer_ = 0.f;

        if (cur.ground_dist < 9999.f && std::abs(cur.ground_dist - standing_height) < height_threshold)
            height_timer_ += dt;
        else
            height_timer_ = 0.f;

        if (std::abs(cur.torso_velocity.x) < low_speed_threshold)
            low_speed_timer_ += dt;
        else
            low_speed_timer_ = 0.f;
    }
}

void PlayerCharacter::apply_action(PhysicsEngine& pe, const Action& action)
{
    if (!is_valid()) return;
    previous_action = current_action;
    current_action  = action;
    for (size_t i = 0; i < (size_t)Joint::Count; ++i) {
        if (i == (size_t)Joint::ShoulderL || i == (size_t)Joint::ElbowL || i == (size_t)Joint::WristL)
            continue;
        float goal = std::clamp(action.goal_angle_change[i], joint_angle_min[i], joint_angle_max[i]);
        current_action.goal_angle_change[i] = goal;
        joint_goal_angle[i] = goal;
    }
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
    float raw_angle          = pe.get_rotation(torso_id);
    s.torso_angle            = std::atan2(std::sin(raw_angle), std::cos(raw_angle));
    s.torso_angular_velocity = pe.get_angular_velocity(torso_id);

    for (size_t i = 0; i < (size_t)Joint::Count; ++i) {
        float pa  = pe.get_rotation        (limbs[(size_t)JOINT_LIMBS[i].parent]);
        float ca  = pe.get_rotation        (limbs[(size_t)JOINT_LIMBS[i].child]);
        float pav = pe.get_angular_velocity(limbs[(size_t)JOINT_LIMBS[i].parent]);
        float cav = pe.get_angular_velocity(limbs[(size_t)JOINT_LIMBS[i].child]);
        float rel = pa - ca;
        s.joint_angles[i]              = std::atan2(std::sin(rel), std::cos(rel));
        s.joint_angular_velocities[i]  = pav - cav;
    }

    glm::vec2 torso_pos   = pe.get_position(torso_id);
    float     torso_angle = pe.get_rotation(torso_id);
    s.torso_pos = torso_pos;

    auto wall_hit = pe.raycast(torso_pos, glm::vec2(facing_dir, 0.f), wall_check_dist, PLAYER_MASK);
    s.wall_ahead  = wall_hit.has_value();

    auto ground_hit = pe.raycast(torso_pos, glm::vec2(0.f, 1.f), 300.f, PLAYER_MASK);
    s.ground_dist   = ground_hit.has_value() ? ground_hit.value().distance : 9999.f;
    s.grounded      = ground_hit.has_value() && s.ground_dist <= ground_check_dist;

    s.left_shoulder_pos = torso_pos + glm::rotate(glm::vec2(-SHOULDER_X, SHOULDER_Y), torso_angle);
    s.left_arm_angle    = pe.get_rotation(limbs[(size_t)Limb::UpperArmL]);
    s.facing_dir        = facing_dir;
    glm::vec2 foot_l_pos = pe.get_position(limbs[(size_t)Limb::FootL]);
    glm::vec2 foot_r_pos = pe.get_position(limbs[(size_t)Limb::FootR]);
    s.feet_mid_x        = (foot_l_pos.x + foot_r_pos.x) * 0.5f;
    s.feet_mid_y        = (foot_l_pos.y + foot_r_pos.y) * 0.5f;

    for (uint32_t i = 0; i < (uint32_t)Limb::Count; ++i) {
        if (limbs[i] != INVALID)
            s.limb_velocities[i] = pe.get_velocity(limbs[i]);
        else
            s.limb_velocities[i] = { 0.f, 0.f };
    }

    get_anim_goals(s.anim_goal_current, s.anim_goal_next);

    state_history.push(s);
}

// RL reward system for teaching controls. Does not work very well with the existing rewards
float PlayerCharacter::compute_reward(float dt) const
{
    if (state_history.count == 0) return 0.f;
    const MovementState& cur = state_history.at(0);

    float reward = reward_alive;

    // 1. Height: how far the torso is above the ground (0 lying flat → 1 at standing_height)
    if (cur.ground_dist < 9999.f)
        reward += reward_height * std::min(1.f, cur.ground_dist / standing_height);

    // 2. Upright: cos(torso_angle), 1 vertical → 0 horizontal → negative upside-down
    reward += reward_upright * std::max(0.f, std::cos(cur.torso_angle));

    // 3. Feet below torso (Y increases downward, so positive diff = feet below torso)
    float feet_below = cur.feet_mid_y - cur.torso_pos.y;
    reward += reward_feet_below * std::min(1.f, std::max(0.f, feet_below / standing_height));

    // 3b. Feet horizontally near torso
    float feet_dx = std::abs(cur.feet_mid_x - cur.torso_pos.x);
    reward += reward_feet_near_x * std::max(0.f, 1.f - feet_dx / feet_near_x_threshold);

    // 4. Velocity penalty: linear and angular separately
    reward -= penalty_velocity   * glm::dot(cur.torso_velocity, cur.torso_velocity);
    reward -= penalty_torso_spin * cur.torso_angular_velocity * cur.torso_angular_velocity;

    // 5. Action smoothness: penalize rapid changes in joint goal angles
    {
        float total = 0.f;
        for (int i = 0; i < (int)JOINT_COUNT; ++i) {
            float d = current_action.goal_angle_change[i] - previous_action.goal_angle_change[i];
            float d2 = d * d;
            total += d2 * d2;
        }
        reward -= penalty_action_rate * total;
    }

    // 7. Joint limit penalty: count joints beyond threshold
    for (int i = 0; i < (int)JOINT_COUNT; ++i)
        if (std::abs(cur.joint_angles[i]) > joint_limit_threshold)
            reward -= penalty_joint_limit;

    // 8. Animation conformance: reward joint angles matching current clip frame
    {
        float total = 0.f;
        int   count = 0;
        for (int i = 0; i < (int)JOINT_COUNT; ++i) {
            if (!joint_animatable[i]) continue;
            float err   = cur.joint_angles[i] - cur.anim_goal_current[i];
            // Wrap error into [-π, π]
            err = std::atan2(std::sin(err), std::cos(err));
            float prox = std::max(0.f, 1.f - std::abs(err) / 1.5708f);
            total += prox;
            ++count;
        }
        if (count > 0)
            reward += reward_anim_match * total / float(count);
    }

    return reward;
}

void PlayerCharacter::get_anim_goals(std::array<float, JOINT_COUNT>& current_out,
                                      std::array<float, JOINT_COUNT>& next_out) const
{
    current_out.fill(0.f);
    next_out.fill(0.f);

    const AnimationClip* clip = nullptr;
    if (anim_state == PlayerAnimState::Standing) clip = &anim_standing;
    if (!clip || clip->empty()) return;

    float next_frame = std::fmod(anim_frame + 1.f, float(clip->length));

    auto fill = [&](float frame, std::array<float, JOINT_COUNT>& out) {
        for (const auto& [jid, angle] : clip->sample(frame)) {
            auto it = JOINT_FROM_ID.find(jid);
            if (it != JOINT_FROM_ID.end())
                out[(size_t)it->second] = angle;
        }
    };
    fill(anim_frame,  current_out);
    fill(next_frame,  next_out);
}

void PlayerCharacter::animate_left_arm(PhysicsEngine& pe)
{
    if (!is_valid()) return;
    glm::vec2 torso_pos   = pe.get_position(limbs[(size_t)Limb::Torso]);
    float     torso_angle = pe.get_rotation(limbs[(size_t)Limb::Torso]);

    glm::vec2 shoulder_l = torso_pos + glm::rotate(glm::vec2(-SHOULDER_X, SHOULDER_Y), torso_angle);

    // Project a virtual target far along the shoulder→mouse direction (avoid weird IK arm angles)
    glm::vec2 raw_dir = aim_pos - shoulder_l;
    float     raw_len = glm::length(raw_dir);
    glm::vec2 aim_dir = (raw_len > 0.001f) ? raw_dir / raw_len : glm::vec2(1.f, 0.f);
    glm::vec2 virtual_target = shoulder_l + aim_dir * 500.f;

    auto [ua, fa] = solve_arm_ik(shoulder_l, virtual_target);

    float target_shoulder = torso_angle - ua;
    float target_elbow    = ua - fa;

    float max_delta = arm_angle_speed * pe.world.last_dt;
    auto step = [&](float& tracked, Joint jnt, float target) {
        float diff = std::atan2(std::sin(target - tracked), std::cos(target - tracked));
        tracked += std::clamp(diff, -max_delta, max_delta);
        joint_goal_angle[(size_t)jnt] = tracked;
    };

    step(arm_shoulder_rest, Joint::ShoulderL, target_shoulder);
    step(arm_elbow_rest,    Joint::ElbowL,    target_elbow);
    step(arm_wrist_rest,    Joint::WristL,    0.f);
}

void PlayerCharacter::apply_joint_goals(PhysicsEngine& /*pe*/)
{
    if (!is_valid()) return;
    for (size_t i = 0; i < (size_t)Joint::Count; ++i) {
        if (!joint_ptrs[i]) continue;
        float goal = joint_limited[i]
                   ? std::clamp(joint_goal_angle[i], joint_angle_min[i], joint_angle_max[i])
                   : joint_goal_angle[i];
        joint_ptrs[i]->rest_angle   = goal;
        joint_ptrs[i]->stiffness[2] = joint_bend_stiffness;
    }
}

void PlayerCharacter::animate(PhysicsEngine& pe, float dt)
{
    if (!is_valid()) return;

    glm::vec2 torso_pos   = pe.get_position(limbs[(size_t)Limb::Torso]);
    float     torso_angle = pe.get_rotation(limbs[(size_t)Limb::Torso]);

    // TODO: fix direction facing
    // facing_dir = (aim_pos.x > torso_pos.x) ? 1.f : -1.f;
    // for (uint32_t id : limbs)
    //    if (id != INVALID) pe.world.bodies[id].flip_h = !(facing_dir > 0.f);

    animate_arms(pe, torso_pos, torso_angle);
    test_animate(pe);
}

void PlayerCharacter::test_animate(PhysicsEngine& pe)
{
    if (!is_valid()) return;

    const AnimationClip* clip = (anim_state == PlayerAnimState::Standing) ? &anim_standing : nullptr;
    if (!clip || clip->empty()) return;

    for (const auto& [jid, angle] : clip->sample(anim_frame)) {
        auto it = JOINT_FROM_ID.find(jid);
        if (it == JOINT_FROM_ID.end()) continue;

        Joint jnt = it->second;
        if (jnt == Joint::ShoulderL || jnt == Joint::ElbowL || jnt == Joint::WristL)
            continue;

        joint_goal_angle[(size_t)jnt] = angle;
    }
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

void PlayerCharacter::animate_arms(PhysicsEngine& pe, glm::vec2, float)
{
    animate_left_arm(pe);
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
