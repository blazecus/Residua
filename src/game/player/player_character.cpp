#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <src/game/debug_draw.h>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>

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
    rb.collision_mask  = PLAYER_MASK;
    rb.angular_damping = 0.98f;
    return pe.add_body(&re, std::move(rb));
}

void PlayerCharacter::connect(PhysicsEngine& pe,
                               JointID jid,
                               Limb parent, glm::vec2 anchor_parent,
                               Limb child,  glm::vec2 anchor_child,
                               float stiffness, float bend_stiffness)
{
    uint32_t idA = limbs[(size_t)parent];
    uint32_t idB = limbs[(size_t)child];

    auto* j = new DistanceJoint(&pe.world, idA, idB,
                                anchor_parent, anchor_child,
                                stiffness, bend_stiffness);
    joints[(size_t)jid] = { j, j->rest_angle, bend_stiffness };
    pe.world.add_force(std::unique_ptr<Force>(j));
}

void PlayerCharacter::spawn(PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 pos)
{
    limbs.fill(INVALID);

    // Hitbox: invisible physics master, drives movement and righting
    hitbox_id = spawn_limb(pe, re, img_hitbox, pos);
    pe.world.bodies[hitbox_id].visible =false;
    pe.world.bodies[hitbox_id].angular_damping = 0.99f;

    // Torso: kinematic — position driven directly from hitbox, not solved by AVBD
    limbs[(size_t)Limb::Torso] = spawn_limb(pe, re, img_torso, pos);
    {
        auto& rb      = pe.world.bodies[limbs[(size_t)Limb::Torso]];
        rb.inv_mass    = 0.f;
        rb.inv_inertia = 0.f;
    }

    // Head: sits above torso (torso top = -16, head bottom = +8)
    limbs[(size_t)Limb::Head]      = spawn_limb(pe, re, img_head,      pos + glm::vec2(  0, -24));

    // Arms (all limbs 16x16, top anchor = -8, bottom anchor = +8)
    limbs[(size_t)Limb::UpperArmL] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2(-8,   0));
    limbs[(size_t)Limb::ForearmL]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2(-8,  14));
    limbs[(size_t)Limb::HandL]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2(-8,  26));
    limbs[(size_t)Limb::UpperArmR] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2( 8,   0));
    limbs[(size_t)Limb::ForearmR]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2( 8,  14));
    limbs[(size_t)Limb::HandR]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2( 8,  26));

    // Legs
    limbs[(size_t)Limb::ThighL]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2( -8,  24));
    limbs[(size_t)Limb::LowerLegL] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2( -8,  40));
    limbs[(size_t)Limb::FootL]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2( -8,  56));
    limbs[(size_t)Limb::ThighR]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2(  8,  24));
    limbs[(size_t)Limb::LowerLegR] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2(  8,  40));
    limbs[(size_t)Limb::FootR]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2(  8,  56));

    const float INF      = std::numeric_limits<float>::infinity();
    const float BEND     = 10.f;
    const float LEG_BEND = 7.f;

    // Head
    connect(pe, JointID::Neck,      Limb::Torso,     {  0, -10}, Limb::Head,      {  0,  8}, INF, BEND);
    // Left arm
    connect(pe, JointID::ShoulderL, Limb::Torso,     {-8,  -11}, Limb::UpperArmL, {  0, -8}, INF, BEND);
    connect(pe, JointID::ElbowL,    Limb::UpperArmL, {  0,   4}, Limb::ForearmL,  {  0, -8}, INF, BEND);
    connect(pe, JointID::WristL,    Limb::ForearmL,  {  0,   1}, Limb::HandL,     {  0, -8}, INF, BEND);
    // Right arm
    connect(pe, JointID::ShoulderR, Limb::Torso,     { 8,  -11}, Limb::UpperArmR, {  0, -8}, INF, BEND);
    connect(pe, JointID::ElbowR,    Limb::UpperArmR, {  0,   4}, Limb::ForearmR,  {  0, -8}, INF, BEND);
    connect(pe, JointID::WristR,    Limb::ForearmR,  {  0,   1}, Limb::HandR,     {  0, -8}, INF, BEND);
    // Left leg
    connect(pe, JointID::HipL,      Limb::Torso,     { -3,  14}, Limb::ThighL,    {  0, -8}, INF, LEG_BEND);
    connect(pe, JointID::KneeL,     Limb::ThighL,    {  0,   7}, Limb::LowerLegL, {  0, -8}, INF, LEG_BEND);
    connect(pe, JointID::AnkleL,    Limb::LowerLegL, { 0,   2}, Limb::FootL,     {  0, -8}, INF, LEG_BEND);
    // Right leg
    connect(pe, JointID::HipR,      Limb::Torso,     {  3,  14}, Limb::ThighR,    {  0, -8}, INF, LEG_BEND);
    connect(pe, JointID::KneeR,     Limb::ThighR,    {  0,   7}, Limb::LowerLegR, {  0, -8}, INF, LEG_BEND);
    connect(pe, JointID::AnkleR,    Limb::LowerLegR, {  0,   2}, Limb::FootR,     {  0, -8}, INF, LEG_BEND);

}

void PlayerCharacter::despawn(PhysicsEngine& pe)
{
    if (!is_valid()) return;

    // Joints first — remove_body does not clean up forces.
    for (auto& pj : joints) {
        if (pj.joint) {
            pe.world.remove_force(pj.joint);
            pj.joint = nullptr;
        }
    }

    if (hitbox_id != INVALID) {
        pe.remove_body(hitbox_id);
        hitbox_id = INVALID;
    }
    for (uint32_t id : limbs)
        if (id != INVALID) pe.remove_body(id);

    limbs.fill(INVALID);
}

void PlayerCharacter::set_desired_angle(JointID jid, float angle)
{
    joints[(size_t)jid].desired_angle = angle;
}

// Analytical 2-joint IK for one leg.
// Returns {hip_rest_angle, knee_rest_angle} suitable for DistanceJoint::rest_angle.
static std::pair<float,float> solve_leg_ik(
    glm::vec2 hip_world, glm::vec2 foot_target, float parent_angle)
{
    const float L1 = 16.f, L2 = 16.f;

    glm::vec2 diff = foot_target - hip_world;
    float dist = std::clamp(glm::length(diff),
                            std::abs(L1 - L2) + 0.01f,
                            L1 + L2 - 0.01f);

    float cos_a = (L1*L1 + dist*dist - L2*L2) / (2.f * L1 * dist);
    float alpha  = std::acos(std::clamp(cos_a, -1.f, 1.f));
    float phi    = std::atan2(diff.y, diff.x);

    // Knee bends forward (−alpha offset keeps knee in front when leg points down)
    glm::vec2 thigh_dir = { std::cos(phi - alpha), std::sin(phi - alpha) };
    float thigh_angle   = std::atan2(-thigh_dir.x, thigh_dir.y);

    glm::vec2 knee_pos  = hip_world + thigh_dir * L1;
    glm::vec2 lower_dir = glm::normalize(foot_target - knee_pos);
    float lower_angle   = std::atan2(-lower_dir.x, lower_dir.y);

    // rest_angle = parent_angle - child_angle_desired
    return { parent_angle - thigh_angle, thigh_angle - lower_angle };
}

void PlayerCharacter::update(PhysicsEngine& pe, float dt)
{
    // Always push desired angles to the solver
    for (auto& pj : joints)
        if (pj.joint) pj.joint->rest_angle = pj.desired_angle;

    if (!is_valid() || hitbox_id == INVALID) return;

    // Hitbox drives movement and righting — torso follows via LeaderJoint
    glm::vec2 hbox_pos   = pe.get_position(hitbox_id);
    glm::vec2 hbox_vel   = pe.get_velocity(hitbox_id);
    float     hbox_angle = pe.get_rotation(hitbox_id);

    // Proportional velocity controller on hitbox
    float target_vx = move_dir * max_speed;
    pe.apply_force(hitbox_id, { (target_vx - hbox_vel.x) * accel, 0.f });

    // Torsional PD on hitbox to keep it upright
    float wrapped_angle = std::atan2(std::sin(hbox_angle), std::cos(hbox_angle));
    float omega         = pe.get_angular_velocity(hitbox_id);
    float target_omega  = -upright_stiffness * wrapped_angle;
    pe.set_angular_velocity(hitbox_id, omega + (target_omega - omega) * upright_damping * dt);

    // Snap torso to hitbox before the physics step.
    // AVBD then resolves limb joints on top of this corrected starting position.
    uint32_t torso_id = limbs[(size_t)Limb::Torso];
    pe.set_position(torso_id, hbox_pos + glm::vec2(0.0, -17.0));
    pe.set_velocity(torso_id, hbox_vel);
    pe.set_rotation(torso_id, hbox_angle);
    pe.set_angular_velocity(torso_id, pe.get_angular_velocity(hitbox_id));

    animate(pe, dt);
}

void PlayerCharacter::animate(PhysicsEngine& pe, float dt) {
    float hvel = pe.get_velocity(hitbox_id).x;
    if(glm::abs(hvel) > 0.0f){

        if (stride_counter < step_time) {
            if (right_airborne) {
                auto maybe_next_step = select_next_step(pe, true);
                if (maybe_next_step.has_value()) {
                    right_step_target = maybe_next_step.value();
                    right_airborne = false;
                }
            }

            left_step_target = position(pe) + airborn_foot_offset + 0.0f;
            left_airborne = true;
        }
        else if (stride_counter < step_time * 2.0f) {
            if (left_airborne) {
                auto maybe_next_step = select_next_step(pe, false);
                if (maybe_next_step.has_value()) {
                    left_step_target = maybe_next_step.value();
                    left_airborne = false;
                }
            }

            right_step_target = position(pe) + airborn_foot_offset + 5.0f;
            right_airborne = true;
        }
        
        
        stride_counter += dt * (glm::abs(hvel) / max_speed + 0.2f);
        if (stride_counter > step_time * 2.0f) stride_counter = 0.0f;

    }
    else{
        stride_counter = 0.0f;
    }

    float t = 1.f - std::exp(-12.f * dt);
    left_step_goal  = glm::mix(left_step_goal,  left_step_target,  t);
    right_step_goal = glm::mix(right_step_goal, right_step_target, t);

    // Hip anchors in world space (from torso, which carries the leg joints)
    glm::vec2 hip_l = position(pe)  + glm::rotate(glm::vec2(-8.f, 16.f), rotation(pe));
    glm::vec2 hip_r = position(pe)  + glm::rotate(glm::vec2(8.f, 16.f), rotation(pe));

    auto [hl, kl] = solve_leg_ik(hip_l, left_step_goal  + glm::vec2(0.f, left_airborne ? 0.0f : 16.f ), rotation(pe));
    auto [hr, kr] = solve_leg_ik(hip_r, right_step_goal + glm::vec2(0.f, right_airborne ? 0.0f : 16.f), rotation(pe));

    float max_d = max_leg_angle_speed * dt;
    auto approach = [&](float current, float target) {
        float diff = std::atan2(std::sin(target - current), std::cos(target - current));
        return current + std::clamp(diff, -max_d, max_d);
    };

    joints[(size_t)JointID::HipL].desired_angle  = approach(joints[(size_t)JointID::HipL].desired_angle,  hl);
    joints[(size_t)JointID::KneeL].desired_angle = approach(joints[(size_t)JointID::KneeL].desired_angle, kl);
    joints[(size_t)JointID::HipR].desired_angle  = approach(joints[(size_t)JointID::HipR].desired_angle,  hr);
    joints[(size_t)JointID::KneeR].desired_angle = approach(joints[(size_t)JointID::KneeR].desired_angle, kr);

    DebugDraw::get().point(left_step_goal,  4.f, 0x00FF00FF); // green = left
    DebugDraw::get().point(right_step_goal, 4.f, 0xFF0000FF); // red   = right
}

std::optional<glm::vec2> PlayerCharacter::select_next_step(PhysicsEngine& pe, bool right){

    float step_offset = right ? 5.0f : 0.0f;
    // TODO: make constant
    glm::vec2 origin = position(pe) + glm::vec2( step_offset + (velocity(pe).x / max_speed + glm::sign(velocity(pe).x) *0.1f) * step_length_test, 0.0f );

    //TODO: max dist needs to be constant too
    std::optional<RaycastHit> step_point = pe.raycast(origin, glm::vec2(0.0, 1.0), 80.0f, PLAYER_MASK);
    if(step_point.has_value()){
        glm::vec2 result = step_point.value().point;
        if(result.y > position(pe).y + 20.0f){
            return result;
        }
    }

    return std::optional<glm::vec2>();
}

glm::vec2 PlayerCharacter::position(PhysicsEngine& pe) const
{
    return pe.get_position(hitbox_id);
}

glm::vec2 PlayerCharacter::velocity(PhysicsEngine& pe) const
{
    return pe.get_velocity(hitbox_id);
}

float PlayerCharacter::rotation(PhysicsEngine& pe) const
{
    return pe.get_rotation(hitbox_id);
}

