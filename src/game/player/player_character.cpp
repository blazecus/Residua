#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <limits>

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

    // Torso (32x32)
    limbs[(size_t)Limb::Torso]     = spawn_limb(pe, re, img_torso,     pos);

    // Head: sits above torso (torso top = -16, head bottom = +8)
    limbs[(size_t)Limb::Head]      = spawn_limb(pe, re, img_head,      pos + glm::vec2(  0, -24));

    // Arms (all limbs 16x16, top anchor = -8, bottom anchor = +8)
    limbs[(size_t)Limb::UpperArmL] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2(-16,   0));
    limbs[(size_t)Limb::ForearmL]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2(-16,  16));
    limbs[(size_t)Limb::HandL]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2(-16,  32));
    limbs[(size_t)Limb::UpperArmR] = spawn_limb(pe, re, img_upper_arm, pos + glm::vec2( 16,   0));
    limbs[(size_t)Limb::ForearmR]  = spawn_limb(pe, re, img_forearm,   pos + glm::vec2( 16,  16));
    limbs[(size_t)Limb::HandR]     = spawn_limb(pe, re, img_hand,      pos + glm::vec2( 16,  32));

    // Legs
    limbs[(size_t)Limb::ThighL]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2( -8,  24));
    limbs[(size_t)Limb::LowerLegL] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2( -8,  40));
    limbs[(size_t)Limb::FootL]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2( -8,  56));
    limbs[(size_t)Limb::ThighR]    = spawn_limb(pe, re, img_thigh,     pos + glm::vec2(  8,  24));
    limbs[(size_t)Limb::LowerLegR] = spawn_limb(pe, re, img_lower_leg, pos + glm::vec2(  8,  40));
    limbs[(size_t)Limb::FootR]     = spawn_limb(pe, re, img_foot,      pos + glm::vec2(  8,  56));

    const float INF  = std::numeric_limits<float>::infinity();
    const float BEND = 10.f;

    // Head
    connect(pe, JointID::Neck,      Limb::Torso,     {  0, -16}, Limb::Head,      {  0,  8}, INF, BEND);
    // Left arm
    connect(pe, JointID::ShoulderL, Limb::Torso,     {-16,  -8}, Limb::UpperArmL, {  0, -8}, INF, BEND);
    connect(pe, JointID::ElbowL,    Limb::UpperArmL, {  0,   8}, Limb::ForearmL,  {  0, -8}, INF, BEND);
    connect(pe, JointID::WristL,    Limb::ForearmL,  {  0,   8}, Limb::HandL,     {  0, -8}, INF, BEND);
    // Right arm
    connect(pe, JointID::ShoulderR, Limb::Torso,     { 16,  -8}, Limb::UpperArmR, {  0, -8}, INF, BEND);
    connect(pe, JointID::ElbowR,    Limb::UpperArmR, {  0,   8}, Limb::ForearmR,  {  0, -8}, INF, BEND);
    connect(pe, JointID::WristR,    Limb::ForearmR,  {  0,   8}, Limb::HandR,     {  0, -8}, INF, BEND);
    // Left leg
    connect(pe, JointID::HipL,      Limb::Torso,     { -8,  16}, Limb::ThighL,    {  0, -8}, INF, BEND);
    connect(pe, JointID::KneeL,     Limb::ThighL,    {  0,   8}, Limb::LowerLegL, {  0, -8}, INF, BEND);
    connect(pe, JointID::AnkleL,    Limb::LowerLegL, {  0,   8}, Limb::FootL,     {  0, -8}, INF, BEND);
    // Right leg
    connect(pe, JointID::HipR,      Limb::Torso,     {  8,  16}, Limb::ThighR,    {  0, -8}, INF, BEND);
    connect(pe, JointID::KneeR,     Limb::ThighR,    {  0,   8}, Limb::LowerLegR, {  0, -8}, INF, BEND);
    connect(pe, JointID::AnkleR,    Limb::LowerLegR, {  0,   8}, Limb::FootR,     {  0, -8}, INF, BEND);
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

    for (uint32_t id : limbs)
        if (id != INVALID) pe.remove_body(id);

    limbs.fill(INVALID);
}

void PlayerCharacter::set_desired_angle(JointID jid, float angle)
{
    joints[(size_t)jid].desired_angle = angle;
}

void PlayerCharacter::update(PhysicsEngine& /*pe*/, float /*dt*/)
{
    for (auto& pj : joints)
        if (pj.joint) pj.joint->rest_angle = pj.desired_angle;
}

glm::vec2 PlayerCharacter::position(PhysicsEngine& pe) const
{
    return pe.get_position(limbs[(size_t)Limb::Torso]);
}
