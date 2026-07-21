#include "player_character.h"
#include <src/renderer/residua_engine.h>
#include <glm/gtx/rotate_vector.hpp>
#include <algorithm>

void PlayerCharacter::load_assets(const char* path)
{
    sprite = load_body_image(path);
}

void PlayerCharacter::spawn(PhysicsEngine& pe, ResiduaEngine& re, glm::vec2 pos)
{
    RigidBody rb;
    rb.sprite = sprite;
    rb.compute_mass_properties();
    rb.generate_shape();
    rb.generate_sdf();
    rb.position = glm::vec3(pos, 0.f);
    body_id = pe.add_body(&re, std::move(rb));
}

void PlayerCharacter::despawn(PhysicsEngine& pe)
{
    if (!is_valid()) return;
    pe.remove_body(body_id);
    body_id = INVALID;
}

void PlayerCharacter::reset_to(PhysicsEngine& pe, glm::vec2 pos)
{
    if (!is_valid()) return;
    pe.set_position        (body_id, pos);
    pe.set_rotation        (body_id, 0.f);
    pe.set_velocity        (body_id, { 0.f, 0.f });
    pe.set_angular_velocity(body_id, 0.f);
}

void PlayerCharacter::apply_inputs(float thrust, float rotate)
{
    thrust_input = std::clamp(thrust, 0.f, 1.f);
    rotate_input = std::clamp(rotate, -1.f, 1.f);
}

void PlayerCharacter::update(PhysicsEngine& pe, float dt)
{
    if (!is_valid()) return;

    pe.set_angular_velocity(body_id, rotate_input * rotate_speed);

    if (thrust_input > 0.f) {
        float angle = pe.get_rotation(body_id);
        glm::vec2 forward = glm::rotate(glm::vec2(0.f, -1.f), angle);
        pe.apply_force(body_id, forward * thrust_force * thrust_input);
    }
}

glm::vec2 PlayerCharacter::position(PhysicsEngine& pe) const { return pe.get_position(body_id); }
glm::vec2 PlayerCharacter::velocity(PhysicsEngine& pe) const { return pe.get_velocity(body_id); }
float     PlayerCharacter::rotation(PhysicsEngine& pe) const { return pe.get_rotation(body_id); }
