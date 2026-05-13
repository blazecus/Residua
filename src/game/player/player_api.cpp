#include "player_api.h"
#include "player_character.h"
#include <src/physics/physics_engine.h>
#include <algorithm>

static PlayerCharacter* s_character = nullptr;
static PhysicsEngine*   s_engine    = nullptr;

int residua_limb_count()
{
    return (int)Limb::Count;
}

void residua_register(void* player_character, void* physics_engine)
{
    s_character = static_cast<PlayerCharacter*>(player_character);
    s_engine    = static_cast<PhysicsEngine*>(physics_engine);
}

void residua_set_action(const float* torques, int count)
{
    if (!s_character || !s_engine) return;

    PlayerCharacter::Action action;
    int n = std::min(count, residua_limb_count());
    for (int i = 0; i < n; ++i)
        action.torques[i] = torques[i];

    s_character->apply_action(*s_engine, action);
}

float residua_get_reward(float dt)
{
    if (!s_character) return 0.f;
    return s_character->compute_reward(dt);
}
