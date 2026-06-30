#pragma once

int   residua_limb_count();

void  residua_register(void* player_character, void* physics_engine);
void  residua_set_action(const float* angle_changes, int count);
float residua_get_reward(float dt);
