#include "py_env.h"
#include "player/player_character.h"
#include "player/movement_state.h"
#include <SDL.h>
#include <cmath>

void PyEnv::init(float spawn_x, float spawn_y, bool render, int scene_idx)
{
    spawn_x_ = spawn_x;
    spawn_y_ = spawn_y;

    game.init();

    if (scene_idx != 3)
        game.scene_manager.load(scene_idx);

    // Respawn player at the requested position (scene may have placed it elsewhere).
    reset(spawn_x, spawn_y);

    if (!render)
        SDL_HideWindow(game._window);

    initialized = true;
}

std::tuple<std::vector<float>, float, bool>
PyEnv::step(const std::vector<float>& torques, glm::vec2 aim_pos, bool render)
{
    // Keep the window alive and handle resize / close events.
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT)
            initialized = false;
        game.engine.process_renderer_input(e);
    }

    auto& player  = game.scene_manager.player;
    auto& physics = game.physics_engine;

    if (player.is_valid()) {
        PlayerCharacter::Action action;
        int n = std::min((int)torques.size(), (int)Limb::Count);
        for (int i = 0; i < n; ++i)
            action.torques[i] = torques[i];
        player.apply_action(physics, action);
    }

    player.apply_inputs(0.f, false, false, aim_pos);
    game.scene_manager.update(game.delta);

    if (render) {
        SDL_ShowWindow(game._window);
        game.engine._dt = game.delta;
        game.engine.draw_frame();
    } else {
        SDL_HideWindow(game._window);
        physics_only_step(game.delta);
    }

    auto  obs    = get_obs();
    float reward = player.compute_reward(game.delta);
    bool  done   = is_done() || !initialized;

    return { obs, reward, done };
}

void PyEnv::physics_only_step(float dt)
{
    game.physics_engine.camera_offset              = game.scene_manager.camera_offset;
    game.physics_engine.particle_sim.camera_offset = game.scene_manager.camera_offset;
    game.physics_engine.step_physics(dt);
}

void PyEnv::reset(float spawn_x, float spawn_y)
{
    spawn_x_ = spawn_x;
    spawn_y_ = spawn_y;

    auto& player  = game.scene_manager.player;
    auto& physics = game.physics_engine;

    player.despawn(physics);
    player.spawn(physics, game.engine, { spawn_x, spawn_y });

    for (uint32_t id : player.limbs) {
        if (id != PlayerCharacter::INVALID) {
            physics.set_velocity        (id, { 0.f, 0.f });
            physics.set_angular_velocity(id, 0.f);
        }
    }
}

void PyEnv::close()
{
    if (initialized) {
        game.end();
        initialized = false;
    }
}

std::vector<float> PyEnv::get_obs() const
{
    std::vector<float> out(OBS_SIZE, 0.f);
    const auto& player = game.scene_manager.player;
    if (!player.is_valid() || player.state_history.count == 0)
        return out;

    const MovementState& st = player.state_history.at(0);
    out[0]  = st.move_dir;
    out[1]  = st.torso_angle;
    out[2]  = st.torso_angular_velocity;
    out[3]  = st.grounded   ? 1.f : 0.f;
    out[4]  = st.wall_ahead ? 1.f : 0.f;
    out[5]  = st.ground_dist;
    out[6]  = st.torso_velocity.x;
    out[7]  = st.torso_velocity.y;
    out[8]  = st.torso_pos.x;
    out[9]  = st.torso_pos.y;
    out[10] = st.aim_pos.x;
    out[11] = st.aim_pos.y;
    out[12] = st.left_shoulder_pos.x;
    out[13] = st.left_shoulder_pos.y;
    out[14] = st.left_arm_angle;
    for (int i = 0; i < (int)JOINT_COUNT; ++i)
        out[15 + i] = st.joint_angles[i];
    return out;
}

bool PyEnv::is_done() const
{
    const auto& player = game.scene_manager.player;
    if (!player.is_valid()) return true;

    float angle = game.physics_engine.get_rotation(player.limb_id(Limb::Torso));
    // Normalise to [-π, π]
    angle = std::atan2(std::sin(angle), std::cos(angle));
    if (std::abs(angle) > FALL_ANGLE_LIMIT) return true;

    // Fell below the world
    glm::vec2 pos = game.physics_engine.get_position(player.limb_id(Limb::Torso));
    if (pos.y > 400.f) return true;

    return false;
}
