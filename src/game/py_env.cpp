#include "py_env.h"
#include "player/player_character.h"
#include "player/movement_state.h"
#include <SDL.h>
#include <cmath>
#include <algorithm>

void PyEnv::init(int n_agents, float spawn_x, float spawn_y, bool render, int scene_idx)
{
    game.init();

    if (scene_idx != 3)
        game.scene_manager.load(scene_idx);

    agents_.resize(n_agents);
    auto& src = game.scene_manager.player;

    for (int i = 0; i < n_agents; ++i) {
        auto& slot = agents_[i];
        auto& ch   = slot.character;

        ch.char_config       = src.char_config;
        ch.limb_images       = src.limb_images;
        ch.joint_animatable  = src.joint_animatable;
        ch.joint_angle_min   = src.joint_angle_min;
        ch.joint_angle_max   = src.joint_angle_max;
        ch.anim_standing     = src.anim_standing;

        slot.spawn_pos = { spawn_x + i * AGENT_SPACING, spawn_y };
        ch.spawn(game.physics_engine, game.engine, slot.spawn_pos);

        for (uint32_t id : ch.limbs) {
            if (id != PlayerCharacter::INVALID) {
                game.physics_engine.set_velocity        (id, { 0.f, 0.f });
                game.physics_engine.set_angular_velocity(id, 0.f);
            }
        }
    }

    if (!render)
        SDL_HideWindow(game._window);

    initialized = true;
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>>
PyEnv::step(const std::vector<std::vector<float>>& all_angles,
            const std::vector<float>& move_dirs,
            const std::vector<bool>&  jumps,
            const std::vector<float>& aim_xs,
            const std::vector<float>& aim_ys,
            bool sim_inputs, bool render)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) initialized = false;
        game.engine.process_renderer_input(e);
    }

    int n = (int)agents_.size();
    auto& physics = game.physics_engine;

    auto getf = [](const std::vector<float>& v, int i, float def) {
        return i < (int)v.size() ? v[i] : def;
    };
    auto getb = [](const std::vector<bool>& v, int i, bool def) {
        return i < (int)v.size() ? v[i] : def;
    };

    for (int i = 0; i < n; ++i) {
        auto& ch = agents_[i].character;
        if (!ch.is_valid()) continue;

        if (i < (int)all_angles.size()) {
            PlayerCharacter::Action action;
            const auto& ang = all_angles[i];
            int cnt = std::min((int)ang.size(), (int)Joint::Count);
            for (int j = 0; j < cnt; ++j)
                action.goal_angle_change[j] = ang[j];
            ch.apply_action(physics, action);
        }

        ch.apply_inputs(
            getf(move_dirs, i, 0.f),
            false,
            getb(jumps, i, false),
            { getf(aim_xs, i, agents_[i].spawn_pos.x + 100.f),
              getf(aim_ys, i, agents_[i].spawn_pos.y) }
        );

        ch.animate_left_arm(physics);
    }

    for (int i = 0; i < n; ++i)
        agents_[i].character.update(physics, game.delta, sim_inputs);

    update_camera();
    physics_step(render);

    std::vector<std::vector<float>> obs_all(n);
    std::vector<float>              rewards(n);
    std::vector<bool>               dones(n);

    for (int i = 0; i < n; ++i) {
        obs_all[i] = get_agent_obs(i);
        rewards[i] = agents_[i].character.compute_reward(game.delta);
        dones[i]   = is_done(i) || !initialized;
    }

    return { obs_all, rewards, dones };
}

void PyEnv::reset_agent(int i, float spawn_x, float spawn_y)
{
    auto& slot = agents_[i];
    if (spawn_x >= 0.f) slot.spawn_pos.x = spawn_x;
    if (spawn_y >= 0.f) slot.spawn_pos.y = spawn_y;
    slot.fallen_timer = 0.f;
    slot.character.reset_to(game.physics_engine, slot.spawn_pos);
}

void PyEnv::reset_all(float spawn_x, float spawn_y)
{
    for (int i = 0; i < (int)agents_.size(); ++i)
        reset_agent(i, spawn_x, spawn_y);
}

std::vector<float> PyEnv::get_agent_obs(int i) const
{
    std::vector<float> out(OBS_SIZE, 0.f);
    const auto& ch = agents_[i].character;
    if (!ch.is_valid() || ch.state_history.count == 0)
        return out;

    const MovementState& st = ch.state_history.at(0);
    float* o = out.data();

    // 16 scalars — positions relative to torso
    glm::vec2 tp = st.torso_pos;
    o[0]  = st.move_dir;
    o[1]  = st.jump        ? 1.f : 0.f;
    o[2]  = st.torso_angle;
    o[3]  = st.torso_angular_velocity;
    o[4]  = st.grounded    ? 1.f : 0.f;
    o[5]  = st.wall_ahead  ? 1.f : 0.f;
    o[6]  = st.ground_dist;
    o[7]  = st.torso_velocity.x;
    o[8]  = st.torso_velocity.y;
    o[9]  = st.aim_pos.x          - tp.x;
    o[10] = st.aim_pos.y          - tp.y;
    o[11] = st.left_shoulder_pos.x - tp.x;
    o[12] = st.left_shoulder_pos.y - tp.y;
    o[13] = st.left_arm_angle;
    o[14] = st.facing_dir;
    o[15] = st.feet_mid_y         - tp.y;

    // Joint angles
    for (int j = 0; j < (int)JOINT_COUNT; ++j)
        o[16 + j] = st.joint_angles[j];

    // Limb linear velocities (x,y per limb, Limb enum order)
    float* p = o + 16 + (int)JOINT_COUNT;
    for (int j = 0; j < (int)LIMB_COUNT; ++j) {
        p[j * 2 + 0] = st.limb_velocities[j].x;
        p[j * 2 + 1] = st.limb_velocities[j].y;
    }

    // Joint angular velocities
    p = o + FRAME_OBS_SIZE;
    for (int j = 0; j < (int)JOINT_COUNT; ++j)
        p[j] = st.joint_angular_velocities[j];

    // Current action goal angles
    p += JOINT_COUNT;
    for (int j = 0; j < (int)JOINT_COUNT; ++j)
        p[j] = ch.current_action.goal_angle_change[j];

    // Current animation frame goals
    p += JOINT_COUNT;
    for (int j = 0; j < (int)JOINT_COUNT; ++j)
        p[j] = st.anim_goal_current[j];

    // Next animation frame goals
    p += JOINT_COUNT;
    for (int j = 0; j < (int)JOINT_COUNT; ++j)
        p[j] = st.anim_goal_next[j];

    return out;
}

std::vector<std::vector<float>> PyEnv::get_obs_all() const
{
    std::vector<std::vector<float>> result;
    result.reserve(agents_.size());
    for (int i = 0; i < (int)agents_.size(); ++i)
        result.push_back(get_agent_obs(i));
    return result;
}

void PyEnv::close()
{
    if (initialized) {
        game.end();
        initialized = false;
    }
}

bool PyEnv::is_done(int i)
{
    auto& slot  = agents_[i];
    auto& ch    = slot.character;
    auto& phys  = game.physics_engine;

    if (!ch.is_valid()) return true;

    glm::vec2 pos = phys.get_position(ch.limb_id(Limb::Torso));
    if (pos.y > 400.f) return true;

    float angle = phys.get_rotation(ch.limb_id(Limb::Torso));
    angle = std::atan2(std::sin(angle), std::cos(angle));
    if (std::abs(angle) > FALL_ANGLE_LIMIT)
        slot.fallen_timer += game.delta;
    else
        slot.fallen_timer = 0.f;

    return slot.fallen_timer >= FALLEN_TIMEOUT;
}

void PyEnv::physics_step(bool render)
{
    game.physics_engine.camera_offset              = game.scene_manager.camera_offset;
    game.physics_engine.particle_sim.camera_offset = game.scene_manager.camera_offset;

    if (render) {
        SDL_ShowWindow(game._window);
        game.engine._dt = game.delta;
        game.engine.draw_frame();
    } else {
        SDL_HideWindow(game._window);
        game.physics_engine.step_physics(game.delta);
    }
}

void PyEnv::update_camera()
{
    if (agents_.empty()) return;
    auto& ch = agents_[0].character;
    if (!ch.is_valid()) return;

    glm::vec2 ppos = ch.position(game.physics_engine);
    game.scene_manager.camera_offset.x = ppos.x - float(PHYSICS_WIDTH)  * 0.5f;
    game.scene_manager.camera_offset.y = ppos.y - float(PHYSICS_HEIGHT) * 0.5f;
}
