#pragma once

#include "game.h"
#include <vector>
#include <tuple>

struct PyEnv {
    static constexpr int   OBS_SIZE         = 28;
    // TODO: don't use this
    static constexpr float FALL_ANGLE_LIMIT = 1.4f;  

    Game  game;
    bool  initialized { false };
    float spawn_x_    { 320.f };
    float spawn_y_    { 170.f };

    // Initialise SDL + Vulkan + physics, load a scene, and spawn the player.
    void init(float spawn_x, float spawn_y, bool render = true, int scene_idx = 5);

    // Advance one simulation step.
    std::tuple<std::vector<float>, float, bool>
    step(const std::vector<float>& torques, glm::vec2 aim_pos, bool render = true);

    void reset(float spawn_x, float spawn_y);

    void close();

private:
    std::vector<float> get_obs()  const;
    bool               is_done()  const;
    void               physics_only_step(float dt);
};
