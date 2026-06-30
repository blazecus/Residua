#pragma once

#include "game.h"
#include "player/player_character.h"
#include <vector>
#include <tuple>

struct AgentSlot {
    PlayerCharacter character;
    glm::vec2       spawn_pos    { 320.f, 170.f };
    float           fallen_timer { 0.f };
};

struct PyEnv {
    static constexpr int FRAME_OBS_SIZE = 16 + (int)JOINT_COUNT + 2 * (int)LIMB_COUNT;
    static constexpr int OBS_SIZE       = FRAME_OBS_SIZE + 4 * (int)JOINT_COUNT;

    static constexpr float FALL_ANGLE_LIMIT = 1.4f;   
    static constexpr float FALLEN_TIMEOUT   = 3.f;   
    static constexpr float MAX_LINEAR_VEL   = 10000.f;
    static constexpr float MAX_ANGULAR_VEL  = 80.f * 6.2832f;

    static constexpr float AGENT_SPACING = 80.f;

    Game game;
    bool initialized { false };

    std::vector<AgentSlot> agents_;

    void init(int n_agents, float spawn_x, float spawn_y,
              bool render = true, int scene_idx = 5);

    // Advance one simulation step for all agents.
    std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>>
    step(const std::vector<std::vector<float>>& all_angles,
         const std::vector<float>& move_dirs,
         const std::vector<bool>&  jumps,
         const std::vector<float>& aim_xs,
         const std::vector<float>& aim_ys,
         bool sim_inputs = false, bool render = true);

    void reset_agent(int i, float spawn_x = -1.f, float spawn_y = -1.f);
    void reset_all  (float spawn_x = -1.f, float spawn_y = -1.f);

    std::vector<float>              get_agent_obs(int i) const;
    std::vector<std::vector<float>> get_obs_all()        const;

    int  n_agents() const { return (int)agents_.size(); }
    void close();

private:
    bool is_done(int i);
    void physics_step(bool render);
    void update_camera();
};
