#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

static constexpr uint32_t STATE_HISTORY_LEN = 1;
static constexpr uint32_t JOINT_COUNT       = 13;
static constexpr uint32_t LIMB_COUNT        = 14;

struct MovementState {
    float     move_dir  { 0.f };
    bool      walking   { false };
    bool      jump      { false };
    glm::vec2 aim_pos   {};

    glm::vec2 torso_velocity         {};
    float     torso_angle            { 0.f };
    float     torso_angular_velocity { 0.f };

    std::array<float, JOINT_COUNT> joint_angles{};
    std::array<float, JOINT_COUNT> joint_angular_velocities{};

    // Environment
    bool      grounded         { false };
    bool      wall_ahead       { false };
    float     ground_dist      { 9999.f };
    glm::vec2 torso_pos        {};
    glm::vec2 left_shoulder_pos{};
    float     left_arm_angle   { 0.f };
    float     facing_dir       { 1.f };
    float     feet_mid_x       { 0.f };
    float     feet_mid_y       { 0.f };

    std::array<glm::vec2, LIMB_COUNT> limb_velocities{};

    // Animation goals for the current and next frame of the active clip.
    std::array<float, JOINT_COUNT> anim_goal_current{};
    std::array<float, JOINT_COUNT> anim_goal_next{};
};

struct MovementStateBuffer {
    std::array<MovementState, STATE_HISTORY_LEN> states{};
    uint32_t head  { 0 };
    uint32_t count { 0 };

    void push(const MovementState& s);

    // 0 = most recent frame, 1 = one frame back, etc. Clamps to available count.
    const MovementState& at(uint32_t steps_back) const;

    bool full() const { return count == STATE_HISTORY_LEN; }
};
