#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

static constexpr uint32_t STATE_HISTORY_LEN = 10;
static constexpr uint32_t JOINT_COUNT       = 13; 

struct MovementState {
    float     move_dir  { 0.f };
    bool      walking   { false };
    bool      jump      { false };
    glm::vec2 aim_pos   {};

    glm::vec2 torso_velocity         {};
    float     torso_angle            { 0.f };
    float     torso_angular_velocity { 0.f };

    std::array<float, JOINT_COUNT> joint_angles{};

    // Environment
    bool      grounded         { false };
    bool      wall_ahead       { false };
    float     ground_dist      { 9999.f }; // downward raycast distance, 9999 if no hit
    glm::vec2 torso_pos        {};
    glm::vec2 left_shoulder_pos{};         
    float     left_arm_angle   { 0.f };   
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
