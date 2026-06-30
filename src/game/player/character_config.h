#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <glm/vec2.hpp>

struct JointConfig {
    std::string parent_limb;
    std::string child_limb;
    glm::vec2   attach_parent{};
    glm::vec2   attach_child{};
    bool        animatable    = true;
    float       default_angle = 0.f;
    float       angle_min     = -3.14159f;
    float       angle_max     =  3.14159f;
    float       max_torque    = std::numeric_limits<float>::infinity();
};

struct CharacterConfig {
    std::string name;

    // limb_id → sprite path (relative to project root)
    std::unordered_map<std::string, std::string>  limb_sprites;
    // limb_id → parent limb_id  (empty string = root limb)
    std::unordered_map<std::string, std::string>  limb_parents;
    // limb_id → density (only present if specified; absent = normalize to 0.1)
    std::unordered_map<std::string, float>        limb_densities;
    // joint_id → config
    std::unordered_map<std::string, JointConfig>  joints;
    // child limb_id → joint_id  (reverse lookup)
    std::unordered_map<std::string, std::string>  joint_for_child;
    // rendering order (list of limb ids)
    std::vector<std::string>                      draw_order;

    bool load(const char* path);

    std::unordered_map<std::string, glm::vec2> compute_spawn_offsets() const;
};
