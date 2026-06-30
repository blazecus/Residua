#include "character_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <queue>

using json = nlohmann::json;

// load characters from json config
// this contains joint info and weights

bool CharacterConfig::load(const char* path)
{
    std::ifstream f(path);
    if (!f.is_open())
        return false;

    json root;
    try { f >> root; } catch (...) { return false; }

    name = root.value("name", "");

    limb_sprites.clear();
    limb_parents.clear();
    joints.clear();
    joint_for_child.clear();
    draw_order.clear();

    for (const auto& limb : root["limbs"]) {
        std::string id   = limb["id"];
        limb_sprites[id] = limb["sprite"].get<std::string>();
        if (limb.contains("parent") && !limb["parent"].is_null())
            limb_parents[id] = limb["parent"].get<std::string>();
        else
            limb_parents[id] = "";
        if (limb.contains("density"))
            limb_densities[id] = limb["density"].get<float>();
    }

    for (const auto& joint : root["joints"]) {
        std::string id = joint["id"];
        JointConfig jc;
        jc.parent_limb  = joint["parent_limb"].get<std::string>();
        jc.child_limb   = joint["child_limb"].get<std::string>();
        auto ap         = joint["attach_parent"];
        jc.attach_parent = { ap[0].get<float>(), ap[1].get<float>() };
        auto ac         = joint["attach_child"];
        jc.attach_child  = { ac[0].get<float>(), ac[1].get<float>() };
        jc.animatable    = joint.value("animatable",  true);
        jc.default_angle = joint.value("default",    0.f);
        jc.angle_min     = joint.value("angle_min", -3.14159f);
        jc.angle_max     = joint.value("angle_max",  3.14159f);
        jc.max_torque    = joint.contains("max_torque")
                           ? joint["max_torque"].get<float>()
                           : std::numeric_limits<float>::infinity();
        joints[id]                     = jc;
        joint_for_child[jc.child_limb] = id;
    }

    for (const auto& id : root.value("draw_order", json::array()))
        draw_order.push_back(id.get<std::string>());

    return true;
}

std::unordered_map<std::string, glm::vec2> CharacterConfig::compute_spawn_offsets() const
{
    std::unordered_map<std::string, glm::vec2> offsets;

    std::string root_id;
    for (const auto& [id, parent] : limb_parents)
        if (parent.empty()) { root_id = id; break; }
    if (root_id.empty()) return offsets;

    offsets[root_id] = { 0.f, 0.f };

    std::queue<std::string> q;
    q.push(root_id);
    while (!q.empty()) {
        auto pid = q.front(); q.pop();
        for (const auto& [lid, parent_id] : limb_parents) {
            if (parent_id != pid) continue;
            auto jt = joint_for_child.find(lid);
            if (jt == joint_for_child.end()) continue;
            const JointConfig& jc = joints.at(jt->second);
            // At angle=0: child_CoM = parent_CoM + attach_parent - attach_child
            offsets[lid] = offsets.at(pid) + jc.attach_parent - jc.attach_child;
            q.push(lid);
        }
    }
    return offsets;
}
