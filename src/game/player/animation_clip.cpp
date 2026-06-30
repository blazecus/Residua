#include "animation_clip.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>

using json = nlohmann::json;

// read user defined animations and sample frames with linear interpolation

bool AnimationClip::load(const char* path)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    json root;
    try { f >> root; } catch (...) { return false; }

    fps    = root.value("fps",    60);
    length = root.value("length", 120);
    keyframes.clear();

    for (const auto& kfd : root.value("keyframes", json::array())) {
        AnimKeyframe kf;
        kf.frame = kfd["frame"].get<int>();
        for (const auto& [jid, val] : kfd["joints"].items())
            kf.angles[jid] = val.get<float>();
        keyframes.push_back(std::move(kf));
    }
    std::sort(keyframes.begin(), keyframes.end(),
              [](const AnimKeyframe& a, const AnimKeyframe& b){ return a.frame < b.frame; });
    return true;
}

std::unordered_map<std::string, float> AnimationClip::sample(float frame) const
{
    if (keyframes.empty()) return {};
    if (frame <= keyframes.front().frame) return keyframes.front().angles;
    if (frame >= keyframes.back().frame)  return keyframes.back().angles;

    for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
        const auto& a = keyframes[i];
        const auto& b = keyframes[i + 1];
        if (frame < a.frame || frame > b.frame) continue;

        float span = float(b.frame - a.frame);
        float t    = span > 0.f ? (frame - float(a.frame)) / span : 0.f;

        std::unordered_map<std::string, float> result;
        for (const auto& [jid, va] : a.angles) {
            float vb   = b.angles.count(jid) ? b.angles.at(jid) : va;
            float diff = std::atan2(std::sin(vb - va), std::cos(vb - va));
            result[jid] = va + diff * t;
        }
        for (const auto& [jid, vb] : b.angles)
            if (!result.count(jid)) result[jid] = vb;
        return result;
    }
    return keyframes.back().angles;
}
