#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct AnimKeyframe {
    int   frame = 0;
    std::unordered_map<std::string, float> angles; 
};

struct AnimationClip {
    int fps    = 60;
    int length = 120;
    std::vector<AnimKeyframe> keyframes; 

    bool load(const char* path);

    std::unordered_map<std::string, float> sample(float frame) const;

    bool empty() const { return keyframes.empty(); }
};
