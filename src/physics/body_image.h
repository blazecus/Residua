#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct LoadedBodyImage {
    uint32_t               width{}, height{};
    std::vector<glm::vec4> pixels;
};

LoadedBodyImage load_body_image(const char* path);
LoadedBodyImage flip_horizontal(const LoadedBodyImage& img);
