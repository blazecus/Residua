#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

std::vector<glm::vec2> marching_squares(
    uint32_t                      width,
    uint32_t                      height,
    const std::vector<glm::vec4>& img,
    float                         threshold = 0.5f
);

std::vector<glm::vec2> douglas_peucker(
    const std::vector<glm::vec2>& contour,
    float                         epsilon
);

std::vector<glm::vec2> generate_shape(
    uint32_t                  width,
    uint32_t                  height,
    const std::vector<glm::vec4>& img,
    float                     threshold = 0.5f,
    float                     epsilon   = 0.5f
);