#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "body_image.h"

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

std::vector<std::vector<glm::vec2>> stitch_contours(
    const std::vector<glm::vec2>& segments
);

std::vector<glm::vec2> generate_shape(
    uint32_t                  width,
    uint32_t                  height,
    const std::vector<glm::vec4>& img,
    float                     threshold = 0.5f,
    float                     epsilon   = 0.5f
);

std::vector<float> generate_sdf(const LoadedBodyImage& img,
                                 const std::vector<glm::vec2>& shape_local);