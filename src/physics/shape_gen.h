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

// Stitch unordered marching-squares segment pairs into ordered closed contours.
// Input is the flat a,b,a,b,... list returned by marching_squares.
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

struct Triangle {
    uint32_t a, b, c;
};

std::vector<Triangle> triangulate(const std::vector<glm::vec2>& polygon);