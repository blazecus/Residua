#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

// Run marching squares on an alpha mask to extract closed contours.
// Output: one vector of 2D vertices per contour, in pixel-grid coordinates
// where (0,0) is the top-left corner and each unit = one pixel.
// threshold: alpha values strictly above this are considered solid.
std::vector<glm::vec2> marching_squares(
    uint32_t           width,
    uint32_t           height,
    const std::vector<float>& alpha,
    float              threshold = 0.5f
);

// Simplify a closed polygon using the Douglas-Peucker algorithm.
// epsilon: maximum allowed perpendicular deviation in the same units as the input.
// Returns a simplified closed polygon (no duplicate closing vertex).
std::vector<glm::vec2> douglas_peucker(
    const std::vector<glm::vec2>& contour,
    float                         epsilon
);

// Extract the dominant collision shape from an alpha mask.
// Runs marching squares to find all contours, picks the largest by perimeter,
// then simplifies it with Douglas-Peucker.
std::vector<glm::vec2> generate_shape(
    uint32_t                  width,
    uint32_t                  height,
    const std::vector<float>& alpha,
    float                     threshold = 0.5f,
    float                     epsilon   = 0.5f
);