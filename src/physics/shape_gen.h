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

static constexpr uint32_t SDF_SCALE = 2;

std::vector<float> generate_sdf(const LoadedBodyImage& img,
                                 const std::vector<glm::vec2>& shape_local,
                                 glm::vec2 com_local,
                                 uint32_t scale = SDF_SCALE,
                                 glm::vec2* out_thickest_px = nullptr // used in fracturing logic
                                );

// Recompute only the SDF cells in [px0..px1, py0..py1] (in SDF pixel coords).
void patch_sdf_region(const LoadedBodyImage& img,
                      const std::vector<glm::vec2>& shape_local,
                      glm::vec2 com_local,
                      uint32_t scale,
                      uint32_t px0, uint32_t py0,
                      uint32_t px1, uint32_t py1,
                      std::vector<float>& sdf);