#pragma once

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <vector>
#include <cstdint>

struct PixelComponent {
    std::vector<glm::ivec2> pixels;  // (lx, ly) coords
    int x0, y0, x1, y1;             // bounding box (inclusive)
};

// Find all 8-connected components of solid pixels (alpha > 0) in a w×h bitmap.
// If ex0..ey1 describes the bounding box of recently erased pixels, an optimized
// seed-based search is used: BFS starts only from solid pixels on the 1-pixel
// ring around that region.  For the no-split case this returns a single
// PixelComponent whose pixels list is EMPTY (bbox only) — fracture_dirty_bodies
// can skip the pixel enumeration entirely.  When a split is detected the
// function falls back to the full scan with complete pixel lists.
std::vector<PixelComponent> find_components(
    const glm::vec4* pixels, uint32_t w, uint32_t h,
    int ex0 = -1, int ey0 = -1, int ex1 = -1, int ey1 = -1);
