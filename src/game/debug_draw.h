#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct DebugDraw {
    static DebugDraw& get();

    // rgba: 0xRRGGBBAA
    void point(glm::vec2 world_pos, float radius = 3.f, uint32_t rgba = 0xFF0000FF);
    void clear();
    void render(float scale_x, float scale_y);

private:
    struct Point { glm::vec2 pos; float radius; uint32_t rgba; };
    std::vector<Point> _points;
};
