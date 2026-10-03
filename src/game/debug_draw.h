#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct DebugDraw {
    static DebugDraw& get();

    // rgba: 0xRRGGBBAA
    void point(glm::vec2 world_pos, float radius = 3.f, uint32_t rgba = 0xFF0000FF);
    void line(glm::vec2 from, glm::vec2 to, uint32_t rgba = 0xFF0000FF, float thickness = 1.f);
    void clear();
    void render(float scale_x, float scale_y, glm::vec2 camera_offset = { 0.f, 0.f });

private:
    struct Point { glm::vec2 pos; float radius; uint32_t rgba; };
    struct Line  { glm::vec2 from, to; uint32_t rgba; float thickness; };
    std::vector<Point> _points;
    std::vector<Line>  _lines;
};
