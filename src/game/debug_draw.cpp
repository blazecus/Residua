#include "debug_draw.h"
#include <imgui.h>

DebugDraw& DebugDraw::get() {
    static DebugDraw instance;
    return instance;
}

void DebugDraw::point(glm::vec2 world_pos, float radius, uint32_t rgba) {
    _points.push_back({ world_pos, radius, rgba });
}

void DebugDraw::line(glm::vec2 from, glm::vec2 to, uint32_t rgba, float thickness) {
    _lines.push_back({ from, to, rgba, thickness });
}

void DebugDraw::clear() {
    _points.clear();
    _lines.clear();
}

void DebugDraw::render(float scale_x, float scale_y, glm::vec2 camera_offset) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    auto to_col = [](uint32_t rgba) {
        return IM_COL32((rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF);
    };
    for (const auto& l : _lines) {
        glm::vec2 relA = l.from - camera_offset;
        glm::vec2 relB = l.to   - camera_offset;
        ImVec2 a = { relA.x * scale_x, relA.y * scale_y };
        ImVec2 b = { relB.x * scale_x, relB.y * scale_y };
        dl->AddLine(a, b, to_col(l.rgba), l.thickness);
    }
    for (const auto& p : _points) {
        glm::vec2 rel = p.pos - camera_offset;
        ImVec2 screen_pos = { rel.x * scale_x, rel.y * scale_y };
        dl->AddCircleFilled(screen_pos, p.radius * scale_x, to_col(p.rgba));
    }
}
