#include "debug_draw.h"
#include <imgui.h>

DebugDraw& DebugDraw::get() {
    static DebugDraw instance;
    return instance;
}

void DebugDraw::point(glm::vec2 world_pos, float radius, uint32_t rgba) {
    _points.push_back({ world_pos, radius, rgba });
}

void DebugDraw::clear() {
    _points.clear();
}

void DebugDraw::render(float scale_x, float scale_y) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    for (const auto& p : _points) {
        ImVec2 screen_pos = { p.pos.x * scale_x, p.pos.y * scale_y };
        ImU32 col = IM_COL32((p.rgba >> 24) & 0xFF,
                             (p.rgba >> 16) & 0xFF,
                             (p.rgba >>  8) & 0xFF,
                              p.rgba        & 0xFF);
        dl->AddCircleFilled(screen_pos, p.radius * scale_x, col);
    }
}
