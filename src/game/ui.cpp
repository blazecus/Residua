#include "ui.h"
#include "scenes.h"
#include "imgui.h"

void UIManager::init(SceneManager* sc, PhysicsEngine* phys) {
    scenes = sc; physics = phys;
}

void UIManager::draw() {
    ImGui::Begin("Scene");
    for (int i = 0; i < NUM_SCENES; i++) {
        bool active = (scenes->current_scene == i);
        if (active) ImGui::BeginDisabled();
        if (ImGui::Button(SCENE_LIST[i].name)) scenes->load(i);
        if (active) ImGui::EndDisabled();
        if (i + 1 < NUM_SCENES) ImGui::SameLine();
    }
    ImGui::End();
}
