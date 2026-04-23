#pragma once

struct SceneManager;
struct PhysicsEngine;

struct UIManager {
    SceneManager*  scenes  { nullptr };
    PhysicsEngine* physics { nullptr };

    void init(SceneManager* sc, PhysicsEngine* phys);

    // Call between ImGui::NewFrame() and ImGui::Render().
    void draw();
};
