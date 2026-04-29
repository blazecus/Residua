#pragma once

struct SceneManager;
class PhysicsEngine;
class ResiduaEngine;

struct UIManager {
    SceneManager*  scenes  { nullptr };
    PhysicsEngine* physics { nullptr };
    ResiduaEngine* engine  { nullptr };

    void init(SceneManager* sc, PhysicsEngine* phys, ResiduaEngine* eng);

    // Call between ImGui::NewFrame() and ImGui::Render().
    void draw();
};
