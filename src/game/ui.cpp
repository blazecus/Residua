#include "ui.h"
#include "scenes.h"
#include "debug_draw.h"
#include "imgui.h"
#include <src/physics/physics_engine.h>
#include <src/renderer/residua_engine.h>

void UIManager::init(SceneManager* sc, PhysicsEngine* phys, ResiduaEngine* eng) {
    scenes = sc; physics = phys; engine = eng;
}

void UIManager::draw() {
    PhysicsWorld& w = physics->world;

    ImGui::Begin("Physics");

    ImGui::SeparatorText("Scene");
    for (int i = 0; i < NUM_SCENES; i++) {
        bool active = (scenes->current_scene == i);
        if (active) ImGui::BeginDisabled();
        if (ImGui::Button(SCENE_LIST[i].name)) scenes->load(i);
        if (active) ImGui::EndDisabled();
        if (i + 1 < NUM_SCENES) ImGui::SameLine();
    }

    ImGui::SeparatorText("World");
    ImGui::SliderFloat("Gravity",    &w.gravity,    0.f,   500.f);
    ImGui::SliderInt  ("Iterations", &w.iterations, 1,     30);
    ImGui::Checkbox   ("Post-stabilize", &w.postStabilize);

    ImGui::SeparatorText("Solver");
    ImGui::SliderFloat("Alpha", &w.alpha, 0.f, 1.f);
    ImGui::SliderFloat("Beta",  &w.beta,  0.f, 2000000.f);
    ImGui::SliderFloat("Gamma", &w.gamma, 0.f, 1.f);

    ImGui::SeparatorText("Stats");
    const PhysicsStats& s = w.stats;
    ImGui::Text("Frametime: %.2f ms", engine->stats.frametime);
    ImGui::Text("Bodies:    %u", s.num_bodies);
    ImGui::Text("Forces:    %u", s.num_forces);
    ImGui::Text("LBVH:      %.2f ms", s.lbvh_ms);
    ImGui::Text("Broad:     %.2f ms", s.broadphase_ms);
    ImGui::Text("Warmstart: %.2f ms", s.warmstart_ms);
    ImGui::Text("Solver:    %.2f ms", s.solver_ms);
    ImGui::Text("Total:     %.2f ms", s.total_ms);

    ImGui::SeparatorText("GPU");
    ImGui::Text("Pairs:     %u / %u", s.gpu_pairs,      MAX_GPU_PAIRS);
    ImGui::Text("Verts:     %u / %u", s.gpu_verts,      MAX_GPU_VERTS);
    ImGui::Text("Contacts:  %u / %u", s.gpu_contacts,   MAX_GPU_CONTACTS);
    ImGui::Text("SDF:       %u / %u KB",
        s.gpu_sdf_floats * 4 / 1024,
        physics->cap_sdf_floats * 4 / 1024);
    ImGui::Text("Particles: %u / %u", physics->particle_sim.particle_count(), MAX_PARTICLES);

    ImGui::End();

    PlayerCharacter& player = scenes->player;
    if (player.is_valid()) {
        ImGui::Begin("Player");
        ImGui::SliderFloat("Step time",        &player.step_time,           0.05f, 1.f);
        ImGui::SliderFloat("Step length",      &player.step_length_test,    0.f,   200.f);
        ImGui::SliderFloat("Leg angle speed",  &player.max_leg_angle_speed,  1.f,  50.f);
        ImGui::SliderFloat("Upright stiffness",    &player.upright_stiffness,          0.f,   30.f);
        ImGui::SliderFloat("Forward lean",         &player.forward_lean,               0.f,    0.6f);
        ImGui::SliderFloat("Ground spring k",      &player.ground_spring_k,            0.f, 10000.f);
        ImGui::SliderFloat("Ground spring rest",   &player.ground_spring_rest_length,  0.f,   100.f);
        ImGui::SeparatorText("Jump");
        ImGui::SliderFloat("Jump impulse",         &player.jump_impulse,               0.f,  1000.f);
        ImGui::SliderFloat("Air accel",            &player.air_accel,                  0.f,  5000.f);
        ImGui::SliderFloat("Ground check dist",    &player.ground_check_dist,          0.f,   200.f);
        ImGui::SliderFloat("Airborne dist",        &player.jump_airborne_dist,         0.f,   100.f);
        ImGui::SliderFloat("Spring disable time",  &player.jump_spring_disable_time,   0.f,     2.f);
        ImGui::End();
    }

    float sx = float(engine->_windowExtent.width)  / float(PHYSICS_WIDTH);
    float sy = float(engine->_windowExtent.height) / float(PHYSICS_HEIGHT);
    DebugDraw::get().render(sx, sy);
}
