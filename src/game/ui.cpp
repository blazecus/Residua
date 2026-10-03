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
        ImGui::SliderFloat("Step period",      &player.step_period,         0.1f,  3.f);
        ImGui::SliderFloat("Leg angle speed",  &player.max_leg_angle_speed,  1.f,  50.f);
        ImGui::SliderFloat("Support arm offset angle", &player.support_arm_offset_angle, -3.14f, 3.14f);
        ImGui::SliderFloat("Foot pin break force", &player.foot_pin_break_force, 1.f, 1e9f,
                            "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Bend frequency (Hz)", &player.joint_bend_hz,            0.1f, 3000.f,
                            "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Bend damping ratio",  &player.joint_bend_damping_ratio, 0.f,  3.f);
        ImGui::SliderFloat("Bend max accel",      &player.joint_bend_max_accel,     1.f,  1e8f,
                            "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::Text("(Torso upright anchor shares the Bend sliders above -- see torso_anchor)");
        ImGui::SeparatorText("Walk");
        ImGui::SliderFloat("Walk leg angle speed",      &player.walk_leg_angle_speed,      1.f, 50.f);
        ImGui::SliderFloat("Walk back step period",     &player.backward_step_period,      0.1f,  3.f);
        ImGui::SliderFloat("Run back step period",      &player.backward_run_step_period,  0.1f,  3.f);
        ImGui::SliderFloat("Run back leg angle speed",  &player.backward_run_leg_angle_speed, 1.f, 50.f);
        ImGui::SliderFloat("Walk swing knee bend",      &player.recovery_bend_angle,       0.f,  3.14f);
        ImGui::SliderFloat("Walk back swing knee bend", &player.backward_swing_knee_bend,  0.f,  3.14f);
        ImGui::SeparatorText("Gait");
        ImGui::SliderFloat("Feet gap (standing)",       &player.feet_gap,                  0.f, 30.f);
        ImGui::SliderFloat("Swing search offset X",     &player.swing_search_offset_x,     0.f, 30.f);
        ImGui::SliderFloat("Swing search back",         &player.swing_search_back,         0.f, 30.f);
        ImGui::SliderFloat("Stance lean angle",         &player.stance_lean_angle,         0.f,  3.14f);
        ImGui::SliderFloat("Swing thigh forward angle", &player.swing_thigh_forward_angle, 0.f,  3.14f);
        ImGui::SliderFloat("Standing lower lean angle", &player.standing_lower_lean_angle, -1.5708f, 1.5708f);
        ImGui::SliderFloat("Standing thigh bend angle", &player.standing_thigh_bend_angle, -1.5708f, 1.5708f);
        ImGui::SliderFloat("Foot land dist",            &player.foot_land_dist,            0.f, 20.f);
        ImGui::SliderFloat("Standing origin Y",         &player.standing_origin_y,       -20.f, 20.f);
        ImGui::SliderFloat("Standing settle speed",     &player.standing_settle_speed,     0.f, 50.f);
        ImGui::SliderFloat("Settle step dist",          &player.settle_step_dist,          0.f, 20.f);
        ImGui::SliderFloat("Trailing kick angle",       &player.trailing_kick_angle,       0.f,  3.14f);
        ImGui::SeparatorText("Run");
        ImGui::SliderFloat("Run speed target",          &player.run_speed_target,          0.f, 300.f);
        ImGui::SliderFloat("Run speed Kp",              &player.run_speed_kp,              0.f,  50.f);
        ImGui::SliderFloat("Run speed Kd",              &player.run_speed_kd,              0.f,   5.f);
        ImGui::SliderFloat("Run max accel",             &player.run_max_accel,             0.f, 3000.f);
        ImGui::SliderFloat("Run release angle",         &player.run_release_angle,         0.f,   1.5708f);
        ImGui::SliderFloat("Run drive lead",            &player.run_drive_lead,            0.f,  20.f);
        ImGui::SliderFloat("Run drive lift",            &player.run_drive_lift,            0.f,  10.f);
        ImGui::SliderFloat("Run swing thigh angle",     &player.run_swing_thigh_angle,     0.f,   2.3562f);
        ImGui::SliderFloat("Run tuck knee bend",        &player.run_tuck_knee_bend,        0.f,   2.7489f);
        ImGui::SliderFloat("Run lift speed",            &player.run_lift_speed,            0.f, 200.f);
        ImGui::SliderFloat("Run flight delay",          &player.run_flight_delay,          0.f,   0.3f);
        ImGui::SliderFloat("Run step ahead",            &player.run_step_ahead,            0.f,  40.f);
        ImGui::SliderFloat("Run stop speed",            &player.run_stop_speed,            0.f, 100.f);
        ImGui::SliderFloat("Run stop decel",            &player.run_stop_decel,            0.f, 3000.f);
        ImGui::SliderFloat("Run airborne delay",        &player.run_airborne_delay,        0.f,   1.f);
        ImGui::SliderFloat("Run foot point angle",      &player.run_foot_point_angle,     -1.5708f, 2.0f);
        ImGui::SeparatorText("Safety / Jump");
        ImGui::SliderFloat("Foot angle speed",          &player.foot_angle_speed,          1.f, 100.f);
        ImGui::SliderFloat("Jump impulse",              &player.jump_impulse,              0.f, 200000.f);
        ImGui::SliderFloat("Jump squat sink speed",     &player.jump_squat_sink_speed,     0.f, 200.f);
        ImGui::SliderFloat("Jump squat min hip height", &player.jump_squat_min_hip_height, 0.f, 16.f);
        ImGui::SliderFloat("Air leg angle speed",       &player.air_leg_angle_speed,       0.5f, 30.f);
        ImGui::SliderFloat("Rise thigh L",              &player.jump_rise_thigh_l,         0.f,  2.5f);
        ImGui::SliderFloat("Rise thigh R",              &player.jump_rise_thigh_r,         0.f,  2.5f);
        ImGui::SliderFloat("Rise knee bend",            &player.jump_rise_knee_bend,       0.f,  2.7f);
        ImGui::SliderFloat("Fall thigh L",              &player.jump_fall_thigh_l,        -1.f,  2.5f);
        ImGui::SliderFloat("Fall thigh R",              &player.jump_fall_thigh_r,        -1.f,  2.5f);
        ImGui::SliderFloat("Fall knee bend",            &player.jump_fall_knee_bend,       0.f,  2.7f);
        ImGui::SliderFloat("Fall pose delay",           &player.jump_fall_delay,           0.f,  2.f);
        ImGui::End();
    }

    float sx = float(engine->_windowExtent.width)  / float(PHYSICS_WIDTH);
    float sy = float(engine->_windowExtent.height) / float(PHYSICS_HEIGHT);
    DebugDraw::get().render(sx, sy, scenes->camera_offset);
}
