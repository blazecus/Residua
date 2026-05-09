#pragma once

#include <src/physics/physics_engine.h>
#include <src/physics/body_image.h>
#include <src/game/player/player_character.h>
#include <src/game/input_manager.h>
#include <string>

class ResiduaEngine;

struct SceneEntry {
    const char* name;
    const char* path;
};

static constexpr SceneEntry SCENE_LIST[] = {
    { "Rope",        "../assets/scenes/spring_chain.scene" },
    { "Free Play",   "../assets/scenes/free_play.scene"    },
    { "Softbody",    "../assets/scenes/softbody.scene"     },
    { "Player Test", "../assets/scenes/player_test.scene"  },
    { "Long Level",  "../assets/scenes/long_level.scene"   },
};
static constexpr int NUM_SCENES = sizeof(SCENE_LIST) / sizeof(SCENE_LIST[0]);

struct SceneManager {
    PhysicsEngine*         physics    { nullptr };
    ResiduaEngine*         engine     { nullptr };
    const LoadedBodyImage* ball_image { nullptr };
    const LoadedBodyImage* cshape     { nullptr };
    const LoadedBodyImage* star       { nullptr };

    int current_scene { -1 };

    glm::vec2 camera_offset { 0.f, 0.f };
    float     world_w       { 1e9f };
    float     world_h       { 1e9f };

    PlayerCharacter player;

    void init(PhysicsEngine* phys, ResiduaEngine* eng,
              const LoadedBodyImage* ball, const LoadedBodyImage* cs,
              const LoadedBodyImage* st);

    void clear();
    void load(int idx);
    void apply_inputs(const InputManager::Inputs& inputs, glm::vec2 aim_pos);
    void update(float dt);

    uint32_t spawn_body(const LoadedBodyImage& img, glm::vec2 world_pos);

private:
    void load_file(const std::string& path);
};
