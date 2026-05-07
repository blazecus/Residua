#pragma once

#include <src/game/client.h>
#include <src/game/scenes.h>
#include <src/game/ui.h>
#include <src/renderer/residua_engine.h>
#include <src/physics/physics_engine.h>
#include <src/physics/joint.h>
#include <src/physics/body_image.h>
#include <chrono>

class Game {
public:

    VkExtent2D _windowExtent { 1700, 900 };
    struct SDL_Window* _window { nullptr };

    SDL_Cursor* cursor;

    ResiduaEngine  engine;
    PhysicsEngine  physics_engine;

    bool freeze_rendering{ false };
    bool fpsLimit = true;

    uint16_t fps = 60;
    float delta = 1.0f / fps;
    std::chrono::microseconds frameTime{ 1000000 / fps };
    std::chrono::system_clock::time_point lastFrame;

    uint32_t   lastSpawn  = 0;
    uint32_t   drag_body  { ~0u };
    MouseDrag* drag_force { nullptr };

    Client client;

    LoadedBodyImage ball_image;
    LoadedBodyImage cshape;
    LoadedBodyImage star;

    SceneManager scene_manager;
    UIManager    ui_manager;

    void SDL_setup();
    void init();
    void run();
    void end();
};
