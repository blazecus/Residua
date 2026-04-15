#pragma once

#include <src/game/client.h>
#include <src/renderer/residua_engine.h>
#include <src/physics/cpu_draw_system.h>
#include <src/physics/body_image.h>
#include <chrono>

class Game {
public:

    VkExtent2D _windowExtent { 1700, 900 };
    struct SDL_Window* _window { nullptr };

    SDL_Cursor* cursor;

    ResiduaEngine  engine;
    CPUDrawSystem  cpu_physics;

    bool freeze_rendering{ false };
    bool fpsLimit = true;

    glm::vec2 temp_player_position = glm::vec2(0.0f);

    uint16_t fps = 144;
    float delta = 1.0f / fps;
    std::chrono::microseconds frameTime{ 1000000 / fps };
    std::chrono::system_clock::time_point lastFrame;

    uint32_t lastSpawn = 0;

    Client client;

    LoadedBodyImage ball_image;
    LoadedBodyImage cshape;
    LoadedBodyImage star;

    void SDL_setup();
    void init();
    void run();
    void end();
};
