#include <src/game/game.h>

void Game::SDL_setup() {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow("Residua", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width, _windowExtent.height, window_flags);

    int32_t cursorData[2] = {0, 0};
    cursor = SDL_CreateCursor((Uint8*)cursorData, (Uint8*)cursorData, 8, 8, 4, 4);
    SDL_FreeCursor(cursor);
}

void Game::init() {
    SDL_setup();

    client.init();

    engine.init(_windowExtent, _window);

    ball_image = load_body_image("../assets/physics/ball.png");
    cshape     = load_body_image("../assets/physics/cshape.png");
    star       = load_body_image("../assets/physics/star.png");

    cpu_physics.init(&engine);
    engine.cpu_physics = &cpu_physics;

    scene_manager.init(&cpu_physics, &engine, &ball_image, &cshape, &star);
    ui_manager.init(&scene_manager, &cpu_physics, &engine);
    engine.ui_callback = [this]() { ui_manager.draw(); };

    scene_manager.load(0);

    lastFrame = std::chrono::system_clock::now();
}

void Game::run() {
    SDL_Event e;
    bool bQuit = false;
    client.resetMouse(_window);

    while (!bQuit) {

        client.resetInput();
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) freeze_rendering = true;
                if (e.window.event == SDL_WINDOWEVENT_RESTORED)  freeze_rendering = false;
            }
            engine.process_renderer_input(e);
            client.processSDLEvent(e);
        }

        auto input = client.inputManager.getInputs();
        glm::vec2 diff = glm::vec2(
            input.flou(InputManager::InputType::RIGHT) - input.flou(InputManager::InputType::LEFT),
            input.flou(InputManager::InputType::BACKWARD) - input.flou(InputManager::InputType::FORWARD)
        ) * 200.0f * delta;
        temp_player_position += diff;
        engine.process_player_update(temp_player_position);

        client.update();

        auto physics_mouse = [&]() -> glm::vec2 {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            return {
                mx * float(PHYSICS_WIDTH)  / float(_windowExtent.width),
                my * float(PHYSICS_HEIGHT) / float(_windowExtent.height)
            };
        };

        if (input.blou(InputManager::InputType::FORWARD) && lastSpawn > 20) {
            scene_manager.spawn_body(ball_image, physics_mouse());
            lastSpawn = 0;
        } else if (input.blou(InputManager::InputType::JUMP) && lastSpawn > 10) {
            scene_manager.spawn_body(cshape, physics_mouse());
            lastSpawn = 0;
        } else if (input.blou(InputManager::InputType::CROUCH) && lastSpawn > 30) {
            scene_manager.spawn_body(star, physics_mouse());
            lastSpawn = 0;
        }

        if (input.blou(InputManager::InputType::INSPECT)) {
            if (auto hit = cpu_physics.body_at(physics_mouse())) {
                if (cpu_physics.world.bodies[*hit].inv_mass > 0.f) // rbs only
                    cpu_physics.remove_body(*hit);
            }
        }

        lastSpawn++;

        if (!freeze_rendering) {
            engine._dt = delta;
            engine.draw_frame();
        }

        std::chrono::microseconds elapsed{ 0 };
        auto end = std::chrono::system_clock::now();
        while (fpsLimit && elapsed < frameTime) {
            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - lastFrame);
            end = std::chrono::system_clock::now();
        }

        lastFrame = end;
    }
}

void Game::end() {
    SDL_FreeCursor(cursor);
    SDL_DestroyWindow(_window);
}
