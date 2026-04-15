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

    // Static boundaries (floor + walls).
    const float W = float(PHYSICS_WIDTH);
    const float H = float(PHYSICS_HEIGHT);
    const float T = 20.f;  // wall thickness
    cpu_physics.world.add_static_rect({ W * 0.5f, H + T * 0.5f }, W, T);  // floor
    cpu_physics.world.add_static_rect({ W * 0.5f, -T * 0.5f    }, W, T);  // ceiling
    cpu_physics.world.add_static_rect({ -T * 0.5f, H * 0.5f    }, T, H);  // left wall
    cpu_physics.world.add_static_rect({ W + T * 0.5f, H * 0.5f }, T, H);  // right wall

    lastFrame = std::chrono::system_clock::now();
}

// Helper: build a RigidBody2 from a loaded image and spawn it at world_pos.
static void spawn_body(CPUDrawSystem& cpu, ResiduaEngine& engine,
                       const LoadedBodyImage& img, glm::vec2 world_pos)
{
    RigidBody2 rb;
    rb.sprite = img;
    rb.generate_shape();
    rb.triangulate_shape();
    rb.compute_mass_properties();
    rb.position = glm::vec3(world_pos, 0.f);
    cpu.add_body(&engine, std::move(rb));
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

        // Get mouse position in physics-space coords.
        auto physics_mouse = [&]() -> glm::vec2 {
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            return {
                mx * float(PHYSICS_WIDTH)  / float(_windowExtent.width),
                my * float(PHYSICS_HEIGHT) / float(_windowExtent.height)
            };
        };

        if (input.blou(InputManager::InputType::SELECT) && lastSpawn > 2) {
            spawn_body(cpu_physics, engine, ball_image, physics_mouse());
            lastSpawn = 0;
        } else if (input.blou(InputManager::InputType::JUMP) && lastSpawn > 10) {
            spawn_body(cpu_physics, engine, cshape, physics_mouse());
            lastSpawn = 0;
        } else if (input.blou(InputManager::InputType::CROUCH) && lastSpawn > 30) {
            spawn_body(cpu_physics, engine, star, physics_mouse());
            lastSpawn = 0;
        }

        if (input.blou(InputManager::InputType::INSPECT)) {
            if (auto hit = cpu_physics.body_at(physics_mouse()))
                cpu_physics.remove_body(*hit);
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
