#include <src/game/game.h>

void Game::SDL_setup() {
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

	_window = SDL_CreateWindow("Residua", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
		_windowExtent.height, window_flags);

	int32_t cursorData[2] = {0, 0};
	cursor = SDL_CreateCursor((Uint8 *)cursorData, (Uint8 *)cursorData, 8, 8, 4, 4);
	SDL_FreeCursor(cursor);
}

void Game::init() {
	SDL_setup();

	client.init();

	engine.init(_windowExtent, _window);

	ball_image = load_body_image("../assets/physics/ball.png");
	cshape = load_body_image("../assets/physics/cshape.png");
	star = load_body_image("../assets/physics/star.png");

	physics.init_avbd(&engine, 1024);
	engine.physics = &physics;

	// Static floor — wide rectangle at the bottom of the physics world
	{
		LoadedBodyImage floor;
		constexpr float hw = 400.f, hh = 10.f;
		floor.shape = { {-hw,-hh}, {hw,-hh}, {hw,hh}, {-hw,hh} };
		physics.add_avbd_body(&engine, floor,
		                      {PHYSICS_WIDTH * 0.5f, PHYSICS_HEIGHT - 30.f},
		                      /*mass=*/0.f);
	}

	lastFrame = std::chrono::system_clock::now();
}

void Game::run() {
    SDL_Event e;
    bool bQuit = false;
	client.resetMouse(_window);

    // main loop
	while (!bQuit) {

		// Handle events on queue
		//input_manager.resetInput();
		client.resetInput();
		while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
				if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
					freeze_rendering = true;
				}
				if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
					freeze_rendering = false;
				}
            }
			engine.process_renderer_input(e);
			client.processSDLEvent(e);

			//engine.process_player_update(temp_player_position / glm::vec2(_windowExtent.width, _windowExtent.height));
		}

		auto input = client.inputManager.getInputs();
		glm::vec2 diff = glm::vec2(
			input.flou(InputManager::InputType::RIGHT) - input.flou(InputManager::InputType::LEFT),
			input.flou(InputManager::InputType::BACKWARD) - input.flou(InputManager::InputType::FORWARD)
		) * 200.0f * delta;
		temp_player_position += diff;
		engine.process_player_update(temp_player_position);


		client.update();

		if (input.blou(InputManager::InputType::SELECT) && lastSpawn > 20) {
			int mx, my;
			SDL_GetMouseState(&mx, &my);
			glm::vec2 spawn_pos = {
				mx * float(PHYSICS_WIDTH)  / float(_windowExtent.width),
				my * float(PHYSICS_HEIGHT) / float(_windowExtent.height)
			};
			physics.add_avbd_body(&engine, ball_image, spawn_pos, 1.0f);
			lastSpawn = 0;
		}
		else if (input.blou(InputManager::InputType::JUMP) && lastSpawn > 10) {
			int mx, my;
			SDL_GetMouseState(&mx, &my);
			glm::vec2 spawn_pos = {
				mx * float(PHYSICS_WIDTH)  / float(_windowExtent.width),
				my * float(PHYSICS_HEIGHT) / float(_windowExtent.height)
			};
			physics.add_avbd_body(&engine, cshape, spawn_pos, 1.0f);
			lastSpawn = 0;
		}
		else if (input.blou(InputManager::InputType::CROUCH) && lastSpawn > 30) {
			int mx, my;
			SDL_GetMouseState(&mx, &my);
			glm::vec2 spawn_pos = {
				mx * float(PHYSICS_WIDTH) / float(_windowExtent.width),
				my * float(PHYSICS_HEIGHT) / float(_windowExtent.height)
			};
			physics.add_avbd_body(&engine, star, spawn_pos, 2.0f);
			lastSpawn = 0;
		}

		lastSpawn++;

		if (!freeze_rendering) {
			engine._dt = delta;
			engine.draw_frame();
		}
			
		std::chrono::microseconds elapsed{ 0 };
		auto end = std::chrono::system_clock::now();
		while (fpsLimit && elapsed < frameTime) {
			elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-lastFrame);
			end = std::chrono::system_clock::now();
		}

		lastFrame = end;
	}
}

void Game::end() {
	//renderer.cleanup();
	SDL_FreeCursor(cursor);
    SDL_DestroyWindow(_window);
}
