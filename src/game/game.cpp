#include <src/game/game.h>
#include <SDL.h>

void Game::SDL_setup() {
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

	_window = SDL_CreateWindow("Graviator", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
		_windowExtent.height, window_flags);
}

void Game::init() {
	SDL_setup();

	client.init();

	engine.init(_windowExtent, _window);

	last_frame = std::chrono::system_clock::now();
}

void Game::run() {
    SDL_Event e;
    bool bQuit = false;

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
			client.processSDLEvent(e);
		}

		client.update();

		if (!freeze_rendering)
			std::cout << "test" << std::endl;
			engine.draw();
			
		std::chrono::microseconds elapsed{ 0 };
		auto end = std::chrono::system_clock::now();
		while (fps_limit && elapsed < frame_time) {
			elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-last_frame);
			end = std::chrono::system_clock::now();
		}

		last_frame = end;
	}
}

void Game::end() {
	//renderer.cleanup();
    SDL_DestroyWindow(_window);
}
