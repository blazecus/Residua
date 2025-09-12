#pragma once

#include <src/game/client.h>
#include <src/renderer/waka_engine.h>
#include <chrono>

class Game{
public:

    VkExtent2D _windowExtent { 1700, 900 };
    struct SDL_Window* _window { nullptr };

	SDL_Cursor *cursor; 

	WakaEngine engine;

	bool freeze_rendering{ false };
	bool fps_limit = true;

	glm::vec2 temp_player_position = glm::vec2(0.0f);

	uint16_t fps = 144;
	float delta = 1.0 / fps;
	// frame time in microseconds - 16666 for 60fps, 6944 for 144fps
	std::chrono::microseconds frame_time{ 1000000 / fps};
	std::chrono::system_clock::time_point last_frame;

	Client client;
	
	void SDL_setup();

	void init();
	
	void run();

	void end();

};