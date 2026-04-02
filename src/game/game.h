#pragma once

#include <src/game/client.h>
#include <src/renderer/residua_engine.h>
#include <src/physics/physics_engine.h>
#include <chrono>

class Game{
public:

    VkExtent2D _windowExtent { 1700, 900 };
    struct SDL_Window* _window { nullptr };

	SDL_Cursor *cursor; 

	ResiduaEngine engine;
	PhysicsPipeline physics;

	bool freeze_rendering{ false };
	bool fpsLimit = true;

	glm::vec2 temp_player_position = glm::vec2(0.0f);

	uint16_t fps = 144;
	float delta = 1.0 / fps;
	// frame time in microseconds - 16666 for 60fps, 6944 for 144fps
	std::chrono::microseconds frameTime{ 1000000 / fps};
	std::chrono::system_clock::time_point lastFrame;

	uint32_t lastSpawn = 0;

	Client client;

	LoadedBodyImage ball_image;

	void SDL_setup();

	void init();
	
	void run();

	void end();

};