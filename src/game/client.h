#pragma once

#include <src/game/input_manager.h>

class Client{
public:

	InputManager input_manager;

	void update();

	void resetInput();
	
	void processSDLEvent(SDL_Event& e);

	void init();

	void resetMouse(SDL_Window* window);

};