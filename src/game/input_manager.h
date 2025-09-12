#pragma once

#include "SDL_events.h"
#include "glm/vec2.hpp"
#include <unordered_map>
#include <vector>
#include <array>
#include <iostream>

class InputManager {

	static constexpr float DEAD_ZONE = 0.02f;
public:

	enum InputType {
		FORWARD, BACKWARD, LEFT, RIGHT,
		SELECT, JUMP, CROUCH, BACK,
		CAMERA_MOTION,
		NO_BINDING,
		INPUT_TYPE_COUNT
	};

	struct Output {
		bool bool_output = false;
		float float_output = 0.0f;
	};

    struct KeyboardInputMap {
		std::unordered_map<SDL_Keycode, InputType> key_bindings;
		std::unordered_map<int, InputType> mouse_bindings;
    };

	struct ControllerInputMap {
		std::unordered_map<uint8_t, InputType> button_bindings;
		std::unordered_map<uint8_t, std::vector<InputType>> axis_bindings;
		std::unordered_map<InputType, int> axis_orientation; 	
	};

	struct InputSettings {
		glm::vec2 mouse_sensitivity = glm::vec2(1.0f);
		glm::vec2 joystick_camera_sensitivity = glm::vec2(1.0f);

		KeyboardInputMap keyboard_map;
		ControllerInputMap controller_map;
		bool keyboard_toggle = true;
	};

    struct Inputs {
		std::unordered_map<InputType, Output> bindings = {
			{FORWARD,  Output{}},
			{BACKWARD, Output{}},
			{LEFT,     Output{}},
			{RIGHT,    Output{}},
			{SELECT,   Output{}},
			{JUMP,     Output{}},
			{CROUCH,   Output{}},
			{BACK,     Output{}}
		};
		
        glm::vec2 camera_movement = glm::vec2(0.0f);
		
		// access functions
		float  flou(const InputType input) const { return bindings.at(input).float_output; }; // float out
		bool   blou(const InputType input) const { return bindings.at(input).bool_output; }; // bool out
		Output rawou(const InputType input) const { return bindings.at(input); }; // raw output representation
    };

	SDL_GameController* controller;

	InputSettings settings;
    Inputs inputs;

	InputManager() {
		inputs = Inputs{};
		settings = InputSettings{};

		defaultKeyboardMap();
		defaultControllerMap();
	}

	// default mappings

	void defaultKeyboardMap() {
		settings.keyboard_map.key_bindings = {
			{SDLK_w, FORWARD},
			{SDLK_s, BACKWARD},
			{SDLK_a, LEFT},
			{SDLK_d, RIGHT},
			{SDLK_SPACE, JUMP},
			{SDLK_LSHIFT, CROUCH},
			{SDLK_BACKSPACE, BACK}
		};
		settings.keyboard_map.mouse_bindings = {
			{SDL_BUTTON_LEFT, SELECT}
		};
	}

	void defaultControllerMap() {
		settings.controller_map.axis_orientation = {
			{JUMP,   0},
			{CROUCH, 0},
			{BACK,   0},
			{SELECT, 0},

			{FORWARD,   1},
			{BACKWARD, -1},
			{LEFT,     -1},
			{RIGHT,     1}
		};
		settings.controller_map.button_bindings = {
			{SDL_CONTROLLER_BUTTON_A, JUMP},
			{SDL_CONTROLLER_BUTTON_X, CROUCH},
			{SDL_CONTROLLER_BUTTON_B, BACK},
			{SDL_CONTROLLER_BUTTON_A, SELECT}
		};
		settings.controller_map.axis_bindings = {
			{SDL_CONTROLLER_AXIS_LEFTY, {FORWARD, BACKWARD}},
			{SDL_CONTROLLER_AXIS_LEFTX, {LEFT, RIGHT}}
		};
	}

	// finds controller, sets to nullptr if no controller
	SDL_GameController* findController() {
		for (int i = 0; i < SDL_NumJoysticks(); i++) {
			if (SDL_IsGameController(i)) {
				return SDL_GameControllerOpen(i);
			}
		}

		return nullptr;
	}

	// converts SDL input events -> game input events, represented by Inputs struct
	void processSDLEvent(SDL_Event& e) {

		// CONTROLLER CONNECTION
		switch (e.type) {

		case SDL_CONTROLLERDEVICEADDED:
			if (!controller) {
				controller = SDL_GameControllerOpen(e.cdevice.which);
			}
			break;

		case SDL_CONTROLLERDEVICEREMOVED:
			if (controller && e.cdevice.which == SDL_JoystickInstanceID(
				SDL_GameControllerGetJoystick(controller))) {
				SDL_GameControllerClose(controller);
				controller = findController();
			}
			break;
		}

		// KEYBOARD / MOUSE CONTROLS
		// enables by default if there is no controller connected
		if (settings.keyboard_toggle || !controller) {

			// event handling - determine output before mapping to game input
			Output output;
			inputs.camera_movement = glm::vec2(0.0f);
			if (e.type == SDL_KEYUP || e.type == SDL_MOUSEBUTTONUP) {
				output.bool_output = false;
				output.float_output = 0.0f;
			}
			else if (e.type == SDL_KEYDOWN || e.type == SDL_MOUSEBUTTONDOWN) {
				output.bool_output = true;
				output.float_output = 1.0f;
			}
			else if (e.type == SDL_MOUSEMOTION) {
				inputs.camera_movement = glm::vec2(e.motion.xrel, e.motion.yrel);
				return;
			}
			else {
				return;
			}
	
			// map event input to game input
			InputType binding;
			if (settings.keyboard_map.key_bindings.find(e.key.keysym.sym) != settings.keyboard_map.key_bindings.end()) {
				binding = settings.keyboard_map.key_bindings[e.key.keysym.sym];
			}
			else if(settings.keyboard_map.mouse_bindings.find(e.button.button) != settings.keyboard_map.mouse_bindings.end()) {
				binding = settings.keyboard_map.mouse_bindings[e.button.button];
			}
			else {
				return;
			}

			// input binding
			inputs.bindings[binding] = output;	
			return;
		}

		// CONTROLLER HANDLING
		else {
			if (e.cdevice.which == SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(controller))) {
				Output output;
				switch (e.type) {
				case SDL_CONTROLLERAXISMOTION:
				{
					// axis events handled separately

					// camera is always on right controller axis
					if (e.caxis.axis == SDL_CONTROLLER_AXIS_RIGHTX) {
						inputs.camera_movement.x = e.caxis.value / 32768; // magic number: 32768 is max absolute value of an axis
						return;
					}
					else if (e.caxis.axis == SDL_CONTROLLER_AXIS_RIGHTY) {
						inputs.camera_movement.y = e.caxis.value / 32768;
						return;
					}
					
					// generalized mapped axis handling
					if (settings.controller_map.axis_bindings.find(e.caxis.axis) == settings.controller_map.axis_bindings.end()) return;

					std::vector<InputType> bindings = settings.controller_map.axis_bindings[e.caxis.axis];
					float mag = e.caxis.value / 32768; //float normalization
					for (InputType& input : bindings) {
						int orientation = settings.controller_map.axis_orientation[input];
						Output axis_output;
						// axis supports float input, which are considered valid if the axis passes a dead zone
						axis_output.bool_output = mag * orientation > DEAD_ZONE;
						axis_output.float_output = abs(mag) * axis_output.bool_output;

						inputs.bindings[input] = axis_output;
					}

					break;
				}
				case SDL_CONTROLLERBUTTONDOWN:
				{
					// buttons work just like keys do
					if (settings.controller_map.button_bindings.find(e.cbutton.button) == settings.controller_map.button_bindings.end()) return;

					InputType binding = settings.controller_map.button_bindings[e.cbutton.button];
					output.bool_output = true;
					output.float_output = 1.0f;
					inputs.bindings[binding] = output;
					break;
				}

				case SDL_CONTROLLERBUTTONUP:
				{
					if (settings.controller_map.button_bindings.find(e.cbutton.button) == settings.controller_map.button_bindings.end()) return;

					InputType binding = settings.controller_map.button_bindings[e.cbutton.button];
					output.bool_output = false;
					output.float_output = 0.0f;
					inputs.bindings[binding] = output;
					break;
				}
				}
			}

			return;
		}
    }

	const Inputs& getInputs() {
		return inputs;
	}

	void resetMouse(SDL_Window* window) {
		SDL_WarpMouseInWindow(window,0,0);
		inputs.camera_movement = glm::vec2(0.0f);
	}

	void resetInput() {
		inputs.camera_movement = glm::vec2(0.0f);
	}
};
