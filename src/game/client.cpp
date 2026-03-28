#include "client.h"

void Client::init() {
    // camera initials 
}

void Client::processSDLEvent(SDL_Event& e) {
    inputManager.processSDLEvent(e);
}

void Client::update() {
    InputManager::Inputs inputs = inputManager.getInputs();
}

void Client::resetInput() {
    inputManager.resetInput();
}

void Client::resetMouse(SDL_Window* window) {
    inputManager.resetMouse(window);
}
