#include "client.h"

void Client::init() {
    // camera initials 
}

void Client::processSDLEvent(SDL_Event& e) {
    input_manager.processSDLEvent(e);
}

void Client::update() {
    InputManager::Inputs inputs = input_manager.getInputs();
}

void Client::resetInput() {
    input_manager.resetInput();
}
