#pragma once

#include <cstdint>

struct GameSettings {
    uint32_t window_width  = 1700;
    uint32_t window_height = 900;

    // Loads from path, returning defaults if the file is missing.
    static GameSettings load(const char* path = "../settings.cfg");
};
