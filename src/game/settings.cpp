#include "settings.h"
#include <fstream>
#include <limits>
#include <string>

GameSettings GameSettings::load(const char* path) {
    GameSettings s;
    std::ifstream file(path);
    if (!file.is_open()) return s;

    std::string key;
    while (file >> key) {
        if (key[0] == '#') {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else if (key == "window_width") {
            file >> s.window_width;
        } else if (key == "window_height") {
            file >> s.window_height;
        } else {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    return s;
}
