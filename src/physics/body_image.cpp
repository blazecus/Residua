#include "body_image.h"
#include <stb_image.h>
#include <cassert>

LoadedBodyImage load_body_image(const char* path) {
    int w, h, channels;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    assert(data && "load_body_image: failed to open file");

    LoadedBodyImage result;
    result.width  = (uint32_t)w;
    result.height = (uint32_t)h;
    result.pixels.resize(w * h);
    for (int i = 0; i < w * h; i++) {
        result.pixels[i] = glm::vec4(
            data[i * 4 + 0] / 255.f,
            data[i * 4 + 1] / 255.f,
            data[i * 4 + 2] / 255.f,
            data[i * 4 + 3] / 255.f
        );
    }
    stbi_image_free(data);
    return result;
}
