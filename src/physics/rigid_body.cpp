#include "../physics/rigid_body.h"
#include "../physics/shape_gen.h"
#include <glm/gtx/rotate_vector.hpp>

AABB RigidBody2::generate_AABB() {
    glm::vec2 mn( 1e20f,  1e20f);
    glm::vec2 mx(-1e20f, -1e20f);
    for (const glm::vec2& v : shape) {
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    return { mn, mx };
}

void RigidBody2::generate_shape() {
    shape = ::generate_shape(sprite.width, sprite.height, sprite.pixels);
    glm::vec2 half(sprite.width * 0.5f, sprite.height * 0.5f);
    for (auto& v : shape)
        v -= half + com_local;
}

void RigidBody2::generate_sdf() {
    sdf   = ::generate_sdf(sprite, shape, com_local);
    sdf_w = sprite.width;
    sdf_h = sprite.height;
}

void RigidBody2::compute_mass_properties(float density) {
    float     pixel_count = 0.f;
    glm::vec2 centroid(0.f);

    for (uint32_t y = 0; y < sprite.height; y++) {
        for (uint32_t x = 0; x < sprite.width; x++) {
            if (sprite.pixels[y * sprite.width + x].w <= 0.5f) continue;
            glm::vec2 p{ (float)x + 0.5f, (float)y + 0.5f };
            centroid += p;
            pixel_count += 1.f;
        }
    }

    if (pixel_count == 0.f) {
        mass = inv_mass = inertia = inv_inertia = 0.f;
        return;
    }

    centroid /= pixel_count;
    com_local = centroid - glm::vec2(sprite.width * 0.5f, sprite.height * 0.5f);
    mass = density * pixel_count;

    float I = 0.f;
    for (uint32_t y = 0; y < sprite.height; y++) {
        for (uint32_t x = 0; x < sprite.width; x++) {
            if (sprite.pixels[y * sprite.width + x].w <= 0.5f) continue;
            glm::vec2 r = glm::vec2{ (float)x + 0.5f, (float)y + 0.5f } - centroid;
            I += glm::dot(r, r);
        }
    }

    inertia     = density * I;
    inv_mass    = 1.f / mass;
    inv_inertia = 1.f / inertia;
}
