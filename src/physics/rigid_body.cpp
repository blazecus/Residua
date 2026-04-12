#include "../physics/rigid_body.h"
#include "../physics/shape_gen.h"

AABB RigidBody2::generate_AABB() {
    glm::vec2 min = {INF, INF};
    glm::vec2 max = {-INF, -INF};
    for (glm::vec2 vert : shape) {
        min = glm::min(min, vert);
        max = glm::max(max, vert);
    }

    return {min, max};
}

void RigidBody2::generate_shape(){
    shape = ::generate_shape(sprite.width, sprite.height, sprite.pixels);
}

void RigidBody2::triangulate_shape() {
    triangles = triangulate(shape);
}

void RigidBody2::compute_mass_properties(float density) {
    float pixel_count = 0.f;
    glm::vec2 centroid(0.f);

    for (uint32_t y = 0; y < sprite.height; y++) {
        for (uint32_t x = 0; x < sprite.width; x++) {
            float alpha = sprite.pixels[y * sprite.width + x].w;
            if (alpha <= 0.5f) continue;
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
    mass      = density * pixel_count;

    float I = 0.f;
    for (uint32_t y = 0; y < sprite.height; y++) {
        for (uint32_t x = 0; x < sprite.width; x++) {
            float alpha = sprite.pixels[y * sprite.width + x].w;
            if (alpha <= 0.5f) continue;
            glm::vec2 r = glm::vec2{ (float)x + 0.5f, (float)y + 0.5f } - centroid;
            I += glm::dot(r, r);
        }
    }

    inertia     = density * I;
    inv_mass    = 1.f / mass;
    inv_inertia = 1.f / inertia;
}