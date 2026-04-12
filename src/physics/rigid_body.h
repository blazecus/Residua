#pragma once

#include "../physics/physics_engine.h"
#include "../physics/lbvh.h"
#include "../physics/shape_gen.h"
#include <string>

struct RigidBody2 {
    LoadedBodyImage sprite;
    std::vector<glm::vec2> shape;      // ordered outline from marching squares + DP
    std::vector<Triangle>  triangles;  // ear-clipped decomposition of shape
    AABB unrotated_AABB;

    glm::vec2 position;
    glm::vec2 position_prev;
    glm::vec2 velocity;
    float rotation;
    float rotation_prev;
    float angular_velocity{0.f};

    float mass{1.f};
    float inertia{1.f};
    float inv_mass{1.f};
    float inv_inertia{1.f};

    void compute_mass_properties(float density = 1.f);

    void generate_shape();
    void triangulate_shape();

    void load_body(std::string& path);

    AABB generate_AABB();
};
