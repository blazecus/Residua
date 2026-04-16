#pragma once

#include "../physics/body_image.h"
#include "../physics/lbvh.h"
#include "../physics/shape_gen.h"
#include <string>

struct RigidBody2 {
    LoadedBodyImage sprite;
    std::vector<glm::vec2> shape;      // contour vertices (local space, centered)
    std::vector<float>     sdf;        // signed distance field (pixel units, negative inside)
    uint32_t sdf_w{0}, sdf_h{0};
    AABB unrotated_AABB;

    glm::vec3 position{0.f};
    glm::vec3 initial{0.f};    
    glm::vec3 inertial{0.f};   
    glm::vec3 velocity{0.f};
    glm::vec3 prev_velocity{0.f};

    float mass{1.f};
    float inertia{1.f};
    float inv_mass{1.f};
    float inv_inertia{1.f};
    float friction{0.3f};

    void generate_shape();
    void generate_sdf();

    void compute_mass_properties(float density = 1.f);

    void load_body(std::string& path);

    AABB generate_AABB();
};
