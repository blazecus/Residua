#pragma once

#include "../physics/body_image.h"
#include "../physics/lbvh.h"
#include "../physics/shape_gen.h"
#include <string>

struct RigidBody {
    LoadedBodyImage sprite;
    std::vector<glm::vec2> shape;
    std::vector<float>     sdf;
    uint32_t sdf_w{0}, sdf_h{0};
    AABB unrotated_AABB;

    // Body-pixel-space position of the deepest interior point
    glm::vec2 thickest_px{0.f};

    glm::vec3 position{0.f};
    glm::vec3 initial{0.f};    
    glm::vec3 inertial{0.f};   
    glm::vec3 velocity{0.f};
    glm::vec3 prev_velocity{0.f};

    glm::vec2 com_local{0.f};  

    float    density{1.f};
    float    mass{1.f};
    float    inertia{1.f};
    float    inv_mass{1.f};
    float    inv_inertia{1.f};
    float    friction{0.3f};
    float    angular_damping{1.f};
    float    fracture_threshold{0.f};  // |normal lambda| needed to fracture; 0 = unbreakable

    uint32_t collision_layer{0x00000001u};
    uint32_t collision_mask {0xFFFFFFFFu};

    glm::vec2 ext_force {0.f};
    float     ext_torque{0.f};

    bool     visible   {true};
    bool     flip_h    {false};
    uint32_t draw_layer{0};

    // Bodies connected by non-contact forces (joints). Maintained incrementally
    std::vector<uint32_t> joint_connected;

    void generate_shape();
    void generate_sdf();

    void compute_mass_properties(float density = 1.f);

    void load_body(std::string& path);

    AABB generate_AABB();
};
