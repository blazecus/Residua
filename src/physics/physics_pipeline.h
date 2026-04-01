#pragma once

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"

class ResiduaEngine;

static constexpr uint32_t PHYSICS_WIDTH  = 480;
static constexpr uint32_t PHYSICS_HEIGHT = 270;

static constexpr uint32_t BODY_W      = 4;
static constexpr uint32_t BODY_H      = 4;
static constexpr uint32_t BODY_PIXELS = BODY_W * BODY_H;

// ─── physics_types.glsl ───────────────────────────────

struct RigidBody {          // 40 bytes
    glm::vec2 position;
    float     rotation;
    float     total_mass;
    glm::vec2 velocity;
    float     angular_velocity;
    float     I_com;
    uint32_t  pixel_index;
    uint32_t  _pad{0};
};
static_assert(sizeof(RigidBody) == 40);

struct PhysicsPixel {        // 40 bytes
    glm::vec2 total_force;
    float     force_moment;
    float     total_mass;
    float     inertia_moment;
    float     collision_count;
    glm::vec2 contact_normal;  // Σ (mass_i × outward_normal_i)
    glm::vec2 contact_centroid;// Σ (mass_i × world_pos_i) of contact pixels
};
static_assert(sizeof(PhysicsPixel) == 40);

// body_id adheres to:
//  COLLISION_EMPTY  = 0
//  COLLISION_STATIC = 0xFFFFFFFF
//  dynamic body     = body_idx + 1
struct CollisionPixel {      // 12 bytes
    uint32_t  body_id;
    float     vel_x;
    float     vel_y;
};
static_assert(sizeof(CollisionPixel) == 12);

struct LoadedBodyImage {
    glm::vec4 pixels[BODY_PIXELS];
};

LoadedBodyImage load_body_image(const char* path);


struct PhysicsPipeline {

    AllocatedImage  output_screen;

    AllocatedBuffer rb[2];                  // RigidBody[]      capacity 
    AllocatedBuffer physics_pixels;         // PhysicsPixel[]   capacity * BODY_PIXELS
    AllocatedBuffer pixel_colors;           // glm::vec4[]      capacity * BODY_PIXELS
    AllocatedBuffer body_l0;                // PhysicsPixel[]   capacity (one per slot, sparse)
    AllocatedBuffer active_indices_buf;     // uint32_t[]       capacity
    AllocatedBuffer static_collision;       // CollisionPixel[] PHYSICS_WIDTH × PHYSICS_HEIGHT
    AllocatedBuffer collision_pixels;       // CollisionPixel[] PHYSICS_WIDTH × PHYSICS_HEIGHT

    struct Pipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    };

    Pipeline pixel_physics_pl;
    Pipeline pixel_reduction_pl;
    Pipeline draw_pl;
    Pipeline gap_fill_pl;
    Pipeline integrate_pl;

    VkDescriptorSet physics_desc[2]{};   // pixel_physics:   reads rb[i]
    VkDescriptorSet reduction_desc{};    // pixel_reduction: physics_pixels → body_l0
    VkDescriptorSet draw_desc[2]{};      // rigidbody_draw:  reads rb[i]
    VkDescriptorSet gap_fill_desc{};     // gap_fill:        output_image + collision_pixels
    VkDescriptorSet integrate_desc[2]{}; // integrate:       rb[i] → rb[1-i]

    DescriptorAllocator desc_allocator;

    uint32_t active_count{0};  // active bodies 
    uint32_t next_slot{0};     // next never-used slot
    uint32_t capacity{0};     

    std::vector<uint32_t> free_list;
    std::vector<uint32_t> active_slots; 

    uint32_t frame_parity{0};

    void init(ResiduaEngine* engine, uint32_t initial_capacity);

    uint32_t add_body(
        ResiduaEngine*         engine,
        const LoadedBodyImage& img,
        glm::vec2              position,
        float                  rotation = 0.f
    );

    void remove_body(uint32_t slot);

    // temporary use, probably - will be better in future to keep cpu side pixel buffers for gameplay mechanics
    // for now, just traverse each rb
    std::optional<uint32_t> body_at(glm::vec2 world_pos) const;

    void upload_collision_layer(ResiduaEngine* engine, const char* path);

    void dispatch(VkCommandBuffer cmd, float dt);

private:
    // double rb buffer
    void grow_buffers(ResiduaEngine* engine);

    void update_all_descriptors(VkDevice device);
};
