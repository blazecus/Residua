#pragma once

#include <glm/glm.hpp>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"

class ResiduaEngine;

// Simulation resolution
static constexpr uint32_t PHYSICS_WIDTH  = 240;
static constexpr uint32_t PHYSICS_HEIGHT = 135;

// Reduction factor per pass
static constexpr uint32_t BODY_W      = 4;
static constexpr uint32_t BODY_H      = 4;
static constexpr uint32_t BODY_PIXELS = BODY_W * BODY_H;

// ─── CPU mirrors of physics_types.glsl (std430) ───────────────────────────────

struct RigidBody {          // 32 bytes
    glm::vec2 position;
    float     rotation;
    float     _pad{0.f};
    glm::vec2 velocity;
    float     angular_velocity;
    uint32_t  pixel_index;
};
static_assert(sizeof(RigidBody) == 32);

struct PhysicsPixel {       // 24 bytes
    glm::vec2 total_force;
    float     force_moment;
    float     total_mass;
    float     inertia_moment;
    float     _pad;
};
static_assert(sizeof(PhysicsPixel) == 24);

struct LoadedBodyImage {
    glm::vec4 pixels[BODY_PIXELS]; // rgba, a = mass (0..1)
};

LoadedBodyImage load_body_image(const char* path);

// ─── PhysicsPipeline ──────────────────────────────────────────────────────────

struct PhysicsPipeline {

    // ── GPU resources ─────────────────────────────────────────────────────
    AllocatedImage collision_layer;  // r8ui,       PHYSICS_WIDTH × PHYSICS_HEIGHT
    AllocatedImage output_screen;    // rgba16f,    PHYSICS_WIDTH × PHYSICS_HEIGHT

    AllocatedBuffer rb[2];           // RigidBody[] ping-pong, CPU_TO_GPU
    AllocatedBuffer physics_pixels;  // PhysicsPixel[] 2D grid (body_count*BODY_W) × BODY_H
    AllocatedBuffer pixel_colors;    // glm::vec4[] flat per-body-pixel, CPU_TO_GPU
    AllocatedBuffer body_l0;         // PhysicsPixel[] one per body (reduced)

    // ── Pipelines ─────────────────────────────────────────────────────────
    struct Pipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    };

    Pipeline pixel_physics_pl;
    Pipeline pixel_reduction_pl;
    Pipeline draw_pl;
    Pipeline integrate_pl;

    // ── Descriptor sets ───────────────────────────────────────────────────
    VkDescriptorSet physics_desc[2]{};   // pixel_physics:   reads rb[i]
    VkDescriptorSet reduction_desc{};    // pixel_reduction: physics_pixels → body_l0
    VkDescriptorSet draw_desc[2]{};      // rigidbody_draw:  reads rb[i]
    VkDescriptorSet integrate_desc[2]{}; // integrate:       rb[i] → rb[1-i]

    DescriptorAllocator desc_allocator;

    // ── State ─────────────────────────────────────────────────────────────
    uint32_t body_count{0};
    uint32_t max_bodies{0};
    uint32_t frame_parity{0};

    void init(ResiduaEngine* engine, uint32_t max_bodies);

    uint32_t add_body(
        ResiduaEngine*         engine,
        const LoadedBodyImage& img,
        glm::vec2              position,
        float                  rotation = 0.f
    );

    void dispatch(VkCommandBuffer cmd, float dt);
};
