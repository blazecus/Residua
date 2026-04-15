#pragma once

#include "physics_world.h"
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include <optional>
#include <vector>
#include <cstdint>
#include <cmath>

class ResiduaEngine;

static constexpr uint32_t PHYSICS_WIDTH  = 480;
static constexpr uint32_t PHYSICS_HEIGHT = 270;

// Mirrors the GLSL CollisionPixel struct (32 bytes, std430).
struct CollisionPixel {
    uint32_t body_id{0};
    uint32_t body_id2{0};
    float    vel_x{0}, vel_y{0};
    float    vel_x2{0}, vel_y2{0};
    float    mass{0};
    float    mass2{0};
};
static_assert(sizeof(CollisionPixel) == 32);

// GPU-side rigid body draw struct — matches the GLSL struct in rigidbody_draw_cpu.comp.
struct RigidBodyDrawGPU {
    glm::vec2 position;           // world-space center (pixels)
    float     rotation;           // radians
    float     total_mass;
    glm::vec2 velocity;           // pixels/s
    float     angular_velocity;   // radians/s
    float     I_com;
    uint32_t  pixel_index;        // base offset into pixel_colors buffer
    uint32_t  body_w;
    uint32_t  body_h;
    uint32_t  _pad{0};
};
static_assert(sizeof(RigidBodyDrawGPU) == 48);

// Replaces PhysicsPipeline for CPU-driven physics.
// Holds the AVBD PhysicsWorld and uploads draw state to the GPU each frame.
class CPUDrawSystem {
public:
    PhysicsWorld world;

    // GPU resources
    AllocatedImage  output_screen;         // rgba16f, PHYSICS_WIDTH × PHYSICS_HEIGHT
    AllocatedBuffer rb_draw_buf;           // RigidBodyDrawGPU[] — written each frame
    AllocatedBuffer pixel_colors_buf;      // glm::vec4[]        — written on body add
    AllocatedBuffer collision_buf;         // CollisionPixel[]   — cleared each frame
    AllocatedBuffer active_indices_buf;    // uint32_t[]         — written each frame

    struct DrawPipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    } draw_pl, gap_fill_pl;

    VkDescriptorSet     draw_desc{VK_NULL_HANDLE};
    VkDescriptorSet     gap_fill_desc{VK_NULL_HANDLE};
    DescriptorAllocator desc_allocator;

    // Per-body pixel layout info (indexed by PhysicsWorld body slot).
    struct BodyDrawInfo {
        uint32_t pixel_index{0};
        uint32_t body_w{0};
        uint32_t body_h{0};
    };
    std::vector<BodyDrawInfo> body_draw_info;
    uint32_t next_pixel{0};    // next free offset in pixel_colors_buf
    uint32_t cap_bodies{0};    // current buffer capacity (bodies)
    uint32_t cap_pixels{0};    // current buffer capacity (pixels)

    void init(ResiduaEngine* engine);

private:
    // Grows rb_draw_buf, active_indices_buf, and/or pixel_colors_buf to cover
    // the requested counts (doubling strategy). Updates the draw descriptor set.
    void grow_buffers(ResiduaEngine* engine, uint32_t needed_bodies, uint32_t needed_pixels);

public:

    // Adds a body to the physics world and uploads its pixel data to the GPU.
    // Returns the world-body index.
    uint32_t add_body(ResiduaEngine* engine, RigidBody2 body);

    void remove_body(uint32_t idx);

    // Returns the body index at the given world-space pixel position, if any.
    std::optional<uint32_t> body_at(glm::vec2 world_pos) const;

    // Steps physics and renders one frame into output_screen.
    void dispatch(VkCommandBuffer cmd, float dt);

    PhysicsStats get_stats();
};
