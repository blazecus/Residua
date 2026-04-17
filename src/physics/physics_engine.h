#pragma once

#include "physics_world.h"
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include <optional>
#include <vector>
#include <cstdint>
#include <cmath>

class ResiduaEngine;

static constexpr uint32_t PHYSICS_WIDTH    = 480;
static constexpr uint32_t PHYSICS_HEIGHT   = 270;
static constexpr uint32_t MAX_GPU_EDGES    = 16384;
static constexpr uint32_t MAX_GPU_CONTACTS = 4096;

struct GPUEdge {
    glm::vec2 v0, v1;
    glm::vec2 ref_pos;
    float     ref_angle;
    float     normal_sign;
    uint32_t  sdf_offset;   
    uint32_t  sdf_w, sdf_h;
    uint32_t  body_a, body_b;
    uint32_t  _pad{0};
};
static_assert(sizeof(GPUEdge) == 56);

struct GPUContact {
    glm::vec2 world_pt;
    glm::vec2 normal;
    float     depth;
    uint32_t  body_a, body_b;
    uint32_t  _pad{0};
};
static_assert(sizeof(GPUContact) == 32);

struct RigidBodyDrawGPU {
    glm::vec2 position;
    float     rotation;
    float     total_mass;
    glm::vec2 velocity;
    float     angular_velocity;
    float     I_com;
    uint32_t  pixel_index;
    uint32_t  body_w;
    uint32_t  body_h;
    uint32_t  sdf_offset{0};
};
static_assert(sizeof(RigidBodyDrawGPU) == 48);

class PhysicsEngine {
public:
    PhysicsWorld world;

    AllocatedImage  output_screen;         
    AllocatedBuffer rb_draw_buf;          
    AllocatedBuffer pixel_colors_buf;     
    AllocatedBuffer active_indices_buf;   

    struct DrawPipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    } draw_pl, gap_fill_pl;

    VkDescriptorSet     draw_desc{VK_NULL_HANDLE};
    VkDescriptorSet     gap_fill_desc{VK_NULL_HANDLE};
    DescriptorAllocator desc_allocator;

    struct BodyGPUInfo {
        uint32_t pixel_index{0};
        uint32_t body_w{0};
        uint32_t body_h{0};
        uint32_t sdf_offset{0};  
    };
    std::vector<BodyGPUInfo> body_draw_info;
    uint32_t next_pixel{0};
    uint32_t cap_bodies{0};
    uint32_t cap_pixels{0};

    AllocatedBuffer sdf_data_buf;
    uint32_t        next_sdf_float{0};
    uint32_t        cap_sdf_floats{0};

    struct DrawPipeline manifold_gen_pl;
    VkDescriptorSet     manifold_gen_desc{VK_NULL_HANDLE};
    AllocatedBuffer     edge_buf;      // CPU→GPU, MAX_GPU_EDGES  * sizeof(GPUEdge)
    AllocatedBuffer     contact_buf;   // GPU→CPU, MAX_GPU_CONTACTS * sizeof(GPUContact)
    AllocatedBuffer     counter_buf;   // GPU→CPU, sizeof(uint32_t)

    ResiduaEngine* engine_ref{nullptr};

    void init(ResiduaEngine* engine);

private:
    void grow_buffers(ResiduaEngine* engine, uint32_t needed_bodies, uint32_t needed_pixels);
    void grow_sdf_buffer(ResiduaEngine* engine, uint32_t needed_floats);

public:
    uint32_t add_body(ResiduaEngine* engine, RigidBody2 body);

    void remove_body(uint32_t idx);

    std::optional<uint32_t> body_at(glm::vec2 world_pos) const;

    void dispatch(VkCommandBuffer cmd, float dt);

    PhysicsStats get_stats();
};
