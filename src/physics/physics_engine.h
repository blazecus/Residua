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
static constexpr uint32_t MAX_GPU_CONTACTS = 16384;
static constexpr uint32_t MAX_GPU_BODIES   = 4096;
static constexpr uint32_t MAX_GPU_VERTS    = 65536;
static constexpr uint32_t MAX_GPU_PAIRS    = 16384;
static constexpr uint32_t MAX_COLOR_BODIES = 4096;
static constexpr uint32_t MAX_COLOR_ADJ    = 65536;
static constexpr uint32_t COLOR_JP_ITERS   = 16;

struct GPUBodyInfo {
    glm::vec2 pos;           
    float     angle;        
    uint32_t  edge_offset;  
    uint32_t  edge_count;  
    uint32_t  sdf_offset;  
    uint32_t  sdf_w; 
    uint32_t  sdf_h; 
    glm::vec2 com_local;  
};                 
static_assert(sizeof(GPUBodyInfo) == 40);

struct GPUPair {
    uint32_t body_a;
    uint32_t body_b;
};
static_assert(sizeof(GPUPair) == 8);

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
    glm::vec2 com_local;
    glm::vec2 _pad{0.f};
};
static_assert(sizeof(RigidBodyDrawGPU) == 64);

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
        uint32_t edge_offset{0};
        uint32_t edge_count{0};  
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
    AllocatedBuffer     body_info_buf;   
    AllocatedBuffer     body_verts_buf;  
    AllocatedBuffer     pair_buf;      
    AllocatedBuffer     contact_buf;
    AllocatedBuffer     counter_buf;
    uint32_t            next_vert{0};

    // ── Jones-Plassmann body coloring ─────────────────────────────────────────
    struct DrawPipeline coloring_pl;
    VkDescriptorSet     coloring_desc{VK_NULL_HANDLE};
    AllocatedBuffer     col_adj_offsets_buf;
    AllocatedBuffer     col_adj_list_buf;
    AllocatedBuffer     col_priorities_buf;
    AllocatedBuffer     col_active_buf;
    AllocatedBuffer     col_colors_buf;

    ResiduaEngine* engine_ref{nullptr};

    void init(ResiduaEngine* engine);

private:
    void grow_buffers(ResiduaEngine* engine, uint32_t needed_bodies, uint32_t needed_pixels);
    void grow_sdf_buffer(ResiduaEngine* engine, uint32_t needed_floats);

public:
    uint32_t add_body(ResiduaEngine* engine, RigidBody2 body);
    uint32_t add_static_rect(ResiduaEngine* engine, glm::vec2 center, float w, float h);

    void remove_body(uint32_t idx);
    void clear();

    std::optional<uint32_t> body_at(glm::vec2 world_pos) const;

    void dispatch(VkCommandBuffer cmd, float dt);

    PhysicsStats get_stats();
};
