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
    uint32_t  _pad{0};
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

    struct BodyDrawInfo {
        uint32_t pixel_index{0};
        uint32_t body_w{0};
        uint32_t body_h{0};
    };
    std::vector<BodyDrawInfo> body_draw_info;
    uint32_t next_pixel{0};   
    uint32_t cap_bodies{0};   
    uint32_t cap_pixels{0};   

    void init(ResiduaEngine* engine);

private:
    void grow_buffers(ResiduaEngine* engine, uint32_t needed_bodies, uint32_t needed_pixels);

public:
    uint32_t add_body(ResiduaEngine* engine, RigidBody2 body);

    void remove_body(uint32_t idx);

    std::optional<uint32_t> body_at(glm::vec2 world_pos) const;

    void dispatch(VkCommandBuffer cmd, float dt);

    PhysicsStats get_stats();
};
