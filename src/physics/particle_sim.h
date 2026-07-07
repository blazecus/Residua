#pragma once

#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x2.hpp>
#include <cstdint>

class ResiduaEngine;

static constexpr uint32_t MAX_PARTICLES    = 1 << 17;      // 131 072
static constexpr uint32_t PARTICLE_ITERS   = 3;
static constexpr uint32_t FIXED_POINT_MULT = 1 << 16;

enum class ParticleMaterial : int {
    Liquid  = 0,
    Elastic = 1,
    Sand    = 2,
    Visco   = 3,
};

struct Particle {
    glm::vec2 position             {};
    glm::vec2 displacement         {};               
    glm::mat2 deformation_gradient = glm::mat2(1.f);
    glm::mat2 deformation_displacement = glm::mat2(0.f); 
    float     liquid_density   { 1.0f };
    float     mass             { 1.f };
    float     material         { 0.f };  
    float     volume           { 1.f };
    float     log_jp           { 0.f };  
    float     enabled          { 1.f };  
    glm::vec2 _pad             {};
    glm::vec4 color            { 1.f, 1.f, 1.f, 1.f };
};
static_assert(sizeof(Particle) == 96);

struct G2P2GPushConstants {
    glm::uvec2 grid_size;   
    uint32_t   particle_count;
    uint32_t   iteration;
    uint32_t   iteration_count;
    float      delta_time;
    float      gravity;
    float      border_friction;
    float      liquid_relaxation;
    float      liquid_viscosity;
    float      elastic_relaxation;
    float      elasticity_ratio;
    float      friction_angle;
    float      plasticity;
    uint32_t   fixed_point_mult;
    uint32_t   cell_size;
    glm::vec2  camera_offset;    // world → screen: screen_pos = world_pos - camera_offset
};
static_assert(sizeof(G2P2GPushConstants) == 72);

struct ParticleDrawPushConstants {
    glm::uvec2 grid_size;
    uint32_t   particle_count;
    uint32_t   cell_size;
    glm::vec2  camera_offset;
};
static_assert(sizeof(ParticleDrawPushConstants) == 24);

struct GridBCPushConstants {
    glm::uvec2 grid_size;
    uint32_t   fixed_point_mult;
    float      gravity_impulse;  
    float      dt;  
    uint32_t   cell_size;
    uint32_t   _pad;
};
static_assert(sizeof(GridBCPushConstants) == 28);

class ParticleSim {
public:
    struct Settings {
        float    gravity             = 100.f / 360.f;
        float    border_friction     = 0.0f;
        float    liquid_relaxation   = 0.1f;
        float    liquid_viscosity    = 0.0f;
        float    elastic_relaxation  = 0.4f;
        float    elasticity_ratio    = 0.9f;
        float    friction_angle      = 45.0f;
        float    plasticity          = 0.5f;
        uint32_t cell_size           = 4u; 
    } settings;

    void init(ResiduaEngine* engine, VkImageView output_view);
    void setup_collision_image(VkImageView vel_image_view);

    void add_particle(const Particle& p);
    void clear_particles();
    void destroy_particles_in_radius(glm::vec2 center, float radius);

    void dispatch_physics(VkCommandBuffer cmd, float dt);

    void dispatch_render(VkCommandBuffer cmd);

    uint32_t  particle_count() const { return particle_count_; }

    glm::vec2 camera_offset { 0.f, 0.f };

private:
    struct Pipeline {
        VkPipeline            pipeline    { VK_NULL_HANDLE };
        VkPipelineLayout      layout      { VK_NULL_HANDLE };
        VkDescriptorSetLayout desc_layout { VK_NULL_HANDLE };
    };

    ResiduaEngine* engine_ref  { nullptr };
    uint32_t       particle_count_ { 0 };

    AllocatedBuffer particle_buf;  
    AllocatedBuffer grid_buf[2];   

    Pipeline        g2p2g_pl;
    VkDescriptorSet g2p2g_desc[2] {};
    VkDescriptorSetLayout g2p2g_vel_layout { VK_NULL_HANDLE };
    VkDescriptorSet       g2p2g_vel_desc   { VK_NULL_HANDLE }; 

    Pipeline        bc_pl;
    VkDescriptorSet bc_desc[2] {};

    Pipeline        draw_pl;
    VkDescriptorSet draw_desc { VK_NULL_HANDLE };

    DescriptorAllocator desc_alloc;
};
