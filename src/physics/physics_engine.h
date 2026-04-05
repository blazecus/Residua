#pragma once

#include <glm/glm.hpp>
#include <optional>
#include <utility>
#include <vector>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include "sdf_generator.h"

class ResiduaEngine;

static constexpr uint32_t PHYSICS_WIDTH  = 480;
static constexpr uint32_t PHYSICS_HEIGHT = 270;

// ─── GPU struct mirrors (must match physics_types.glsl) ───────────────────────

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

struct PhysicsPixel {       // 32 bytes
    glm::vec2 total_force;
    float     force_moment;
    float     total_mass;
    float     inertia_moment;
    float     collision_count;
    glm::vec2 contact_normal;
};
static_assert(sizeof(PhysicsPixel) == 32);

struct ContactEntry {       // 40 bytes
    uint32_t other_id;
    float    weight;
    float    normal_x,      normal_y;
    float    centroid_x,    centroid_y;
    float    other_vel_x,   other_vel_y;
    float    other_inv_m;
    float    depth{0};
};
static_assert(sizeof(ContactEntry) == 40);

struct ContactPoint {       // 40 bytes
    uint32_t other_id{0};
    float    depth{0};
    float    pos_x{0},       pos_y{0};
    float    normal_x{0},    normal_y{0};
    float    other_vel_x{0}, other_vel_y{0};
    float    other_inv_m{0};
    float    _pad{0};
};
static_assert(sizeof(ContactPoint) == 40);

static constexpr uint32_t MAX_MANIFOLD_POINTS = 8;

// body_id encoding:
//   COLLISION_EMPTY  = 0
//   COLLISION_STATIC = 0xFFFFFFFF
//   dynamic body     = (tier_index << 24) | (slot + 1)
struct CollisionPixel {     // 32 bytes
    uint32_t body_id;
    uint32_t body_id2;
    float    vel_x,  vel_y;   // velocity of body_id  at this pixel
    float    vel_x2, vel_y2;  // velocity of body_id2 at this pixel
    float    mass;            // total_mass of body_id  (0 for static/empty)
    float    mass2;           // total_mass of body_id2 (0 for static/empty)
};
static_assert(sizeof(CollisionPixel) == 32);

struct LoadedBodyImage {
    uint32_t               width{}, height{};
    std::vector<glm::vec4> pixels;
    std::vector<float>     sdf;   // signed distance field, computed on load
};

LoadedBodyImage load_body_image(const char* path);

struct BodyTier {
    uint32_t body_w{}, body_h{}, body_pixels{};
    uint32_t tier_index{};
    uint32_t capacity{0};
    uint32_t active_count{0};
    uint32_t next_slot{0};

    AllocatedBuffer rb[2];               // RigidBody[]        capacity
    AllocatedBuffer physics_pixels;      // PhysicsPixel[]     capacity * body_pixels
    AllocatedBuffer pixel_colors;        // glm::vec4[]        capacity * body_pixels
    AllocatedBuffer body_l0;             // PhysicsPixel[]     capacity  (reduced, one per slot)
    AllocatedBuffer active_indices_buf;  // uint32_t[]         capacity
    AllocatedBuffer contact_entries;     // ContactEntry[]   capacity * body_pixels * 2
    AllocatedBuffer contact_counters;    // uint32_t[]       capacity  (atomic, reset each frame)
    AllocatedBuffer contact_manifold;    // ContactPoint[]   capacity * MAX_MANIFOLD_POINTS
    AllocatedBuffer body_sdf;            // float[]          capacity * body_pixels

    struct TierPipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    };

    TierPipeline pixel_physics_pl;
    TierPipeline pixel_reduction_pl;
    TierPipeline contact_manifold_pl;
    TierPipeline draw_pl;
    TierPipeline integrate_pl;

    VkDescriptorSet physics_desc[2]{};   // rb, physics_pixels, collision_pixels, active_indices, contact_entries, contact_counters, static_sdf
    VkDescriptorSet reduction_desc{};    // physics_pixels → body_l0
    VkDescriptorSet manifold_desc{};     // contact_entries, contact_counters → contact_manifold
    VkDescriptorSet draw_desc[2]{};      // reads rb[i], writes output_screen + collision_pixels
    VkDescriptorSet integrate_desc[2]{}; // rb[i] → rb[1-i] via body_l0 + contact_manifold

    DescriptorAllocator desc_allocator;

    std::vector<uint32_t> free_list;
    std::vector<uint32_t> active_slots;
};

struct PhysicsPipeline {

    AllocatedImage  output_screen;
    AllocatedBuffer static_collision;
    AllocatedBuffer collision_pixels;
    AllocatedBuffer static_sdf;          // float[] PHYSICS_WIDTH × PHYSICS_HEIGHT

    struct GapFillPipeline {
        VkPipeline            pipeline{VK_NULL_HANDLE};
        VkPipelineLayout      layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    } gap_fill_pl;
    VkDescriptorSet     gap_fill_desc{};
    DescriptorAllocator gap_fill_allocator;

    BodyTier tiers[3];  // 0 = 4×4,  1 = 16×16,  2 = 64×64

    uint32_t     frame_parity{0};
    SdfGenerator sdf_gen;

    void init(ResiduaEngine* engine, uint32_t initial_capacity);

    uint32_t add_body(
        ResiduaEngine*         engine,
        const LoadedBodyImage& img,
        uint32_t               tier_idx,
        glm::vec2              position,
        float                  rotation = 0.f
    );

    void remove_body(uint32_t tier_idx, uint32_t slot);

    std::optional<std::pair<uint32_t, uint32_t>> body_at(glm::vec2 world_pos) const;

    void upload_collision_layer(ResiduaEngine* engine, const char* path);

    void dispatch(VkCommandBuffer cmd, float dt);

private:
    void init_tier(ResiduaEngine* engine, uint32_t tier_idx, uint32_t initial_capacity);
    void grow_tier_buffers(ResiduaEngine* engine, uint32_t tier_idx);
    void update_tier_descriptors(VkDevice device, uint32_t tier_idx);
};
