#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"

class ResiduaEngine;

static constexpr uint32_t PHYSICS_WIDTH  = 480;
static constexpr uint32_t PHYSICS_HEIGHT = 270;

// ─── CPU mirrors of lbvh_types.glsl ──────────────────────────────────────────

struct AABB {                  // 16 bytes
    glm::vec2 min_pt;
    glm::vec2 max_pt;
};
static_assert(sizeof(AABB) == 16);

struct BVHNode {               // 32 bytes
    glm::vec2 aabb_min;
    glm::vec2 aabb_max;
    int32_t   left      {-1};
    int32_t   right     {-1};
    int32_t   parent    {-1};
    int32_t   object_id {-1};
};
static_assert(sizeof(BVHNode) == 32);

// New AVBD rigid body — replaces the old pixel-based RigidBody.
struct RigidBody {             // 48 bytes
    glm::vec2 position    {0.f};
    glm::vec2 velocity    {0.f};
    float     rotation    {0.f};
    float     ang_vel     {0.f};
    float     inv_mass    {0.f};   // 1/mass; 0 = static
    float     inv_inertia {0.f};   // 1/I_com; 0 = static
    glm::vec2 prev_pos    {0.f};
    float     prev_rot    {0.f};
    float     _pad        {0.f};
};
static_assert(sizeof(RigidBody) == 48);

// Collision shape (polygon from marching-squares → Douglas-Peucker).
// Vertices are in local space (relative to body CoM, before rotation).
static constexpr uint32_t MAX_SHAPE_VERTS = 128;

struct CollisionShape {        // 2064 bytes
    glm::vec2 verts[MAX_SHAPE_VERTS];    // local-space polygon vertices
    glm::vec2 normals[MAX_SHAPE_VERTS];  // outward edge normal for edge i→(i+1)%count, local space
    uint32_t  count {0};
    float     _pad[3] {};
};
static_assert(sizeof(CollisionShape) == 2064);

// ─── LBVH pipeline ───────────────────────────────────────────────────────────

class LBvhPipeline {
public:
    // GPU buffers
    AllocatedBuffer bodies_buf;       // RigidBody[]        max_bodies
    AllocatedBuffer shapes_buf;       // CollisionShape[]   max_bodies
    AllocatedBuffer aabbs_buf;        // AABB[]             max_bodies
    AllocatedBuffer morton_buf;       // uint32[]           max_bodies  (unsorted / sorted after even passes)
    AllocatedBuffer indices_buf;      // uint32[]           max_bodies  (body indices)
    AllocatedBuffer morton_tmp_buf;   // uint32[]           max_bodies  (ping-pong)
    AllocatedBuffer indices_tmp_buf;  // uint32[]           max_bodies  (ping-pong)
    AllocatedBuffer block_hist_buf;   // uint32[]           max_blocks * 256
    AllocatedBuffer bvh_nodes_buf;    // BVHNode[]          2 * max_bodies - 1
    AllocatedBuffer refit_flags_buf;  // int32[]            2 * max_bodies - 1

    uint32_t max_bodies{0};
    uint32_t body_count{0};   // active bodies this frame

    void init(ResiduaEngine* engine, uint32_t max_bodies);

    // Write a RigidBody and its CollisionShape into the CPU-visible mapped buffers
    // at the given slot.  Call before build() each frame (or on spawn).
    void set_body(uint32_t slot, const RigidBody& body, const CollisionShape& shape);

    // Convenience: build a CollisionShape from a vector of local-space vertices.
    // Truncates silently to MAX_SHAPE_VERTS.
    static CollisionShape make_shape(const std::vector<glm::vec2>& verts);

    // Record all BVH build passes into cmd for body_count active bodies.
    // bodies_buf / shapes_buf must be filled before calling.
    void build(VkCommandBuffer cmd, uint32_t body_count);

    // Expose buffers for downstream passes.
    VkBuffer nodes_buffer()  const { return bvh_nodes_buf.buffer; }
    VkBuffer shapes_buffer() const { return shapes_buf.buffer; }

private:
    struct Pipeline {
        VkPipeline            pipeline   {VK_NULL_HANDLE};
        VkPipelineLayout      layout     {VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout{VK_NULL_HANDLE};
    };

    Pipeline morton_pl;
    Pipeline radix_count_pl;
    Pipeline radix_scan_pl;
    Pipeline radix_scatter_pl;
    Pipeline build_pl;
    Pipeline refit_pl;

    // Descriptor sets
    VkDescriptorSet morton_desc     {};
    VkDescriptorSet radix_count_desc[2]{};   // [pass%2]: reads real vs temp key buffer
    VkDescriptorSet radix_scan_desc {};
    VkDescriptorSet radix_scatter_desc[2]{}; // [pass%2]: scatters real→temp vs temp→real
    VkDescriptorSet build_desc      {};
    VkDescriptorSet refit_desc      {};

    DescriptorAllocatorGrowable desc_alloc;

    void init_pipelines(VkDevice device);
    void init_descriptors(VkDevice device);
};
