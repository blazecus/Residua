#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include "lbvh.h"
#include "collision.h"
#include "vertex_color.h"

class ResiduaEngine;

// ─── CPU mirror of lbvh_types.glsl:InertiaTarget ─────────────────────────────
struct InertiaTarget {   // 16 bytes
    glm::vec2 pos  {0.f};
    float     rot  {0.f};
    float     _pad {0.f};
};
static_assert(sizeof(InertiaTarget) == 16);

// ─── Tunable solver parameters ────────────────────────────────────────────────
struct AvbdParams {
    glm::vec2 gravity  { 0.f,  200.f }; // pixel/s²  (y down in screen space: +y = down)
    float     mu       { 0.4f };         // Coulomb friction coefficient
    float     alpha    { 0.95f };        // λ warm-start scale  (αγ applied together)
    float     gamma    { 0.9f };         // k warm-start scale
    float     k_start  { 1000.f };       // minimum penalty stiffness
    float     beta     { 10.f };         // k adaptation rate
    uint32_t  n_iter   { 10 };           // solver iterations per timestep
};

// ─── AVBD pipeline ────────────────────────────────────────────────────────────
//
// Implements one full AVBD timestep (Algorithm 1):
//   predict → warmstart → [solve × n_colors] × n_iter → update_λ → integrate
//
// Reads / writes bodies_buf from LBvhPipeline (positions, velocities, etc.).
// Reads / writes contacts_buf from CollisionPipeline (λ, ρ warm-started here).
// Reads body_colors_buf from VertexColorPipeline to order the solve passes.

class AvbdPipeline {
public:
    AllocatedBuffer inertia_buf;   // InertiaTarget[]  max_bodies

    uint32_t max_bodies   {0};
    uint32_t max_contacts {0};

    // All three upstream pipelines must outlive this one.
    void init(ResiduaEngine*             engine,
              const LBvhPipeline&        lbvh,
              const CollisionPipeline&   collision,
              const VertexColorPipeline& vertex_color,
              uint32_t                   max_bodies);

    // Run one full timestep.  n_colors should match the number of colors actually
    // produced by VertexColorPipeline (or an upper bound); extra color passes are
    // no-ops (all threads return early when body_colors[i] != color).
    void dispatch(VkCommandBuffer    cmd,
                  uint32_t           body_count,
                  uint32_t           n_colors,
                  float              dt,
                  const AvbdParams&  params);

private:
    struct Pipeline {
        VkPipeline            pipeline    {VK_NULL_HANDLE};
        VkPipelineLayout      layout      {VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout {VK_NULL_HANDLE};
    };

    Pipeline predict_pl;
    Pipeline warmstart_pl;
    Pipeline solve_pl;
    Pipeline update_lambda_pl;
    Pipeline integrate_pl;

    VkDescriptorSet predict_desc       {};
    VkDescriptorSet warmstart_desc     {};
    VkDescriptorSet solve_desc         {};
    VkDescriptorSet update_lambda_desc {};
    VkDescriptorSet integrate_desc     {};

    DescriptorAllocatorGrowable desc_alloc;

    // Cached buffer handles for synchronization in dispatch()
    VkBuffer _bodies_buf   {VK_NULL_HANDLE};
    VkBuffer _contacts_buf {VK_NULL_HANDLE};
    VkBuffer _ct_count_buf {VK_NULL_HANDLE};

    void init_pipelines(VkDevice device);
    void init_descriptors(VkDevice                   device,
                          const LBvhPipeline&        lbvh,
                          const CollisionPipeline&   collision,
                          const VertexColorPipeline& vertex_color);
};
