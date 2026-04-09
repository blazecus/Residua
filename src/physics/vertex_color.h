#pragma once

#include <cstdint>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include "lbvh.h"
#include "collision.h"

class ResiduaEngine;

static constexpr uint32_t MAX_BODY_ADJ    = 64;  // adjacency slots per body
static constexpr uint32_t MAX_COLOR_PASSES = 16;  // Jones-Plassman iteration cap

// ─── VertexColorPipeline ──────────────────────────────────────────────────────
//
// Assigns a color index to every dynamic rigid body such that no two bodies
// sharing a contact constraint have the same color.  Same-color bodies can be
// solved in parallel in the AVBD solver without data races.
//
// Algorithm: Jones-Plassman parallel greedy graph coloring.
//   1. build_adj   — one thread per contact → adjacency list
//   2. × N passes:
//        propose   — highest-priority uncolored body in each clique proposes a color
//        commit    — accept all proposals (no conflicts by construction)
//
// body_colors[i] == UINT_MAX means the body is static and received no color.

class VertexColorPipeline {
public:
    AllocatedBuffer body_colors_buf;    // uint32[]  max_bodies  output
    AllocatedBuffer adj_count_buf;      // uint32[]  max_bodies
    AllocatedBuffer adj_list_buf;       // uint32[]  max_bodies * MAX_BODY_ADJ
    AllocatedBuffer proposed_color_buf; // uint32[]  max_bodies  scratch

    uint32_t max_bodies {0};

    // init() captures buffer references from lbvh and collision into descriptors;
    // both pipelines must outlive this one.
    void init(ResiduaEngine* engine, const LBvhPipeline& lbvh,
              const CollisionPipeline& collision, uint32_t max_bodies);

    // Reset, build adjacency, then run MAX_COLOR_PASSES propose/commit rounds.
    // collision.dispatch() must have been called earlier this frame.
    void dispatch(VkCommandBuffer cmd, uint32_t body_count);

    VkBuffer body_colors_buffer() const { return body_colors_buf.buffer; }

private:
    struct Pipeline {
        VkPipeline            pipeline    {VK_NULL_HANDLE};
        VkPipelineLayout      layout      {VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout {VK_NULL_HANDLE};
    };

    Pipeline build_adj_pl;
    Pipeline propose_pl;
    Pipeline commit_pl;

    VkDescriptorSet build_adj_desc {};
    VkDescriptorSet propose_desc   {};
    VkDescriptorSet commit_desc    {};

    DescriptorAllocatorGrowable desc_alloc;

    uint32_t _max_contacts {0};

    void init_pipelines(VkDevice device);
    void init_descriptors(VkDevice device, const LBvhPipeline& lbvh,
                          const CollisionPipeline& collision);
};
