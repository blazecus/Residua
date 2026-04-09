#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include "../renderer/vk_types.h"
#include "../renderer/vk_descriptors.h"
#include "lbvh.h"

class ResiduaEngine;

// ─── CPU mirrors of collision_types.glsl ─────────────────────────────────────

struct CandidatePair {   // 8 bytes
    uint32_t body_a;
    uint32_t body_b;
};
static_assert(sizeof(CandidatePair) == 8);

struct Contact {         // 64 bytes
    uint32_t  body_a;
    uint32_t  body_b;
    glm::vec2 normal;
    float     depth;
    uint32_t  valid;
    glm::vec2 r_a;
    glm::vec2 r_b;
    float     lambda_n;
    float     lambda_t;
    float     rho_n;
    float     rho_t;
    glm::vec2 _pad;
};
static_assert(sizeof(Contact) == 64);

// ─── Collision pipeline ───────────────────────────────────────────────────────

class CollisionPipeline {
public:
    AllocatedBuffer candidate_pairs_buf;  // CandidatePair[]  max_pairs
    AllocatedBuffer pair_count_buf;       // uint32           atomic counter
    AllocatedBuffer contacts_buf;         // Contact[]        max_contacts
    AllocatedBuffer contact_count_buf;    // uint32           atomic counter

    uint32_t max_pairs    {0};
    uint32_t max_contacts {0};

    // init() binds directly to the LBvhPipeline's buffers so descriptors stay valid
    // as long as the lbvh pipeline lives.
    void init(ResiduaEngine* engine, const LBvhPipeline& lbvh, uint32_t max_bodies);

    // Reset counters, dispatch broadphase then narrowphase.
    // lbvh.build() must have been called earlier this frame.
    void dispatch(VkCommandBuffer cmd, uint32_t body_count, const LBvhPipeline& lbvh);

    VkBuffer contacts_buffer()      const { return contacts_buf.buffer; }
    VkBuffer contact_count_buffer() const { return contact_count_buf.buffer; }

private:
    struct Pipeline {
        VkPipeline            pipeline    {VK_NULL_HANDLE};
        VkPipelineLayout      layout      {VK_NULL_HANDLE};
        VkDescriptorSetLayout desc_layout {VK_NULL_HANDLE};
    };

    Pipeline broadphase_pl;
    Pipeline narrowphase_pl;

    VkDescriptorSet broadphase_desc {};
    VkDescriptorSet narrowphase_desc{};

    DescriptorAllocatorGrowable desc_alloc;

    void init_pipelines(VkDevice device);
    void init_descriptors(VkDevice device, const LBvhPipeline& lbvh);
};
