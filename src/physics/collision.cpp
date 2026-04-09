#include "collision.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <fmt/core.h>
#include <cassert>

// ─── Helpers (duplicated pattern from lbvh.cpp) ───────────────────────────────

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                         VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("collision: failed to load {}\n", spv);
        return VK_NULL_HANDLE;
    }
    VkPipelineShaderStageCreateInfo stage {
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = mod,
        .pName  = "main",
    };
    VkComputePipelineCreateInfo info {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = stage,
        .layout = layout,
    };
    VkPipeline pl;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pl));
    vkDestroyShaderModule(device, mod, nullptr);
    return pl;
}

static VkDescriptorSetLayout make_ssbo_layout(VkDevice device, uint32_t count)
{
    std::vector<VkDescriptorSetLayoutBinding> b(count);
    for (uint32_t i = 0; i < count; i++)
        b[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo ci {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count,
        .pBindings    = b.data(),
    };
    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
    return layout;
}

static VkPipelineLayout make_pipeline_layout(VkDevice device,
                                              VkDescriptorSetLayout desc,
                                              uint32_t push_size)
{
    VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size };
    VkPipelineLayoutCreateInfo ci {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &desc,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pc,
    };
    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &ci, nullptr, &layout));
    return layout;
}

static void buf_barrier(VkCommandBuffer cmd, VkBuffer buf,
                         VkAccessFlags2 src, VkAccessFlags2 dst)
{
    VkBufferMemoryBarrier2 b {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = src,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = dst,
        .buffer        = buf,
        .offset        = 0,
        .size          = VK_WHOLE_SIZE,
    };
    VkDependencyInfo dep {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &b,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

static void transfer_barrier(VkCommandBuffer cmd, VkBuffer buf)
{
    VkBufferMemoryBarrier2 b {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .buffer        = buf,
        .offset        = 0,
        .size          = VK_WHOLE_SIZE,
    };
    VkDependencyInfo dep {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &b,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ─── CollisionPipeline ────────────────────────────────────────────────────────

void CollisionPipeline::init(ResiduaEngine* engine, const LBvhPipeline& lbvh,
                              uint32_t max_bodies)
{
    max_pairs    = max_bodies * 16;   // each body overlaps at most ~16 others
    max_contacts = max_pairs * 2;     // at most 2 contacts per pair

    const VkBufferUsageFlags ssbo_td =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    candidate_pairs_buf = engine->create_buffer(
        max_pairs    * sizeof(CandidatePair), ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);
    pair_count_buf      = engine->create_buffer(
        sizeof(uint32_t), ssbo_td | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    contacts_buf        = engine->create_buffer(
        max_contacts * sizeof(Contact),       ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);
    contact_count_buf   = engine->create_buffer(
        sizeof(uint32_t), ssbo_td | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    VkDevice dev = engine->_device;
    init_pipelines(dev);
    init_descriptors(dev, lbvh);
}

void CollisionPipeline::init_pipelines(VkDevice dev)
{
    // Broadphase: aabbs, nodes, pairs, pair_count  (4 bindings)
    struct BroadPC { uint32_t n, max_pairs; };
    broadphase_pl.desc_layout = make_ssbo_layout(dev, 4);
    broadphase_pl.layout      = make_pipeline_layout(dev, broadphase_pl.desc_layout, sizeof(BroadPC));
    broadphase_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/collision_broadphase.comp.spv", broadphase_pl.layout);

    // Narrowphase: bodies, shapes, pairs, pair_count, contacts, contact_count  (6 bindings)
    narrowphase_pl.desc_layout = make_ssbo_layout(dev, 6);
    narrowphase_pl.layout      = make_pipeline_layout(dev, narrowphase_pl.desc_layout, sizeof(uint32_t));
    narrowphase_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/collision_narrowphase.comp.spv", narrowphase_pl.layout);
}

void CollisionPipeline::init_descriptors(VkDevice dev, const LBvhPipeline& lbvh)
{
    DescriptorAllocatorGrowable::PoolSizeRatio col_ratios[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32.f }};
    desc_alloc.init(dev, 4, col_ratios);

    auto wb = [&](DescriptorWriter& w, uint32_t binding, VkBuffer buf, size_t size) {
        w.write_buffer(binding, buf, size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    const uint32_t max_nodes = 2 * lbvh.max_bodies - 1;

    // Broadphase descriptor set
    {
        broadphase_desc = desc_alloc.allocate(dev, broadphase_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.aabbs_buf.buffer,          lbvh.max_bodies * sizeof(AABB));
        wb(w, 1, lbvh.bvh_nodes_buf.buffer,      max_nodes       * sizeof(BVHNode));
        wb(w, 2, candidate_pairs_buf.buffer,      max_pairs       * sizeof(CandidatePair));
        wb(w, 3, pair_count_buf.buffer,           sizeof(uint32_t));
        w.update_set(dev, broadphase_desc);
    }

    // Narrowphase descriptor set
    {
        narrowphase_desc = desc_alloc.allocate(dev, narrowphase_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.bodies_buf.buffer,          lbvh.max_bodies * sizeof(RigidBody));
        wb(w, 1, lbvh.shapes_buf.buffer,          lbvh.max_bodies * sizeof(CollisionShape));
        wb(w, 2, candidate_pairs_buf.buffer,      max_pairs       * sizeof(CandidatePair));
        wb(w, 3, pair_count_buf.buffer,           sizeof(uint32_t));
        wb(w, 4, contacts_buf.buffer,             max_contacts    * sizeof(Contact));
        wb(w, 5, contact_count_buf.buffer,        sizeof(uint32_t));
        w.update_set(dev, narrowphase_desc);
    }
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void CollisionPipeline::dispatch(VkCommandBuffer cmd, uint32_t n,
                                  const LBvhPipeline& /*lbvh*/)
{
    assert(n >= 2);

    // ── 1. Reset atomic counters ──────────────────────────────────────────────
    vkCmdFillBuffer(cmd, pair_count_buf.buffer,    0, sizeof(uint32_t), 0u);
    vkCmdFillBuffer(cmd, contact_count_buf.buffer, 0, sizeof(uint32_t), 0u);
    transfer_barrier(cmd, pair_count_buf.buffer);
    transfer_barrier(cmd, contact_count_buf.buffer);

    // ── 2. Broadphase ─────────────────────────────────────────────────────────
    struct BroadPC { uint32_t n, max_pairs; } bpc { n, max_pairs };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, broadphase_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             broadphase_pl.layout, 0, 1, &broadphase_desc, 0, nullptr);
    vkCmdPushConstants(cmd, broadphase_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(bpc), &bpc);
    vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);

    // Pairs and count must be visible to narrowphase.
    buf_barrier(cmd, candidate_pairs_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    buf_barrier(cmd, pair_count_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // ── 3. Narrowphase ────────────────────────────────────────────────────────
    // Dispatch max_pairs threads; threads with pair_idx >= pair_count return early.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, narrowphase_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             narrowphase_pl.layout, 0, 1, &narrowphase_desc, 0, nullptr);
    vkCmdPushConstants(cmd, narrowphase_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(uint32_t), &max_contacts);
    vkCmdDispatch(cmd, (max_pairs + 63) / 64, 1, 1);

    buf_barrier(cmd, contacts_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    buf_barrier(cmd, contact_count_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}
