#include "vertex_color.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <fmt/core.h>
#include <cassert>

// ─── Local helpers (mirrors collision.cpp pattern) ────────────────────────────

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                         VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("vertex_color: failed to load {}\n", spv);
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

// ─── VertexColorPipeline ──────────────────────────────────────────────────────

void VertexColorPipeline::init(ResiduaEngine* engine, const LBvhPipeline& lbvh,
                                const CollisionPipeline& collision,
                                uint32_t max_bodies_in)
{
    max_bodies    = max_bodies_in;
    _max_contacts = collision.max_contacts;

    const VkBufferUsageFlags ssbo_td =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    body_colors_buf    = engine->create_buffer(max_bodies * sizeof(uint32_t),
                                                ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);
    adj_count_buf      = engine->create_buffer(max_bodies * sizeof(uint32_t),
                                                ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);
    adj_list_buf       = engine->create_buffer(
                            max_bodies * MAX_BODY_ADJ * sizeof(uint32_t),
                            ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);
    proposed_color_buf = engine->create_buffer(max_bodies * sizeof(uint32_t),
                                                ssbo_td, VMA_MEMORY_USAGE_GPU_ONLY);

    VkDevice dev = engine->_device;
    init_pipelines(dev);
    init_descriptors(dev, lbvh, collision);
}

void VertexColorPipeline::init_pipelines(VkDevice dev)
{
    // build_adj: contacts, contact_count, bodies, adj_count, adj_list  (5 bindings)
    // push: max_contacts (uint)
    build_adj_pl.desc_layout = make_ssbo_layout(dev, 5);
    build_adj_pl.layout      = make_pipeline_layout(dev, build_adj_pl.desc_layout, sizeof(uint32_t));
    build_adj_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/color_build_adj.comp.spv", build_adj_pl.layout);

    // propose: body_colors, adj_count, adj_list, proposed_color  (4 bindings)
    // push: body_count (uint)
    propose_pl.desc_layout = make_ssbo_layout(dev, 4);
    propose_pl.layout      = make_pipeline_layout(dev, propose_pl.desc_layout, sizeof(uint32_t));
    propose_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/color_propose.comp.spv", propose_pl.layout);

    // commit: body_colors, proposed_color  (2 bindings)
    // push: body_count (uint)
    commit_pl.desc_layout = make_ssbo_layout(dev, 2);
    commit_pl.layout      = make_pipeline_layout(dev, commit_pl.desc_layout, sizeof(uint32_t));
    commit_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/color_commit.comp.spv", commit_pl.layout);
}

void VertexColorPipeline::init_descriptors(VkDevice dev, const LBvhPipeline& lbvh,
                                            const CollisionPipeline& collision)
{
    DescriptorAllocatorGrowable::PoolSizeRatio vc_ratios[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32.f }};
    desc_alloc.init(dev, 4, vc_ratios);

    auto wb = [&](DescriptorWriter& w, uint32_t binding, VkBuffer buf, size_t size) {
        w.write_buffer(binding, buf, size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    // build_adj descriptor
    {
        build_adj_desc = desc_alloc.allocate(dev, build_adj_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, collision.contacts_buf.buffer,      collision.max_contacts * sizeof(Contact));
        wb(w, 1, collision.contact_count_buf.buffer, sizeof(uint32_t));
        wb(w, 2, lbvh.bodies_buf.buffer,             lbvh.max_bodies * sizeof(RigidBody));
        wb(w, 3, adj_count_buf.buffer,               max_bodies * sizeof(uint32_t));
        wb(w, 4, adj_list_buf.buffer,                max_bodies * MAX_BODY_ADJ * sizeof(uint32_t));
        w.update_set(dev, build_adj_desc);
    }

    // propose descriptor
    {
        propose_desc = desc_alloc.allocate(dev, propose_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, body_colors_buf.buffer,    max_bodies * sizeof(uint32_t));
        wb(w, 1, adj_count_buf.buffer,      max_bodies * sizeof(uint32_t));
        wb(w, 2, adj_list_buf.buffer,       max_bodies * MAX_BODY_ADJ * sizeof(uint32_t));
        wb(w, 3, proposed_color_buf.buffer, max_bodies * sizeof(uint32_t));
        w.update_set(dev, propose_desc);
    }

    // commit descriptor
    {
        commit_desc = desc_alloc.allocate(dev, commit_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, body_colors_buf.buffer,    max_bodies * sizeof(uint32_t));
        wb(w, 1, proposed_color_buf.buffer, max_bodies * sizeof(uint32_t));
        w.update_set(dev, commit_desc);
    }
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void VertexColorPipeline::dispatch(VkCommandBuffer cmd, uint32_t body_count)
{
    assert(body_count >= 1);

    // ── 1. Reset per-frame buffers ────────────────────────────────────────────
    // Colors start as UINT_MAX (uncolored). adj_count starts at 0.
    vkCmdFillBuffer(cmd, body_colors_buf.buffer, 0,
                    max_bodies * sizeof(uint32_t), 0xFFFFFFFFu);
    vkCmdFillBuffer(cmd, adj_count_buf.buffer, 0,
                    max_bodies * sizeof(uint32_t), 0u);
    transfer_barrier(cmd, body_colors_buf.buffer);
    transfer_barrier(cmd, adj_count_buf.buffer);

    // ── 2. Build adjacency list ───────────────────────────────────────────────
    uint32_t max_c = _max_contacts;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_adj_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             build_adj_pl.layout, 0, 1, &build_adj_desc, 0, nullptr);
    vkCmdPushConstants(cmd, build_adj_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(uint32_t), &max_c);
    vkCmdDispatch(cmd, (max_c + 63u) / 64u, 1, 1);

    buf_barrier(cmd, adj_count_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    buf_barrier(cmd, adj_list_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // ── 3. Jones-Plassman propose / commit rounds ─────────────────────────────
    uint32_t groups = (body_count + 63u) / 64u;

    for (uint32_t pass = 0; pass < MAX_COLOR_PASSES; pass++) {
        // Propose
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, propose_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 propose_pl.layout, 0, 1, &propose_desc, 0, nullptr);
        vkCmdPushConstants(cmd, propose_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(uint32_t), &body_count);
        vkCmdDispatch(cmd, groups, 1, 1);

        buf_barrier(cmd, proposed_color_buf.buffer,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // Commit
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, commit_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 commit_pl.layout, 0, 1, &commit_desc, 0, nullptr);
        vkCmdPushConstants(cmd, commit_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(uint32_t), &body_count);
        vkCmdDispatch(cmd, groups, 1, 1);

        buf_barrier(cmd, body_colors_buf.buffer,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        buf_barrier(cmd, proposed_color_buf.buffer,
                    VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    }
}
