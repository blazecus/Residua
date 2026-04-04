#include "physics_engine.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_images.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <stb_image.h>
#include <fmt/core.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>

// ─── Helpers ─────────────────────────────────────────────────────────────────

LoadedBodyImage load_body_image(const char* path) {
    int w, h, channels;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    assert(data && "load_body_image: failed to open file");

    LoadedBodyImage result;
    result.width  = (uint32_t)w;
    result.height = (uint32_t)h;
    result.pixels.resize(w * h);
    for (int i = 0; i < w * h; i++) {
        result.pixels[i] = glm::vec4(
            data[i * 4 + 0] / 255.f,
            data[i * 4 + 1] / 255.f,
            data[i * 4 + 2] / 255.f,
            data[i * 4 + 3] / 255.f
        );
    }
    stbi_image_free(data);
    return result;
}

static VkPipeline create_compute_pipeline(
    VkDevice                    device,
    const char*                 spv_path,
    VkPipelineLayout            layout,
    const VkSpecializationInfo* spec = nullptr
) {
    VkShaderModule module;
    if (!vkutil::load_shader_module(spv_path, device, &module)) {
        fmt::print("create_compute_pipeline: failed to load {}\n", spv_path);
        return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stage {
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
        .module              = module,
        .pName               = "main",
        .pSpecializationInfo = spec,
    };
    VkComputePipelineCreateInfo info {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = stage,
        .layout = layout,
    };

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

static void buffer_barrier(VkCommandBuffer cmd, VkBuffer buffer,
    VkAccessFlags2 src_access, VkAccessFlags2 dst_access)
{
    VkBufferMemoryBarrier2 barrier {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = src_access,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = dst_access,
        .buffer        = buffer,
        .offset        = 0,
        .size          = VK_WHOLE_SIZE,
    };
    VkDependencyInfo dep {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

void PhysicsPipeline::update_tier_descriptors(VkDevice device, uint32_t tier_idx) {
    BodyTier& t = tiers[tier_idx];

    const size_t rb_size               = t.capacity * sizeof(RigidBody);
    const size_t phys_pixels_size      = t.capacity * t.body_pixels * sizeof(PhysicsPixel);
    const size_t pixel_colors_size     = t.capacity * t.body_pixels * sizeof(glm::vec4);
    const size_t body_l0_size          = t.capacity * sizeof(PhysicsPixel);
    const size_t active_idx_size       = t.capacity * sizeof(uint32_t);
    const size_t collision_size        = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);
    const size_t contact_entries_size  = t.capacity * t.body_pixels * 2 * sizeof(ContactEntry);
    const size_t contact_counters_size = t.capacity * sizeof(uint32_t);
    const size_t contact_agg_size      = t.capacity * MAX_CONTACTS_PER_BODY * sizeof(ContactAggregate);

    for (int i = 0; i < 2; i++) {
        // physics_desc: rb, physics_pixels, collision_pixels, active_indices, contact_entries, contact_counters
        {
            DescriptorWriter w;
            w.write_buffer(0, t.rb[i].buffer,              rb_size,               0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, t.physics_pixels.buffer,     phys_pixels_size,      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, collision_pixels.buffer,     collision_size,        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(3, t.active_indices_buf.buffer, active_idx_size,       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(4, t.contact_entries.buffer,    contact_entries_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(5, t.contact_counters.buffer,   contact_counters_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, t.physics_desc[i]);
        }
        // draw_desc: rb, pixel_colors, physics_pixels, output_screen, collision_pixels, active_indices
        {
            DescriptorWriter w;
            w.write_buffer(0, t.rb[i].buffer,              rb_size,           0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, t.pixel_colors.buffer,       pixel_colors_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, t.physics_pixels.buffer,     phys_pixels_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_image (3, output_screen.imageView,     VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_buffer(4, collision_pixels.buffer,     collision_size,    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(5, t.active_indices_buf.buffer, active_idx_size,   0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, t.draw_desc[i]);
        }
        // integrate_desc: rb src, rb dst, body_l0, active_indices, contact_agg
        {
            DescriptorWriter w;
            w.write_buffer(0, t.rb[i].buffer,              rb_size,          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, t.rb[1-i].buffer,            rb_size,          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, t.body_l0.buffer,            body_l0_size,     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(3, t.active_indices_buf.buffer, active_idx_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(4, t.contact_agg.buffer,        contact_agg_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, t.integrate_desc[i]);
        }
    }
    // reduction_desc: physics_pixels → body_l0
    {
        DescriptorWriter w;
        w.write_buffer(0, t.physics_pixels.buffer,     phys_pixels_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, t.body_l0.buffer,            body_l0_size,     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(2, t.active_indices_buf.buffer, active_idx_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, t.reduction_desc);
    }
    // contact_desc: contact_entries, contact_counters → contact_agg, active_indices
    {
        DescriptorWriter w;
        w.write_buffer(0, t.contact_entries.buffer,    contact_entries_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, t.contact_counters.buffer,   contact_counters_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(2, t.contact_agg.buffer,        contact_agg_size,      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(3, t.active_indices_buf.buffer, active_idx_size,       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, t.contact_desc);
    }
}

void PhysicsPipeline::grow_tier_buffers(ResiduaEngine* engine, uint32_t tier_idx) {
    BodyTier& t = tiers[tier_idx];
    const uint32_t new_capacity = t.capacity * 2;
    fmt::print("PhysicsPipeline tier {}: growing {} → {} slots\n", tier_idx, t.capacity, new_capacity);

    vkDeviceWaitIdle(engine->_device);

    const size_t new_rb_size               = new_capacity * sizeof(RigidBody);
    const size_t new_phys_pixels_size      = new_capacity * t.body_pixels * sizeof(PhysicsPixel);
    const size_t new_pixel_colors_size     = new_capacity * t.body_pixels * sizeof(glm::vec4);
    const size_t new_body_l0_size          = new_capacity * sizeof(PhysicsPixel);
    const size_t new_active_idx_size       = new_capacity * sizeof(uint32_t);
    const size_t new_contact_entries_size  = new_capacity * t.body_pixels * 2 * sizeof(ContactEntry);
    const size_t new_contact_counters_size = new_capacity * sizeof(uint32_t);
    const size_t new_contact_agg_size      = new_capacity * MAX_CONTACTS_PER_BODY * sizeof(ContactAggregate);

    AllocatedBuffer new_rb0             = engine->create_buffer(new_rb_size,               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_rb1             = engine->create_buffer(new_rb_size,               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_phys_pixels     = engine->create_buffer(new_phys_pixels_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_pixel_colors    = engine->create_buffer(new_pixel_colors_size,     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_body_l0         = engine->create_buffer(new_body_l0_size,          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_active_idx      = engine->create_buffer(new_active_idx_size,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_contact_entries = engine->create_buffer(new_contact_entries_size,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    AllocatedBuffer new_contact_counters= engine->create_buffer(new_contact_counters_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    AllocatedBuffer new_contact_agg     = engine->create_buffer(new_contact_agg_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    memcpy(new_rb0.info.pMappedData,          t.rb[0].info.pMappedData,          t.capacity * sizeof(RigidBody));
    memcpy(new_rb1.info.pMappedData,          t.rb[1].info.pMappedData,          t.capacity * sizeof(RigidBody));
    memcpy(new_phys_pixels.info.pMappedData,  t.physics_pixels.info.pMappedData, t.capacity * t.body_pixels * sizeof(PhysicsPixel));
    memcpy(new_pixel_colors.info.pMappedData, t.pixel_colors.info.pMappedData,   t.capacity * t.body_pixels * sizeof(glm::vec4));
    memcpy(new_body_l0.info.pMappedData,      t.body_l0.info.pMappedData,        t.capacity * sizeof(PhysicsPixel));
    memcpy(new_active_idx.info.pMappedData,   t.active_slots.data(),             t.active_count * sizeof(uint32_t));

    engine->destroy_buffer(t.rb[0]);
    engine->destroy_buffer(t.rb[1]);
    engine->destroy_buffer(t.physics_pixels);
    engine->destroy_buffer(t.pixel_colors);
    engine->destroy_buffer(t.body_l0);
    engine->destroy_buffer(t.active_indices_buf);
    engine->destroy_buffer(t.contact_entries);
    engine->destroy_buffer(t.contact_counters);
    engine->destroy_buffer(t.contact_agg);

    t.rb[0]              = new_rb0;
    t.rb[1]              = new_rb1;
    t.physics_pixels     = new_phys_pixels;
    t.pixel_colors       = new_pixel_colors;
    t.body_l0            = new_body_l0;
    t.active_indices_buf = new_active_idx;
    t.contact_entries    = new_contact_entries;
    t.contact_counters   = new_contact_counters;
    t.contact_agg        = new_contact_agg;
    t.capacity           = new_capacity;

    update_tier_descriptors(engine->_device, tier_idx);
}

void PhysicsPipeline::init_tier(ResiduaEngine* engine, uint32_t tier_idx, uint32_t initial_capacity) {
    static constexpr uint32_t dims[3][2] = { {4, 4}, {16, 16}, {64, 64} };

    BodyTier& t    = tiers[tier_idx];
    t.tier_index   = tier_idx;
    t.body_w       = dims[tier_idx][0];
    t.body_h       = dims[tier_idx][1];
    t.body_pixels  = t.body_w * t.body_h;
    t.capacity     = initial_capacity;

    VkDevice device = engine->_device;

    const size_t rb_size               = t.capacity * sizeof(RigidBody);
    const size_t phys_pixels_size      = t.capacity * t.body_pixels * sizeof(PhysicsPixel);
    const size_t pixel_colors_size     = t.capacity * t.body_pixels * sizeof(glm::vec4);
    const size_t body_l0_size          = t.capacity * sizeof(PhysicsPixel);
    const size_t active_idx_size       = t.capacity * sizeof(uint32_t);
    const size_t contact_entries_size  = t.capacity * t.body_pixels * 2 * sizeof(ContactEntry);
    const size_t contact_counters_size = t.capacity * sizeof(uint32_t);
    const size_t contact_agg_size      = t.capacity * MAX_CONTACTS_PER_BODY * sizeof(ContactAggregate);

    t.rb[0]              = engine->create_buffer(rb_size,               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.rb[1]              = engine->create_buffer(rb_size,               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.physics_pixels     = engine->create_buffer(phys_pixels_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.pixel_colors       = engine->create_buffer(pixel_colors_size,     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.body_l0            = engine->create_buffer(body_l0_size,          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.active_indices_buf = engine->create_buffer(active_idx_size,       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    t.contact_entries    = engine->create_buffer(contact_entries_size,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_GPU_ONLY);
    t.contact_counters   = engine->create_buffer(contact_counters_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    t.contact_agg        = engine->create_buffer(contact_agg_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                                    VMA_MEMORY_USAGE_GPU_ONLY);

    // Spec constants: BODY_W, BODY_H
    uint32_t spec_data[] = { t.body_w, t.body_h };
    VkSpecializationMapEntry spec_entries[] = {
        { 0, 0,                sizeof(uint32_t) },
        { 1, sizeof(uint32_t), sizeof(uint32_t) },
    };
    VkSpecializationInfo spec_info {
        .mapEntryCount = 2,
        .pMapEntries   = spec_entries,
        .dataSize      = sizeof(spec_data),
        .pData         = spec_data,
    };

    DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   4.f },
    };
    t.desc_allocator.init_pool(device, 9, pool_ratios);

    // ── pixel_physics ─────────────────────────────────────────────────────
    {
        auto& pl = t.pixel_physics_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // physics_pixels
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_pixels
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_entries (write)
        b.add_binding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_counters (atomic write)
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) + gravity(f32) + physics_width(u32) + physics_height(u32) + tier_index(u32) = 20 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4 + sizeof(float) };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/pixel_physics.comp.spv", pl.layout, &spec_info);

        for (int i = 0; i < 2; i++)
            t.physics_desc[i] = t.desc_allocator.allocate(device, pl.desc_layout);
    }

    // ── pixel_reduction ───────────────────────────────────────────────────
    {
        auto& pl = t.pixel_reduction_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // src (physics_pixels)
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // dst (body_l0)
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) = 4 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/pixel_reduction.comp.spv", pl.layout, &spec_info);

        t.reduction_desc = t.desc_allocator.allocate(device, pl.desc_layout);
    }

    // ── contact_reduction ─────────────────────────────────────────────────
    {
        auto& pl = t.contact_reduction_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_entries (read)
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_counters (read)
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_agg (write)
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) = 4 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/contact_reduction.comp.spv", pl.layout, &spec_info);

        t.contact_desc = t.desc_allocator.allocate(device, pl.desc_layout);
    }

    // ── rigidbody_draw ────────────────────────────────────────────────────
    {
        auto& pl = t.draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // physics_pixels (mass check)
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_image
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_pixels (write)
        b.add_binding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) + physics_width(u32) + tier_index(u32) = 12 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 3 };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/rigidbody_draw.comp.spv", pl.layout, &spec_info);

        for (int i = 0; i < 2; i++)
            t.draw_desc[i] = t.desc_allocator.allocate(device, pl.desc_layout);
    }

    // ── rigidbody_integrate ───────────────────────────────────────────────
    {
        auto& pl = t.integrate_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb src
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb dst
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // body_l0
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_agg (read)
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) + dt(f32) + restitution(f32) = 12 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) + sizeof(float) * 2 };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/rigidbody_integrate.comp.spv", pl.layout, &spec_info);

        for (int i = 0; i < 2; i++)
            t.integrate_desc[i] = t.desc_allocator.allocate(device, pl.desc_layout);
    }

    update_tier_descriptors(device, tier_idx);

    engine->_mainDeletionQueue.push_function([this, engine, device, tier_idx]() {
        BodyTier& tier = tiers[tier_idx];

        vkDestroyPipeline(device, tier.pixel_physics_pl.pipeline,      nullptr);
        vkDestroyPipeline(device, tier.pixel_reduction_pl.pipeline,   nullptr);
        vkDestroyPipeline(device, tier.contact_reduction_pl.pipeline, nullptr);
        vkDestroyPipeline(device, tier.draw_pl.pipeline,              nullptr);
        vkDestroyPipeline(device, tier.integrate_pl.pipeline,         nullptr);

        vkDestroyPipelineLayout(device, tier.pixel_physics_pl.layout,      nullptr);
        vkDestroyPipelineLayout(device, tier.pixel_reduction_pl.layout,    nullptr);
        vkDestroyPipelineLayout(device, tier.contact_reduction_pl.layout,  nullptr);
        vkDestroyPipelineLayout(device, tier.draw_pl.layout,               nullptr);
        vkDestroyPipelineLayout(device, tier.integrate_pl.layout,          nullptr);

        vkDestroyDescriptorSetLayout(device, tier.pixel_physics_pl.desc_layout,      nullptr);
        vkDestroyDescriptorSetLayout(device, tier.pixel_reduction_pl.desc_layout,    nullptr);
        vkDestroyDescriptorSetLayout(device, tier.contact_reduction_pl.desc_layout,  nullptr);
        vkDestroyDescriptorSetLayout(device, tier.draw_pl.desc_layout,               nullptr);
        vkDestroyDescriptorSetLayout(device, tier.integrate_pl.desc_layout,          nullptr);

        tier.desc_allocator.destroy_pool(device);

        engine->destroy_buffer(tier.rb[0]);
        engine->destroy_buffer(tier.rb[1]);
        engine->destroy_buffer(tier.physics_pixels);
        engine->destroy_buffer(tier.pixel_colors);
        engine->destroy_buffer(tier.body_l0);
        engine->destroy_buffer(tier.active_indices_buf);
        engine->destroy_buffer(tier.contact_entries);
        engine->destroy_buffer(tier.contact_counters);
        engine->destroy_buffer(tier.contact_agg);
    });
}

void PhysicsPipeline::init(ResiduaEngine* engine, uint32_t initial_capacity) {
    VkDevice device = engine->_device;

    VkExtent3D screen_extent { PHYSICS_WIDTH, PHYSICS_HEIGHT, 1 };
    output_screen = engine->create_image(screen_extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, output_screen.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    const size_t collision_size = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);
    static_collision = engine->create_buffer(collision_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    collision_pixels = engine->create_buffer(collision_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // ── gap_fill ──────────────────────────────────────────────────────────
    {
        DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2.f },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2.f },
        };
        gap_fill_allocator.init_pool(device, 2, pool_ratios);

        auto& pl = gap_fill_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_image
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_pixels
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // width(u32) + height(u32) = 8 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2 };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/gap_fill.comp.spv", pl.layout, nullptr);

        gap_fill_desc = gap_fill_allocator.allocate(device, pl.desc_layout);
        DescriptorWriter w;
        w.write_image (0, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_buffer(1, collision_pixels.buffer, collision_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, gap_fill_desc);
    }

    // ── Initialize all three tiers ────────────────────────────────────────
    init_tier(engine, 0, initial_capacity); // 4×4
    init_tier(engine, 1, initial_capacity); // 16×16
    init_tier(engine, 2, initial_capacity); // 64×64

    engine->_mainDeletionQueue.push_function([this, engine, device]() {
        vkDestroyPipeline(device, gap_fill_pl.pipeline, nullptr);
        vkDestroyPipelineLayout(device, gap_fill_pl.layout, nullptr);
        vkDestroyDescriptorSetLayout(device, gap_fill_pl.desc_layout, nullptr);
        gap_fill_allocator.destroy_pool(device);

        engine->destroy_buffer(static_collision);
        engine->destroy_buffer(collision_pixels);
        engine->destroy_image(output_screen);
    });
}

void PhysicsPipeline::upload_collision_layer(ResiduaEngine* engine, const char* path) {
    int w, h, channels;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_grey);
    assert(data && "upload_collision_layer: failed to open file");
    assert(w == (int)PHYSICS_WIDTH && h == (int)PHYSICS_HEIGHT
        && "upload_collision_layer: image must be PHYSICS_WIDTH × PHYSICS_HEIGHT");

    const uint32_t pixel_count  = PHYSICS_WIDTH * PHYSICS_HEIGHT;
    const size_t   staging_size = pixel_count * sizeof(CollisionPixel);

    AllocatedBuffer staging = engine->create_buffer(staging_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto* out = reinterpret_cast<CollisionPixel*>(staging.info.pMappedData);
    for (uint32_t i = 0; i < pixel_count; i++)
        out[i] = { data[i] > 0 ? 0xFFFFFFFFu : 0u, 0u };
    stbi_image_free(data);

    engine->immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy region { 0, 0, staging_size };
        vkCmdCopyBuffer(cmd, staging.buffer, static_collision.buffer, 1, &region);
    });

    engine->destroy_buffer(staging);
}

uint32_t PhysicsPipeline::add_body(
    ResiduaEngine*         engine,
    const LoadedBodyImage& img,
    uint32_t               tier_idx,
    glm::vec2              position,
    float                  rotation
) {
    assert(tier_idx < 3 && "add_body: tier_idx out of range");
    BodyTier& t = tiers[tier_idx];
    assert(img.width == t.body_w && img.height == t.body_h
        && "add_body: image dimensions don't match tier");

    uint32_t slot;
    if (!t.free_list.empty()) {
        slot = t.free_list.back();
        t.free_list.pop_back();
    } else {
        if (t.next_slot >= t.capacity)
            grow_tier_buffers(engine, tier_idx);
        slot = t.next_slot++;
    }

    static std::mt19937 rng{ std::random_device{}() };
    static std::uniform_real_distribution<float> hue_dist(0.f, 1.f);

    float h = hue_dist(rng);
    float s = 0.85f, v = 0.95f;
    float c = v * s, x = c * (1.f - std::abs(std::fmod(h * 6.f, 2.f) - 1.f));
    float m = v - c;
    glm::vec3 rgb;
    int hi = int(h * 6.f);
    switch (hi % 6) {
        case 0: rgb = { c, x, 0 }; break;
        case 1: rgb = { x, c, 0 }; break;
        case 2: rgb = { 0, c, x }; break;
        case 3: rgb = { 0, x, c }; break;
        case 4: rgb = { x, 0, c }; break;
        default: rgb = { c, 0, x }; break;
    }
    rgb += m;

    auto* colors = reinterpret_cast<glm::vec4*>(t.pixel_colors.info.pMappedData);
    // memcpy(colors + slot * BODY_PIXELS, img.pixels, BODY_PIXELS * sizeof(glm::vec4));
    glm::vec4* dst = colors + slot * t.body_pixels;
    for (uint32_t i = 0; i < t.body_pixels; i++)
        dst[i] = glm::vec4(rgb * glm::vec3(img.pixels[i]), img.pixels[i].a);

    auto* phys = reinterpret_cast<PhysicsPixel*>(t.physics_pixels.info.pMappedData);
    for (uint32_t i = 0; i < t.body_pixels; i++)
        phys[slot * t.body_pixels + i].total_mass = img.pixels[i].a;

    float total_mass = 0.f, I_com = 0.f;
    for (uint32_t i = 0; i < t.body_pixels; i++) {
        float m  = img.pixels[i].a;
        float lx = float(i % t.body_w) - t.body_w * 0.5f + 0.5f;
        float ly = float(i / t.body_w) - t.body_h * 0.5f + 0.5f;
        total_mass += m;
        I_com      += m * (lx * lx + ly * ly);
    }

    RigidBody body {
        .position         = position,
        .rotation         = rotation,
        .total_mass       = total_mass,
        .velocity         = {0.f, 0.f},
        .angular_velocity = 0.f,
        .I_com            = I_com,
        .pixel_index      = slot * t.body_pixels,
    };

    for (int i = 0; i < 2; i++) {
        auto* bodies = reinterpret_cast<RigidBody*>(t.rb[i].info.pMappedData);
        bodies[slot] = body;
    }

    t.active_slots.push_back(slot);
    t.active_count++;

    auto* indices = reinterpret_cast<uint32_t*>(t.active_indices_buf.info.pMappedData);
    indices[t.active_count - 1] = slot;

    return slot;
}

void PhysicsPipeline::remove_body(uint32_t tier_idx, uint32_t slot) {
    assert(tier_idx < 3 && "remove_body: tier_idx out of range");
    BodyTier& t = tiers[tier_idx];

    auto it = std::find(t.active_slots.begin(), t.active_slots.end(), slot);
    assert(it != t.active_slots.end() && "remove_body: slot not active");

    uint32_t pos = (uint32_t)(it - t.active_slots.begin());
    t.active_slots[pos] = t.active_slots.back();
    t.active_slots.pop_back();
    t.active_count--;

    auto* indices = reinterpret_cast<uint32_t*>(t.active_indices_buf.info.pMappedData);
    if (pos < t.active_count)
        indices[pos] = t.active_slots[pos];

    t.free_list.push_back(slot);
}

std::optional<std::pair<uint32_t, uint32_t>> PhysicsPipeline::body_at(glm::vec2 world_pos) const {
    for (uint32_t ti = 0; ti < 3; ti++) {
        const BodyTier& t = tiers[ti];
        if (t.active_count == 0) continue;

        auto* bodies = reinterpret_cast<const RigidBody*>(t.rb[frame_parity].info.pMappedData);
        auto* phys   = reinterpret_cast<const PhysicsPixel*>(t.physics_pixels.info.pMappedData);

        for (uint32_t slot : t.active_slots) {
            const RigidBody& b = bodies[slot];
            float c = cosf(-b.rotation), s = sinf(-b.rotation);
            glm::vec2 d     = world_pos - b.position;
            glm::vec2 local = { c*d.x - s*d.y, s*d.x + c*d.y };
            local += glm::vec2(t.body_w, t.body_h) * 0.5f - 0.5f;
            int lx = (int)floorf(local.x), ly = (int)floorf(local.y);
            if (lx < 0 || ly < 0 || lx >= (int)t.body_w || ly >= (int)t.body_h) continue;
            if (phys[slot * t.body_pixels + ly * t.body_w + lx].total_mass > 0.f)
                return std::make_pair(ti, slot);
        }
    }
    return std::nullopt;
}

void PhysicsPipeline::dispatch(VkCommandBuffer cmd, float dt) {
    bool any_active = false;
    for (uint32_t ti = 0; ti < 3; ti++)
        if (tiers[ti].active_count > 0) { any_active = true; break; }
    if (!any_active) return;

    const uint32_t p             = frame_parity;
    const size_t   collision_size = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);

    // 0. Clear output_screen to black
    VkClearColorValue clear_color { .float32 = { 0.f, 0.f, 0.f, 0.f } };
    VkImageSubresourceRange clear_range {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    vkCmdClearColorImage(cmd, output_screen.image, VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &clear_range);

    {
        VkImageMemoryBarrier2 img_barrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = VK_PIPELINE_STAGE_2_CLEAR_BIT,
            .srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .image            = output_screen.image,
            .subresourceRange = clear_range,
        };
        VkDependencyInfo dep {
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &img_barrier,
        };
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 1. Copy static_collision → collision_pixels
    {
        VkBufferCopy region { 0, 0, collision_size };
        vkCmdCopyBuffer(cmd, static_collision.buffer, collision_pixels.buffer, 1, &region);
    }
    {
        VkBufferMemoryBarrier2 barrier {
            .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .buffer        = collision_pixels.buffer,
            .offset        = 0,
            .size          = VK_WHOLE_SIZE,
        };
        VkDependencyInfo bdep {
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &barrier,
        };
        vkCmdPipelineBarrier2(cmd, &bdep);
    }

    // 2. Draw all tiers into output_screen + collision_pixels
    for (uint32_t ti = 0; ti < 3; ti++) {
        BodyTier& t = tiers[ti];
        if (t.active_count == 0) continue;

        struct { uint32_t body_count; uint32_t physics_width; uint32_t tier_index; }
            pc { t.active_count, PHYSICS_WIDTH, ti };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t.draw_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            t.draw_pl.layout, 0, 1, &t.draw_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, t.draw_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, t.active_count, 1, 1);
    }

    {
        VkImageMemoryBarrier2 img_bar {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .image            = output_screen.image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        VkBufferMemoryBarrier2 buf_bar {
            .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .buffer        = collision_pixels.buffer,
            .offset        = 0,
            .size          = VK_WHOLE_SIZE,
        };
        VkDependencyInfo dep2 {
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &buf_bar,
            .imageMemoryBarrierCount  = 1,
            .pImageMemoryBarriers     = &img_bar,
        };
        vkCmdPipelineBarrier2(cmd, &dep2);
    }

    // 2b. gap_fill
    {
        struct { uint32_t width; uint32_t height; } pc { PHYSICS_WIDTH, PHYSICS_HEIGHT };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gap_fill_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            gap_fill_pl.layout, 0, 1, &gap_fill_desc, 0, nullptr);
        vkCmdPushConstants(cmd, gap_fill_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (PHYSICS_WIDTH + 7) / 8, (PHYSICS_HEIGHT + 7) / 8, 1);
    }

    buffer_barrier(cmd, collision_pixels.buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 3–6. Per-tier: clear counters → pixel_physics → pixel_reduction + contact_reduction → integrate
    for (uint32_t ti = 0; ti < 3; ti++) {
        BodyTier& t = tiers[ti];
        if (t.active_count == 0) continue;

        // Clear contact_counters to 0 before pixel_physics writes to them
        vkCmdFillBuffer(cmd, t.contact_counters.buffer, 0, VK_WHOLE_SIZE, 0u);
        {
            VkBufferMemoryBarrier2 fill_bar {
                .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .buffer        = t.contact_counters.buffer,
                .offset        = 0, .size = VK_WHOLE_SIZE,
            };
            VkDependencyInfo dep { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &fill_bar };
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // pixel_physics — writes physics_pixels + contact_entries + contact_counters
        {
            struct { uint32_t body_count; float gravity; uint32_t phys_w; uint32_t phys_h; uint32_t tier_index; }
                pc { t.active_count, 100.0f, PHYSICS_WIDTH, PHYSICS_HEIGHT, ti };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t.pixel_physics_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                t.pixel_physics_pl.layout, 0, 1, &t.physics_desc[p], 0, nullptr);
            vkCmdPushConstants(cmd, t.pixel_physics_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, t.active_count, 1, 1);
        }

        // Barrier covering all pixel_physics outputs
        buffer_barrier(cmd, t.physics_pixels.buffer,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        buffer_barrier(cmd, t.contact_entries.buffer,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        buffer_barrier(cmd, t.contact_counters.buffer,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // pixel_reduction — physics_pixels → body_l0
        {
            struct { uint32_t body_count; } pc { t.active_count };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t.pixel_reduction_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                t.pixel_reduction_pl.layout, 0, 1, &t.reduction_desc, 0, nullptr);
            vkCmdPushConstants(cmd, t.pixel_reduction_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, t.active_count, 1, 1);
        }

        // contact_reduction — contact_entries → contact_agg (one impulse per unique other-body)
        {
            struct { uint32_t body_count; } pc { t.active_count };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t.contact_reduction_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                t.contact_reduction_pl.layout, 0, 1, &t.contact_desc, 0, nullptr);
            vkCmdPushConstants(cmd, t.contact_reduction_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, t.active_count, 1, 1);
        }

        // Wait for both reductions before integrate
        buffer_barrier(cmd, t.body_l0.buffer,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        buffer_barrier(cmd, t.contact_agg.buffer,
            VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // integrate — applies gravity + per-contact impulses
        {
            struct { uint32_t body_count; float dt; float restitution; }
                pc { t.active_count, dt, 0.5f };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, t.integrate_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                t.integrate_pl.layout, 0, 1, &t.integrate_desc[p], 0, nullptr);
            vkCmdPushConstants(cmd, t.integrate_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (t.active_count + 63) / 64, 1, 1);
        }
    }

    frame_parity ^= 1;
}
