#include "physics_pipeline.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_images.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <stb_image.h>
#include <fmt/core.h>
#include <algorithm>
#include <cassert>
#include <cstring>

// ─── Helpers ─────────────────────────────────────────────────────────────────

LoadedBodyImage load_body_image(const char* path) {
    int w, h, channels;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    assert(data && "load_body_image: failed to open file");
    assert(w == (int)BODY_W && h == (int)BODY_H
        && "load_body_image: image must be exactly BODY_W × BODY_H pixels");

    LoadedBodyImage result;
    for (uint32_t i = 0; i < BODY_PIXELS; i++) {
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
    VkDevice                   device,
    const char*                spv_path,
    VkPipelineLayout           layout,
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

void PhysicsPipeline::update_all_descriptors(VkDevice device) {
    const size_t rb_size           = capacity * sizeof(RigidBody);
    const size_t phys_pixels_size  = capacity * BODY_PIXELS * sizeof(PhysicsPixel);
    const size_t pixel_colors_size = capacity * BODY_PIXELS * sizeof(glm::vec4);
    const size_t body_l0_size      = capacity * sizeof(PhysicsPixel);
    const size_t collision_size    = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);
    const size_t active_idx_size   = capacity * sizeof(uint32_t);

    for (int i = 0; i < 2; i++) {
        {
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,              rb_size,          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, physics_pixels.buffer,     phys_pixels_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, collision_pixels.buffer,   collision_size,   0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(3, active_indices_buf.buffer, active_idx_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, physics_desc[i]);
        }
        {
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,              rb_size,           0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, pixel_colors.buffer,       pixel_colors_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, physics_pixels.buffer,     phys_pixels_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_image (3, output_screen.imageView,   VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_buffer(4, collision_pixels.buffer,   collision_size,    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(5, active_indices_buf.buffer, active_idx_size,   0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, draw_desc[i]);
        }
        {
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,              rb_size,         0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, rb[1-i].buffer,            rb_size,         0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, body_l0.buffer,            body_l0_size,    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(3, active_indices_buf.buffer, active_idx_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, integrate_desc[i]);
        }
    }
    {
        DescriptorWriter w;
        w.write_buffer(0, physics_pixels.buffer,     phys_pixels_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, body_l0.buffer,            body_l0_size,     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(2, active_indices_buf.buffer, active_idx_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, reduction_desc);
    }
    // gap_fill only touches output_screen and collision_pixels — neither grows, no update needed
}

void PhysicsPipeline::grow_buffers(ResiduaEngine* engine) {
    const uint32_t new_capacity = capacity * 2;
    fmt::print("PhysicsPipeline: growing from {} to {} slots\n", capacity, new_capacity);

    vkDeviceWaitIdle(engine->_device);

    const size_t new_rb_size           = new_capacity * sizeof(RigidBody);
    const size_t new_phys_pixels_size  = new_capacity * BODY_PIXELS * sizeof(PhysicsPixel);
    const size_t new_pixel_colors_size = new_capacity * BODY_PIXELS * sizeof(glm::vec4);
    const size_t new_body_l0_size      = new_capacity * sizeof(PhysicsPixel);
    const size_t new_active_idx_size   = new_capacity * sizeof(uint32_t);

    AllocatedBuffer new_rb0          = engine->create_buffer(new_rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_rb1          = engine->create_buffer(new_rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_phys_pixels  = engine->create_buffer(new_phys_pixels_size,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_pixel_colors = engine->create_buffer(new_pixel_colors_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_body_l0      = engine->create_buffer(new_body_l0_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    AllocatedBuffer new_active_idx   = engine->create_buffer(new_active_idx_size,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    const size_t old_rb_size           = capacity * sizeof(RigidBody);
    const size_t old_phys_pixels_size  = capacity * BODY_PIXELS * sizeof(PhysicsPixel);
    const size_t old_pixel_colors_size = capacity * BODY_PIXELS * sizeof(glm::vec4);
    const size_t old_body_l0_size      = capacity * sizeof(PhysicsPixel);

    memcpy(new_rb0.info.pMappedData,         rb[0].info.pMappedData,         old_rb_size);
    memcpy(new_rb1.info.pMappedData,         rb[1].info.pMappedData,         old_rb_size);
    memcpy(new_phys_pixels.info.pMappedData, physics_pixels.info.pMappedData, old_phys_pixels_size);
    memcpy(new_pixel_colors.info.pMappedData, pixel_colors.info.pMappedData,  old_pixel_colors_size);
    memcpy(new_body_l0.info.pMappedData,     body_l0.info.pMappedData,       old_body_l0_size);
    memcpy(new_active_idx.info.pMappedData,  active_slots.data(),            active_count * sizeof(uint32_t));

    engine->destroy_buffer(rb[0]);
    engine->destroy_buffer(rb[1]);
    engine->destroy_buffer(physics_pixels);
    engine->destroy_buffer(pixel_colors);
    engine->destroy_buffer(body_l0);
    engine->destroy_buffer(active_indices_buf);

    rb[0]             = new_rb0;
    rb[1]             = new_rb1;
    physics_pixels    = new_phys_pixels;
    pixel_colors      = new_pixel_colors;
    body_l0           = new_body_l0;
    active_indices_buf = new_active_idx;
    capacity          = new_capacity;

    update_all_descriptors(engine->_device);
}

void PhysicsPipeline::init(ResiduaEngine* engine, uint32_t initial_capacity) {
    capacity = initial_capacity;
    VkDevice device = engine->_device;

    VkExtent3D screen_extent { PHYSICS_WIDTH, PHYSICS_HEIGHT, 1 };

    output_screen = engine->create_image(screen_extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, output_screen.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    const size_t rb_size           = capacity * sizeof(RigidBody);
    const size_t phys_pixels_size  = capacity * BODY_PIXELS * sizeof(PhysicsPixel);
    const size_t pixel_colors_size = capacity * BODY_PIXELS * sizeof(glm::vec4);
    const size_t body_l0_size      = capacity * sizeof(PhysicsPixel);
    const size_t collision_size    = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);
    const size_t active_idx_size   = capacity * sizeof(uint32_t);

    rb[0]              = engine->create_buffer(rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    rb[1]              = engine->create_buffer(rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    physics_pixels     = engine->create_buffer(phys_pixels_size,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    pixel_colors       = engine->create_buffer(pixel_colors_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    body_l0            = engine->create_buffer(body_l0_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    active_indices_buf = engine->create_buffer(active_idx_size,   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    static_collision   = engine->create_buffer(collision_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    collision_pixels   = engine->create_buffer(collision_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   2.f },
    };
    desc_allocator.init_pool(device, 12, pool_ratios);

    uint32_t spec_data[] = { BODY_W, BODY_H };
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

    // ── pixel_physics ─────────────────────────────────────────────────────
    {
        auto& pl = pixel_physics_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // RigidBodyBuffer
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // PhysicsPixelBuffer
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_pixels (read)
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) + gravity(f32) + physics_width(u32) + physics_height(u32) = 16 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 3 + sizeof(float) };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/pixel_physics.comp.spv", pl.layout, &spec_info);

        for (int i = 0; i < 2; i++) {
            physics_desc[i] = desc_allocator.allocate(device, pl.desc_layout);
        }
    }

    // ── pixel_reduction ───────────────────────────────────────────────────
    {
        auto& pl = pixel_reduction_pl;
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

        reduction_desc = desc_allocator.allocate(device, pl.desc_layout);
    }

    // ── rigidbody_draw ────────────────────────────────────────────────────
    {
        auto& pl = draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // RigidBodyBuffer
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // physics_pixels (read)
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_pixels (write)
        b.add_binding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // body_count(u32) + physics_width(u32) = 8 bytes
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2 };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/rigidbody_draw.comp.spv", pl.layout, &spec_info);

        for (int i = 0; i < 2; i++) {
            draw_desc[i] = desc_allocator.allocate(device, pl.desc_layout);
        }
    }

    // ── gap_fill ──────────────────────────────────────────────────────────
    {
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

        gap_fill_desc = desc_allocator.allocate(device, pl.desc_layout);
        DescriptorWriter w;
        w.write_image (0, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_buffer(1, collision_pixels.buffer, collision_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, gap_fill_desc);
    }

    // ── rigidbody_integrate ───────────────────────────────────────────────
    {
        auto& pl = integrate_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb src
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb dst
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // body_l0
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) + sizeof(float) * 2 };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = create_compute_pipeline(device, "../shaders/rigidbody_integrate.comp.spv", pl.layout, nullptr);

        for (int i = 0; i < 2; i++) {
            integrate_desc[i] = desc_allocator.allocate(device, pl.desc_layout);
        }
    }

    update_all_descriptors(device);

    engine->_mainDeletionQueue.push_function([this, engine, device]() {
        vkDestroyPipeline(device, pixel_physics_pl.pipeline,   nullptr);
        vkDestroyPipeline(device, pixel_reduction_pl.pipeline, nullptr);
        vkDestroyPipeline(device, draw_pl.pipeline,            nullptr);
        vkDestroyPipeline(device, gap_fill_pl.pipeline,        nullptr);
        vkDestroyPipeline(device, integrate_pl.pipeline,       nullptr);

        vkDestroyPipelineLayout(device, pixel_physics_pl.layout,   nullptr);
        vkDestroyPipelineLayout(device, pixel_reduction_pl.layout,  nullptr);
        vkDestroyPipelineLayout(device, draw_pl.layout,             nullptr);
        vkDestroyPipelineLayout(device, gap_fill_pl.layout,         nullptr);
        vkDestroyPipelineLayout(device, integrate_pl.layout,        nullptr);

        vkDestroyDescriptorSetLayout(device, pixel_physics_pl.desc_layout,   nullptr);
        vkDestroyDescriptorSetLayout(device, pixel_reduction_pl.desc_layout,  nullptr);
        vkDestroyDescriptorSetLayout(device, draw_pl.desc_layout,             nullptr);
        vkDestroyDescriptorSetLayout(device, gap_fill_pl.desc_layout,         nullptr);
        vkDestroyDescriptorSetLayout(device, integrate_pl.desc_layout,        nullptr);

        desc_allocator.destroy_pool(device);

        engine->destroy_buffer(rb[0]);
        engine->destroy_buffer(rb[1]);
        engine->destroy_buffer(physics_pixels);
        engine->destroy_buffer(pixel_colors);
        engine->destroy_buffer(body_l0);
        engine->destroy_buffer(active_indices_buf);
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
        && "upload_collision_layer: image must be PHYSICS_WIDTH x PHYSICS_HEIGHT");

    const uint32_t pixel_count  = PHYSICS_WIDTH * PHYSICS_HEIGHT;
    const size_t   staging_size = pixel_count * sizeof(CollisionPixel);

    AllocatedBuffer staging = engine->create_buffer(staging_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto* out = reinterpret_cast<CollisionPixel*>(staging.info.pMappedData);
    for (uint32_t i = 0; i < pixel_count; i++) {
        out[i] = { data[i] > 0 ? 0xFFFFFFFFu : 0u, 0.f, 0.f };
    }
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
    glm::vec2              position,
    float                  rotation
) {
    uint32_t slot;
    if (!free_list.empty()) {
        slot = free_list.back();
        free_list.pop_back();
    } else {
        if (next_slot >= capacity) {
            grow_buffers(engine);
        }
        slot = next_slot++;
    }

    auto* colors = reinterpret_cast<glm::vec4*>(pixel_colors.info.pMappedData);
    memcpy(colors + slot * BODY_PIXELS, img.pixels, BODY_PIXELS * sizeof(glm::vec4));

    auto* phys = reinterpret_cast<PhysicsPixel*>(physics_pixels.info.pMappedData);
    for (uint32_t i = 0; i < BODY_PIXELS; i++) {
        phys[slot * BODY_PIXELS + i].total_mass = img.pixels[i].a;
    }

    float total_mass = 0.f;
    float I_com      = 0.f;
    for (uint32_t i = 0; i < BODY_PIXELS; i++) {
        float m  = img.pixels[i].a;
        float lx = (float)(i % BODY_W) - BODY_W * 0.5f + 0.5f;
        float ly = (float)(i / BODY_W) - BODY_H * 0.5f + 0.5f;
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
        .pixel_index      = slot * BODY_PIXELS,
    };

    for (int i = 0; i < 2; i++) {
        auto* bodies = reinterpret_cast<RigidBody*>(rb[i].info.pMappedData);
        bodies[slot] = body;
    }

    active_slots.push_back(slot);
    active_count++;

    auto* indices = reinterpret_cast<uint32_t*>(active_indices_buf.info.pMappedData);
    indices[active_count - 1] = slot;

    return slot;
}

void PhysicsPipeline::remove_body(uint32_t slot) {
    auto it = std::find(active_slots.begin(), active_slots.end(), slot);
    assert(it != active_slots.end() && "remove_body: slot not active");

    uint32_t pos = (uint32_t)(it - active_slots.begin());

    active_slots[pos] = active_slots.back();
    active_slots.pop_back();
    active_count--;

    auto* indices = reinterpret_cast<uint32_t*>(active_indices_buf.info.pMappedData);
    if (pos < active_count) {
        indices[pos] = active_slots[pos]; 
    }

    free_list.push_back(slot);
}

std::optional<uint32_t> PhysicsPipeline::body_at(glm::vec2 world_pos) const {
    auto* bodies = reinterpret_cast<const RigidBody*>(rb[frame_parity].info.pMappedData);
    auto* phys   = reinterpret_cast<const PhysicsPixel*>(physics_pixels.info.pMappedData);

    for (uint32_t slot : active_slots) {
        const RigidBody& b = bodies[slot];
        float c = cosf(-b.rotation), s = sinf(-b.rotation);
        glm::vec2 d     = world_pos - b.position;
        glm::vec2 local = { c*d.x - s*d.y, s*d.x + c*d.y };
        local += glm::vec2(BODY_W, BODY_H) * 0.5f - 0.5f;
        int lx = (int)floorf(local.x), ly = (int)floorf(local.y);
        if (lx < 0 || ly < 0 || lx >= (int)BODY_W || ly >= (int)BODY_H) continue;
        if (phys[slot * BODY_PIXELS + ly * BODY_W + lx].total_mass > 0.f)
            return slot;
    }
    return std::nullopt;
}

void PhysicsPipeline::dispatch(VkCommandBuffer cmd, float dt) {
    if (active_count == 0) return;

    const uint32_t p              = frame_parity;
    const size_t   collision_size = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);

    // 0. Clear output_screen to black
    // TODO: renderer will take care of this in the future, but its still important for the collision layer to clear
    VkClearColorValue clear_color { .float32 = { 0.f, 0.f, 0.f, 0.f } };
    VkImageSubresourceRange clear_range {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,
    };
    vkCmdClearColorImage(cmd, output_screen.image, VK_IMAGE_LAYOUT_GENERAL,
        &clear_color, 1, &clear_range);

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

    // 1. Copy static_collision → collision_pixels 
    // TODO: static bodies will be handled as either one or multiple big images that can be modified
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

    // 2. rigidbody_draw
    {
        struct { uint32_t body_count; uint32_t physics_width; } pc { active_count, PHYSICS_WIDTH };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, draw_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            draw_pl.layout, 0, 1, &draw_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, draw_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, active_count, 1, 1);
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

    // 3. pixel_physics
    {
        struct { uint32_t body_count; float gravity; uint32_t phys_w; uint32_t phys_h; }
            pc { active_count, 100.0f, PHYSICS_WIDTH, PHYSICS_HEIGHT };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pixel_physics_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pixel_physics_pl.layout, 0, 1, &physics_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, pixel_physics_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, active_count, 1, 1);
    }

    buffer_barrier(cmd, physics_pixels.buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 4. pixel_reduction
    {
        struct { uint32_t body_count; } pc { active_count };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pixel_reduction_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pixel_reduction_pl.layout, 0, 1, &reduction_desc, 0, nullptr);
        vkCmdPushConstants(cmd, pixel_reduction_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, active_count, 1, 1);
    }

    buffer_barrier(cmd, body_l0.buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 5. rigidbody_integrate
    {
        struct { uint32_t body_count; float dt; float restitution; } pc { active_count, dt, 0.3f };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            integrate_pl.layout, 0, 1, &integrate_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, integrate_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (active_count + 63) / 64, 1, 1);
    }

    frame_parity ^= 1;
}
