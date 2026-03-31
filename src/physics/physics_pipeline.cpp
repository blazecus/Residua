#include "physics_pipeline.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_images.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <stb_image.h>
#include <fmt/core.h>
#include <cassert>
#include <cstring>

// ─── Image loading ────────────────────────────────────────────────────────────

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

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

// -- physics --

void PhysicsPipeline::init(ResiduaEngine* engine, uint32_t max_bodies_) {
    max_bodies = max_bodies_;
    VkDevice device = engine->_device;

    VkExtent3D screen_extent { PHYSICS_WIDTH, PHYSICS_HEIGHT, 1 };

    collision_layer = engine->create_image(screen_extent, VK_FORMAT_R8_UINT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    output_screen = engine->create_image(screen_extent, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, collision_layer.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkutil::transition_image(cmd, output_screen.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    const size_t rb_size           = max_bodies * sizeof(RigidBody);
    const size_t phys_pixels_size  = max_bodies * BODY_PIXELS * sizeof(PhysicsPixel);
    const size_t pixel_colors_size = max_bodies * BODY_PIXELS * sizeof(glm::vec4);
    const size_t body_l0_size      = max_bodies * sizeof(PhysicsPixel);

    rb[0]          = engine->create_buffer(rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    rb[1]          = engine->create_buffer(rb_size,           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    physics_pixels = engine->create_buffer(phys_pixels_size,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    pixel_colors   = engine->create_buffer(pixel_colors_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    body_l0        = engine->create_buffer(body_l0_size,      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   4.f },
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
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // collision_layer
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) + sizeof(float) };
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
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,           rb_size,           0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, physics_pixels.buffer,  phys_pixels_size,  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_image (2, collision_layer.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.write_buffer(3, pixel_colors.buffer,    pixel_colors_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, physics_desc[i]);
        }
    }

    // ── pixel_reduction ───────────────────────────────────────────────────
    {
        auto& pl = pixel_reduction_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // src (physics_pixels)
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // dst (body_l0)
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2 };
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
        DescriptorWriter w;
        w.write_buffer(0, physics_pixels.buffer, phys_pixels_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, body_l0.buffer,        body_l0_size,     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, reduction_desc);
    }

    // ── rigidbody_draw ────────────────────────────────────────────────────
    {
        auto& pl = draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // RigidBodyBuffer
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
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
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,        rb_size,           0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, pixel_colors.buffer, pixel_colors_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_image (2, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            w.update_set(device, draw_desc[i]);
        }
    }

    // ── rigidbody_integrate ───────────────────────────────────────────────
    {
        auto& pl = integrate_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb src
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb dst
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // body_l0
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) + sizeof(float) };
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
            DescriptorWriter w;
            w.write_buffer(0, rb[i].buffer,     rb_size,      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, rb[1-i].buffer,   rb_size,      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(2, body_l0.buffer,   body_l0_size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.update_set(device, integrate_desc[i]);
        }
    }

    engine->_mainDeletionQueue.push_function([this, engine, device]() {
        vkDestroyPipeline(device, pixel_physics_pl.pipeline,  nullptr);
        vkDestroyPipeline(device, pixel_reduction_pl.pipeline, nullptr);
        vkDestroyPipeline(device, draw_pl.pipeline,           nullptr);
        vkDestroyPipeline(device, integrate_pl.pipeline,      nullptr);

        vkDestroyPipelineLayout(device, pixel_physics_pl.layout,  nullptr);
        vkDestroyPipelineLayout(device, pixel_reduction_pl.layout, nullptr);
        vkDestroyPipelineLayout(device, draw_pl.layout,           nullptr);
        vkDestroyPipelineLayout(device, integrate_pl.layout,      nullptr);

        vkDestroyDescriptorSetLayout(device, pixel_physics_pl.desc_layout,  nullptr);
        vkDestroyDescriptorSetLayout(device, pixel_reduction_pl.desc_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, draw_pl.desc_layout,           nullptr);
        vkDestroyDescriptorSetLayout(device, integrate_pl.desc_layout,      nullptr);

        desc_allocator.destroy_pool(device);

        engine->destroy_buffer(rb[0]);
        engine->destroy_buffer(rb[1]);
        engine->destroy_buffer(physics_pixels);
        engine->destroy_buffer(pixel_colors);
        engine->destroy_buffer(body_l0);

        engine->destroy_image(collision_layer);
        engine->destroy_image(output_screen);
    });
}

uint32_t PhysicsPipeline::add_body(
    ResiduaEngine*         engine,
    const LoadedBodyImage& img,
    glm::vec2              position,
    float                  rotation
) {
    assert(body_count < max_bodies && "PhysicsPipeline: max_bodies exceeded");
    uint32_t idx = body_count++;

    auto* colors = reinterpret_cast<glm::vec4*>(pixel_colors.info.pMappedData);
    memcpy(colors + idx * BODY_PIXELS, img.pixels, BODY_PIXELS * sizeof(glm::vec4));

    auto* phys = reinterpret_cast<PhysicsPixel*>(physics_pixels.info.pMappedData);
    for (uint32_t i = 0; i < BODY_PIXELS; i++) {
        phys[idx * BODY_PIXELS + i].total_mass = img.pixels[i].a;
    }
    //TODO: memcpy?

    RigidBody body {
        .position         = position,
        .rotation         = rotation,
        .velocity         = {0.f, 0.f},
        .angular_velocity = 0.f,
        .pixel_index      = idx * BODY_PIXELS,
    };

    for (int i = 0; i < 2; i++) {
        auto* bodies = reinterpret_cast<RigidBody*>(rb[i].info.pMappedData);
        bodies[idx] = body;
    }

    return idx;
}

// ─── PhysicsPipeline::dispatch ────────────────────────────────────────────────

void PhysicsPipeline::dispatch(VkCommandBuffer cmd, float dt) {
    if (body_count == 0) return;

    const uint32_t p = frame_parity;

    // 1. pixel_physics: one workgroup per body 
    {
        struct { uint32_t body_count; float gravity; } pc { body_count, 9.81f };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pixel_physics_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pixel_physics_pl.layout, 0, 1, &physics_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, pixel_physics_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, body_count, 1, 1);
    }

    buffer_barrier(cmd, physics_pixels.buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 2. pixel_reduction: reduces physics_pixels → body_l0, one workgroup per body
    {
        struct { uint32_t src_width; uint32_t src_height; } pc { body_count * BODY_W, BODY_H };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pixel_reduction_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pixel_reduction_pl.layout, 0, 1, &reduction_desc, 0, nullptr);
        vkCmdPushConstants(cmd, pixel_reduction_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, body_count, 1, 1);
    }

    buffer_barrier(cmd, body_l0.buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 3. rigidbody_integrate: reads rb[p] + body_l0, writes rb[1-p]
    {
        struct { uint32_t body_count; float dt; } pc { body_count, dt };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            integrate_pl.layout, 0, 1, &integrate_desc[p], 0, nullptr);
        vkCmdPushConstants(cmd, integrate_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (body_count + 63) / 64, 1, 1);
    }

    buffer_barrier(cmd, rb[1 - p].buffer,
        VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // 4. rigidbody_draw: reads rb[1-p] (newly integrated state)
    {
        uint32_t draw_rb = 1 - p;
        struct { uint32_t body_count; } pc { body_count };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, draw_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            draw_pl.layout, 0, 1, &draw_desc[draw_rb], 0, nullptr);
        vkCmdPushConstants(cmd, draw_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, body_count, 1, 1);
    }

    frame_parity ^= 1;
}
