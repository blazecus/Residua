#include "cpu_draw_system.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_images.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <fmt/core.h>
#include <cassert>
#include <cstring>

// ─── Local helpers ────────────────────────────────────────────────────────────

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                        VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("cpu_draw_system: failed to load shader {}\n", spv);
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
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(device, mod, nullptr);
    return pipeline;
}

// ─── Init ─────────────────────────────────────────────────────────────────────

void CPUDrawSystem::grow_buffers(ResiduaEngine* engine,
                                  uint32_t needed_bodies, uint32_t needed_pixels)
{
    uint32_t new_cap_bodies = std::max(cap_bodies, 16u);
    while (new_cap_bodies < needed_bodies) new_cap_bodies *= 2;

    uint32_t new_cap_pixels = std::max(cap_pixels, 1u << 16);
    while (new_cap_pixels < needed_pixels) new_cap_pixels *= 2;

    const bool bodies_grew = new_cap_bodies > cap_bodies;
    const bool pixels_grew = new_cap_pixels > cap_pixels;
    if (!bodies_grew && !pixels_grew) return;

    vkDeviceWaitIdle(engine->_device);

    AllocatedBuffer new_rb_buf     = rb_draw_buf;
    AllocatedBuffer new_active_buf = active_indices_buf;
    AllocatedBuffer new_pixel_buf  = pixel_colors_buf;

    if (bodies_grew) {
        new_rb_buf = engine->create_buffer(
            new_cap_bodies * sizeof(RigidBodyDrawGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        new_active_buf = engine->create_buffer(
            new_cap_bodies * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    }

    if (pixels_grew) {
        new_pixel_buf = engine->create_buffer(
            new_cap_pixels * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        if (next_pixel > 0)
            std::memcpy(new_pixel_buf.info.pMappedData,
                        pixel_colors_buf.info.pMappedData,
                        next_pixel * sizeof(glm::vec4));
    }

    // Rebind the draw descriptor set to the new buffers.
    DescriptorWriter w;
    w.write_buffer(0, new_rb_buf.buffer,       new_cap_bodies * sizeof(RigidBodyDrawGPU),          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_buffer(1, new_pixel_buf.buffer,    new_cap_pixels * sizeof(glm::vec4),                 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_image (2, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.write_buffer(3, collision_buf.buffer,    PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_buffer(4, new_active_buf.buffer,   new_cap_bodies * sizeof(uint32_t),                  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.update_set(engine->_device, draw_desc);

    if (bodies_grew && cap_bodies > 0) {
        vmaDestroyBuffer(engine->_allocator, rb_draw_buf.buffer,        rb_draw_buf.allocation);
        vmaDestroyBuffer(engine->_allocator, active_indices_buf.buffer, active_indices_buf.allocation);
    }
    if (pixels_grew && cap_pixels > 0)
        vmaDestroyBuffer(engine->_allocator, pixel_colors_buf.buffer, pixel_colors_buf.allocation);

    rb_draw_buf        = new_rb_buf;
    active_indices_buf = new_active_buf;
    pixel_colors_buf   = new_pixel_buf;
    cap_bodies         = new_cap_bodies;
    cap_pixels         = new_cap_pixels;
}

void CPUDrawSystem::init(ResiduaEngine* engine)
{
    VkDevice device = engine->_device;

    // ── Fixed-size GPU buffers ───────────────────────────────────────────────

    collision_buf = engine->create_buffer(
        PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // ── Output image ─────────────────────────────────────────────────────────

    output_screen = engine->create_image(
        { PHYSICS_WIDTH, PHYSICS_HEIGHT, 1 },
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // Transition to GENERAL layout once so it stays there.
    engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, output_screen.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    // ── Descriptor pool ───────────────────────────────────────────────────────

    DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2.f },
    };
    desc_allocator.init_pool(device, 4, pool_ratios);

    // ── Draw pipeline ─────────────────────────────────────────────────────────
    {
        auto& pl = draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb_draw_buf
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors_buf
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_buf
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices_buf
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
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/rigidbody_draw_cpu.comp.spv", pl.layout);

        draw_desc = desc_allocator.allocate(device, pl.desc_layout);
    }

    // Initial buffer allocation — grow_buffers also writes the draw descriptor set.
    grow_buffers(engine, 16, 1 << 16);

    // ── Gap-fill pipeline ─────────────────────────────────────────────────────
    {
        auto& pl = gap_fill_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // collision_buf
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
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/gap_fill.comp.spv", pl.layout);

        gap_fill_desc = desc_allocator.allocate(device, pl.desc_layout);

        const size_t col_size = PHYSICS_WIDTH * PHYSICS_HEIGHT * sizeof(CollisionPixel);
        DescriptorWriter w;
        w.write_image (0, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.write_buffer(1, collision_buf.buffer,    col_size,       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, gap_fill_desc);
    }

    // Register cleanup with the engine's deletion queue.
    engine->_mainDeletionQueue.push_function([this, device]() {
        vkDestroyPipeline(device, draw_pl.pipeline,     nullptr);
        vkDestroyPipeline(device, gap_fill_pl.pipeline, nullptr);

        vkDestroyPipelineLayout(device, draw_pl.layout,     nullptr);
        vkDestroyPipelineLayout(device, gap_fill_pl.layout, nullptr);

        vkDestroyDescriptorSetLayout(device, draw_pl.desc_layout,     nullptr);
        vkDestroyDescriptorSetLayout(device, gap_fill_pl.desc_layout, nullptr);

        desc_allocator.destroy_pool(device);
    });
}

// ─── Add / remove bodies ──────────────────────────────────────────────────────

uint32_t CPUDrawSystem::add_body(ResiduaEngine* engine, RigidBody2 body)
{
    const uint32_t w = body.sprite.width;
    const uint32_t h = body.sprite.height;
    const uint32_t n = w * h;

    const uint32_t pixel_offset = next_pixel;
    next_pixel += n;

    // Grow GPU buffers before writing if capacity would be exceeded.
    uint32_t slot = world.add_body(body);
    grow_buffers(engine,
        (uint32_t)world.bodies.size(),
        next_pixel);

    // Grow body_draw_info to cover the new slot.
    if (slot >= (uint32_t)body_draw_info.size())
        body_draw_info.resize(slot + 1);

    body_draw_info[slot] = { pixel_offset, w, h };

    // Write pixel colors directly into the CPU-visible buffer.
    const size_t byte_size = n * sizeof(glm::vec4);
    auto* dst = static_cast<glm::vec4*>(pixel_colors_buf.info.pMappedData);
    std::memcpy(dst + pixel_offset, body.sprite.pixels.data(), byte_size);

    return slot;
}

void CPUDrawSystem::remove_body(uint32_t idx)
{
    world.remove_body(idx);
    // Pixel data in pixel_colors_buf is not reclaimed; the slot is simply inactive.
}

// ─── Body query ───────────────────────────────────────────────────────────────

std::optional<uint32_t> CPUDrawSystem::body_at(glm::vec2 world_pos) const
{
    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (!world.active[i]) continue;
        const RigidBody2&   rb  = world.bodies[i];
        const BodyDrawInfo& bdi = body_draw_info[i];

        // Rotate query into body-local space.
        glm::vec2 delta = world_pos - glm::vec2(rb.position);
        float c = std::cos(-rb.position.z), s = std::sin(-rb.position.z);
        glm::vec2 local = { c * delta.x - s * delta.y,
                            s * delta.x + c * delta.y };

        // Shift to pixel coords (sprite is centered at body.position).
        local += glm::vec2((float)bdi.body_w, (float)bdi.body_h) * 0.5f - 0.5f;

        int px = (int)std::floor(local.x), py = (int)std::floor(local.y);
        if (px < 0 || py < 0 ||
            (uint32_t)px >= bdi.body_w || (uint32_t)py >= bdi.body_h) continue;

        if (rb.sprite.pixels[(uint32_t)py * bdi.body_w + (uint32_t)px].a > 0.5f)
            return i;
    }
    return std::nullopt;
}

// ─── Per-frame dispatch ───────────────────────────────────────────────────────

void CPUDrawSystem::dispatch(VkCommandBuffer cmd, float dt)
{
    // 1. Step the CPU physics simulation.
    world.step(dt);

    // 2. Upload draw data for all active bodies.
    auto* rb_ptr     = static_cast<RigidBodyDrawGPU*>(rb_draw_buf.info.pMappedData);
    auto* active_ptr = static_cast<uint32_t*>(active_indices_buf.info.pMappedData);
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (!world.active[i]) continue;
        // Static bodies added directly to world have no draw info — skip.
        if (i >= (uint32_t)body_draw_info.size()) continue;
        const BodyDrawInfo& bdi = body_draw_info[i];
        if (bdi.body_w == 0 || bdi.body_h == 0) continue;

        const RigidBody2& rb = world.bodies[i];
        rb_ptr[i] = RigidBodyDrawGPU{
            .position         = glm::vec2(rb.position),
            .rotation         = rb.position.z,
            .total_mass       = rb.mass,
            .velocity         = glm::vec2(rb.velocity),
            .angular_velocity = rb.velocity.z,
            .I_com            = rb.inertia,
            .pixel_index      = bdi.pixel_index,
            .body_w           = bdi.body_w,
            .body_h           = bdi.body_h,
        };
        active_ptr[active_count++] = i;
    }

    if (active_count == 0) return;

    // 3. Clear output_screen and collision_buf.
    VkClearColorValue clear_color{};
    VkImageSubresourceRange full_range {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1,
        .layerCount = 1,
    };
    vkCmdClearColorImage(cmd, output_screen.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clear_color, 1, &full_range);
    vkCmdFillBuffer(cmd, collision_buf.buffer, 0, VK_WHOLE_SIZE, 0);

    // 4. Barrier: clear → draw compute.
    VkBufferMemoryBarrier2 col_clear_barrier {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .buffer        = collision_buf.buffer,
        .offset        = 0,
        .size          = VK_WHOLE_SIZE,
    };
    VkImageMemoryBarrier2 img_clear_barrier {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .image            = output_screen.image,
        .subresourceRange = full_range,
    };
    VkDependencyInfo dep1 {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &col_clear_barrier,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &img_clear_barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep1);

    // 5. Draw pass — one workgroup per active body.
    struct { uint32_t body_count; uint32_t physics_width; } draw_pc {
        active_count, PHYSICS_WIDTH
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, draw_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        draw_pl.layout, 0, 1, &draw_desc, 0, nullptr);
    vkCmdPushConstants(cmd, draw_pl.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(draw_pc), &draw_pc);
    vkCmdDispatch(cmd, active_count, 1, 1);

    // 6. Barrier: draw → gap_fill.
    VkBufferMemoryBarrier2 col_draw_barrier {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .buffer        = collision_buf.buffer,
        .offset        = 0,
        .size          = VK_WHOLE_SIZE,
    };
    VkImageMemoryBarrier2 img_draw_barrier {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask    = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .image            = output_screen.image,
        .subresourceRange = full_range,
    };
    VkDependencyInfo dep2 {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &col_draw_barrier,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &img_draw_barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep2);

    // 7. Gap-fill pass.
    struct { uint32_t width; uint32_t height; } gap_pc { PHYSICS_WIDTH, PHYSICS_HEIGHT };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gap_fill_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        gap_fill_pl.layout, 0, 1, &gap_fill_desc, 0, nullptr);
    vkCmdPushConstants(cmd, gap_fill_pl.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gap_pc), &gap_pc);
    vkCmdDispatch(cmd, (PHYSICS_WIDTH + 7) / 8, (PHYSICS_HEIGHT + 7) / 8, 1);
}

PhysicsStats CPUDrawSystem::get_stats(){
    return world.stats;
}
