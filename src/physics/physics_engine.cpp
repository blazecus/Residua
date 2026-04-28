#include "physics_engine.h"
#include "shape_gen.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_images.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include "manifold.h"
#include <fmt/core.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <glm/gtx/rotate_vector.hpp>


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

void PhysicsEngine::grow_buffers(ResiduaEngine* engine,
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

    DescriptorWriter w;
    w.write_buffer(0, new_rb_buf.buffer,       new_cap_bodies * sizeof(RigidBodyDrawGPU), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_buffer(1, new_pixel_buf.buffer,    new_cap_pixels * sizeof(glm::vec4),        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_image (2, output_screen.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL,     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    w.write_buffer(3, new_active_buf.buffer,   new_cap_bodies * sizeof(uint32_t),         0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    w.write_buffer(4, sdf_data_buf.buffer,     cap_sdf_floats * sizeof(float),            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
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

void PhysicsEngine::grow_sdf_buffer(ResiduaEngine* engine, uint32_t needed_floats)
{
    uint32_t new_cap = std::max(cap_sdf_floats, 1u << 20); // start at 4MB
    while (new_cap < needed_floats) new_cap *= 2;
    if (new_cap == cap_sdf_floats) return;

    vkDeviceWaitIdle(engine->_device);

    AllocatedBuffer new_buf = engine->create_buffer(
        new_cap * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    if (next_sdf_float > 0)
        std::memcpy(new_buf.info.pMappedData,
                    sdf_data_buf.info.pMappedData,
                    next_sdf_float * sizeof(float));

    if (cap_sdf_floats > 0)
        vmaDestroyBuffer(engine->_allocator, sdf_data_buf.buffer, sdf_data_buf.allocation);

    sdf_data_buf   = new_buf;
    cap_sdf_floats = new_cap;

    if (manifold_gen_desc != VK_NULL_HANDLE) {
        DescriptorWriter w;
        w.write_buffer(1, sdf_data_buf.buffer, cap_sdf_floats * sizeof(float), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(engine->_device, manifold_gen_desc);
    }
    if (draw_desc != VK_NULL_HANDLE) {
        DescriptorWriter w;
        w.write_buffer(4, sdf_data_buf.buffer, cap_sdf_floats * sizeof(float), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(engine->_device, draw_desc);
    }
}

void PhysicsEngine::init(ResiduaEngine* engine)
{
    engine_ref = engine;
    VkDevice device = engine->_device;

    output_screen = engine->create_image(
        { PHYSICS_WIDTH, PHYSICS_HEIGHT, 1 },
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    engine->immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, output_screen.image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });

    // ── Descriptor pool ───────────────────────────────────────────────────────

    DescriptorAllocator::PoolSizeRatio pool_ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2.f },
    };
    desc_allocator.init_pool(device, 8, pool_ratios);

    grow_sdf_buffer(engine, 1u << 20);

    // ── Draw pipeline ─────────────────────────────────────────────────────────
    {
        auto& pl = draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // rb_draw_buf
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pixel_colors_buf
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_indices_buf
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // sdf_data_buf
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
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/rigidbody_draw.comp.spv", pl.layout);

        draw_desc = desc_allocator.allocate(device, pl.desc_layout);
    }

    grow_buffers(engine, 16, 1 << 16);

    // ── Gap-fill pipeline ─────────────────────────────────────────────────────
    {
        auto& pl = gap_fill_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // output_screen
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
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/gap_fill.comp.spv", pl.layout);

        gap_fill_desc = desc_allocator.allocate(device, pl.desc_layout);

        DescriptorWriter w;
        w.write_image(0, output_screen.imageView, VK_NULL_HANDLE,
                      VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(device, gap_fill_desc);
    }

    // ── Manifold-gen pipeline ─────────────────────────────────────────────────
    {
        auto& pl = manifold_gen_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // body_info_buf
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // sdf_data_buf
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // body_verts_buf
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // pair_buf
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // contact_buf
        b.add_binding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // counter_buf
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        struct MgenPC { uint32_t pair_count; uint32_t max_contacts; float sdf_scale; };
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MgenPC) };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/manifold_gen.comp.spv", pl.layout);

        manifold_gen_desc = desc_allocator.allocate(device, pl.desc_layout);

        body_info_buf = engine->create_buffer(
            MAX_GPU_BODIES   * sizeof(GPUBodyInfo),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        body_verts_buf = engine->create_buffer(
            MAX_GPU_VERTS    * sizeof(glm::vec2),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        pair_buf = engine->create_buffer(
            MAX_GPU_PAIRS    * sizeof(GPUPair),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        contact_buf = engine->create_buffer(
            MAX_GPU_CONTACTS * sizeof(GPUContact),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        counter_buf = engine->create_buffer(
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

        DescriptorWriter w;
        w.write_buffer(0, body_info_buf.buffer,  MAX_GPU_BODIES   * sizeof(GPUBodyInfo), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, sdf_data_buf.buffer,   cap_sdf_floats   * sizeof(float),       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(2, body_verts_buf.buffer, MAX_GPU_VERTS    * sizeof(glm::vec2),   0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(3, pair_buf.buffer,       MAX_GPU_PAIRS    * sizeof(GPUPair),     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(4, contact_buf.buffer,    MAX_GPU_CONTACTS * sizeof(GPUContact),  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(5, counter_buf.buffer,    sizeof(uint32_t),                       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, manifold_gen_desc);
    }

    // ── Body-coloring pipeline (Jones-Plassmann) ─────────────────────────────
    {
        auto& pl = coloring_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // adj_offsets
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // adj_list
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // priorities
        b.add_binding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // active_flags
        b.add_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // colors
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
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/body_coloring.comp.spv", pl.layout);

        col_adj_offsets_buf = engine->create_buffer(
            (MAX_COLOR_BODIES + 1) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        col_adj_list_buf = engine->create_buffer(
            MAX_COLOR_ADJ * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        col_priorities_buf = engine->create_buffer(
            MAX_COLOR_BODIES * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        col_active_buf = engine->create_buffer(
            MAX_COLOR_BODIES * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        col_colors_buf = engine->create_buffer(
            MAX_COLOR_BODIES * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY);

        coloring_desc = desc_allocator.allocate(device, pl.desc_layout);
        DescriptorWriter w;
        w.write_buffer(0, col_adj_offsets_buf.buffer, (MAX_COLOR_BODIES+1)*sizeof(uint32_t), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(1, col_adj_list_buf.buffer,    MAX_COLOR_ADJ*sizeof(uint32_t),        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(2, col_priorities_buf.buffer,  MAX_COLOR_BODIES*sizeof(uint32_t),     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(3, col_active_buf.buffer,      MAX_COLOR_BODIES*sizeof(uint32_t),     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_buffer(4, col_colors_buf.buffer,      MAX_COLOR_BODIES*sizeof(uint32_t),     0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.update_set(device, coloring_desc);
    }

    engine->_mainDeletionQueue.push_function([this, device]() {
        vkDestroyPipeline(device, draw_pl.pipeline,              nullptr);
        vkDestroyPipeline(device, gap_fill_pl.pipeline,          nullptr);
        vkDestroyPipeline(device, manifold_gen_pl.pipeline,      nullptr);
        vkDestroyPipeline(device, coloring_pl.pipeline,          nullptr);
        vkDestroyPipelineLayout(device, draw_pl.layout,              nullptr);
        vkDestroyPipelineLayout(device, gap_fill_pl.layout,          nullptr);
        vkDestroyPipelineLayout(device, manifold_gen_pl.layout,      nullptr);
        vkDestroyPipelineLayout(device, coloring_pl.layout,          nullptr);
        vkDestroyDescriptorSetLayout(device, draw_pl.desc_layout,         nullptr);
        vkDestroyDescriptorSetLayout(device, gap_fill_pl.desc_layout,     nullptr);
        vkDestroyDescriptorSetLayout(device, manifold_gen_pl.desc_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, coloring_pl.desc_layout,     nullptr);
        desc_allocator.destroy_pool(device);
    });
}

uint32_t PhysicsEngine::add_body(ResiduaEngine* engine, RigidBody2 body)
{
    const uint32_t w = body.sprite.width;
    const uint32_t h = body.sprite.height;
    const uint32_t n = w * h;

    const uint32_t pixel_offset = next_pixel;
    next_pixel += n;

    const uint32_t sdf_size   = body.sdf_w * body.sdf_h;
    const uint32_t sdf_offset = next_sdf_float;

    uint32_t slot = world.add_body(body);
    grow_buffers(engine, (uint32_t)world.bodies.size(), next_pixel);
    grow_sdf_buffer(engine, next_sdf_float + sdf_size);
    next_sdf_float += sdf_size;

    if (slot >= (uint32_t)body_draw_info.size())
        body_draw_info.resize(slot + 1);

    const uint32_t edge_offset = next_vert;
    const uint32_t edge_count  = (uint32_t)body.shape.size();
    next_vert += edge_count;

    body_draw_info[slot] = { pixel_offset, w, h, sdf_offset, edge_offset, edge_count };

    // Upload local-space polygon vertices for this body.
    auto* vert_dst = static_cast<glm::vec2*>(body_verts_buf.info.pMappedData);
    std::memcpy(vert_dst + edge_offset, body.shape.data(), edge_count * sizeof(glm::vec2));

    auto* px_dst = static_cast<glm::vec4*>(pixel_colors_buf.info.pMappedData);
    std::memcpy(px_dst + pixel_offset, body.sprite.pixels.data(), n * sizeof(glm::vec4));

    auto* sdf_dst = static_cast<float*>(sdf_data_buf.info.pMappedData);
    std::memcpy(sdf_dst + sdf_offset, body.sdf.data(), sdf_size * sizeof(float));

    return slot;
}

uint32_t PhysicsEngine::add_static_rect(ResiduaEngine* engine, glm::vec2 center, float w, float h)
{
    uint32_t slot = world.add_static_rect(center, w, h);
    const RigidBody2& body = world.bodies[slot];

    grow_sdf_buffer(engine, next_sdf_float + body.sdf_w * body.sdf_h);

    if (slot >= (uint32_t)body_draw_info.size())
        body_draw_info.resize(slot + 1);

    const uint32_t sdf_offset  = next_sdf_float;
    next_sdf_float += body.sdf_w * body.sdf_h;
    std::memcpy(static_cast<float*>(sdf_data_buf.info.pMappedData) + sdf_offset,
                body.sdf.data(), body.sdf_w * body.sdf_h * sizeof(float));

    const uint32_t edge_offset = next_vert;
    const uint32_t edge_count  = (uint32_t)body.shape.size();
    next_vert += edge_count;
    std::memcpy(static_cast<glm::vec2*>(body_verts_buf.info.pMappedData) + edge_offset,
                body.shape.data(), edge_count * sizeof(glm::vec2));

    body_draw_info[slot] = { 0, 0, 0, sdf_offset, edge_offset, edge_count };

    return slot;
}

void PhysicsEngine::remove_body(uint32_t idx)
{
    world.remove_body(idx);
}

void PhysicsEngine::clear()
{
    world.bodies.clear();
    world.active.clear();
    world.open_slots.clear();
    world.forces.clear();
    world.colors.clear();
    world.precomputed_contacts.clear();
    world.max_color = 0;

    next_pixel     = 0;
    next_sdf_float = 0;
    next_vert      = 0;
    body_draw_info.clear();
}

std::optional<uint32_t> PhysicsEngine::body_at(glm::vec2 world_pos) const
{
    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (!world.active[i]) continue;
        const RigidBody2&   rb  = world.bodies[i];
        const BodyGPUInfo& bdi = body_draw_info[i];

        glm::vec2 delta = world_pos - glm::vec2(rb.position);
        float c = std::cos(-rb.position.z), s = std::sin(-rb.position.z);
        glm::vec2 local = { c * delta.x - s * delta.y,
                            s * delta.x + c * delta.y };

        local += glm::vec2((float)bdi.body_w, (float)bdi.body_h) * 0.5f - 0.5f;

        int px = (int)std::floor(local.x), py = (int)std::floor(local.y);
        if (px < 0 || py < 0 ||
            (uint32_t)px >= bdi.body_w || (uint32_t)py >= bdi.body_h) continue;

        if (rb.sprite.pixels[(uint32_t)py * bdi.body_w + (uint32_t)px].a > 0.5f)
            return i;
    }
    return std::nullopt;
}

void PhysicsEngine::dispatch(VkCommandBuffer cmd, float dt)
{
    // 1. Broadphase — creates manifold stubs, clears precomputed_contacts.
    world.prepare();

    // 2. Jones-Plassmann body coloring — must run before solve so solver can use colors.
    // TODO: move into its own function / file
    {
        const uint32_t body_count = (uint32_t)world.bodies.size();
        if (body_count <= MAX_COLOR_BODIES && coloring_pl.pipeline != VK_NULL_HANDLE) {
            // build adjacency grid
            std::vector<std::vector<uint32_t>> adj(body_count);
            for (const auto& f : world.forces) {
                if (!f) continue;
                uint32_t a = f->bodyA, b = f->bodyB;
                if (a >= body_count || b >= body_count) continue;
                if (world.bodies[a].inv_mass == 0.f || world.bodies[b].inv_mass == 0.f) continue;
                adj[a].push_back(b);
                adj[b].push_back(a);
            }

            auto* off_ptr = static_cast<uint32_t*>(col_adj_offsets_buf.info.pMappedData);
            auto* lst_ptr = static_cast<uint32_t*>(col_adj_list_buf.info.pMappedData);
            auto* pri_ptr = static_cast<uint32_t*>(col_priorities_buf.info.pMappedData);
            auto* act_ptr = static_cast<uint32_t*>(col_active_buf.info.pMappedData);

            uint32_t adj_total = 0;
            for (uint32_t i = 0; i < body_count; i++) {
                off_ptr[i] = adj_total;
                if (adj_total + (uint32_t)adj[i].size() <= MAX_COLOR_ADJ) {
                    for (uint32_t nb : adj[i]) lst_ptr[adj_total++] = nb;
                } else {
                    adj_total = MAX_COLOR_ADJ; // truncate
                }
            }
            off_ptr[body_count] = adj_total;

            for (uint32_t i = 0; i < body_count; i++) {
                pri_ptr[i] = (uint32_t)rand();
                act_ptr[i] = (world.active[i] && world.bodies[i].inv_mass > 0.f) ? 1u : 0u;
            }

            engine_ref->immediate_submit([&](VkCommandBuffer icmd) {
                vkCmdFillBuffer(icmd, col_colors_buf.buffer, 0,
                                body_count * sizeof(uint32_t), 0u);

                VkBufferMemoryBarrier2 pre_barriers[5] = {
                    {
                        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                        .buffer = col_colors_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                    },
                    {
                        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .buffer = col_adj_offsets_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                    },
                    {
                        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .buffer = col_adj_list_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                    },
                    {
                        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .buffer = col_priorities_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                    },
                    {
                        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                        .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                        .buffer = col_active_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                    },
                };
                VkDependencyInfo pre_dep {
                    .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .bufferMemoryBarrierCount  = 5,
                    .pBufferMemoryBarriers     = pre_barriers,
                };
                vkCmdPipelineBarrier2(icmd, &pre_dep);

                vkCmdBindPipeline(icmd, VK_PIPELINE_BIND_POINT_COMPUTE, coloring_pl.pipeline);
                vkCmdBindDescriptorSets(icmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    coloring_pl.layout, 0, 1, &coloring_desc, 0, nullptr);
                vkCmdPushConstants(icmd, coloring_pl.layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &body_count);

                const uint32_t groups = (body_count + 63) / 64;
                for (uint32_t iter = 0; iter < COLOR_JP_ITERS; iter++) {
                    vkCmdDispatch(icmd, groups, 1, 1);
                    if (iter + 1 < COLOR_JP_ITERS) {
                        // Barrier: colors writes from this iter visible to reads in next iter.
                        VkBufferMemoryBarrier2 iter_barrier {
                            .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                            .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                            .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                            .buffer = col_colors_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE,
                        };
                        VkDependencyInfo iter_dep {
                            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .bufferMemoryBarrierCount = 1,
                            .pBufferMemoryBarriers    = &iter_barrier,
                        };
                        vkCmdPipelineBarrier2(icmd, &iter_dep);
                    }
                }
            });

            // Read back colors into world for the solver to use.
            auto* col_ptr = static_cast<const uint32_t*>(col_colors_buf.info.pMappedData);
            world.colors.assign(col_ptr, col_ptr + body_count);
            world.max_color = 0;
            for (uint32_t i = 0; i < body_count; i++)
                if (world.colors[i] != 0xFFFFFFFFu && world.colors[i] > world.max_color)
                    world.max_color = world.colors[i];
        } else {
            // Too many body slots for GPU coloring — clear so solver falls back to
            // single sequential group (correct but unparallelized).
            printf("too many bodies");
            world.colors.clear();
        }
    }

    // 3. Upload body info (current poses) and build pair list from manifolds.
    auto* body_info_ptr = static_cast<GPUBodyInfo*>(body_info_buf.info.pMappedData);
    auto* pair_ptr      = static_cast<GPUPair*>(pair_buf.info.pMappedData);
    uint32_t pair_count = 0;

    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (i >= MAX_GPU_BODIES) break;
        const RigidBody2& rb = world.bodies[i];
        const BodyGPUInfo& info = (i < body_draw_info.size()) ? body_draw_info[i] : BodyGPUInfo{};
        body_info_ptr[i] = {
            glm::vec2(rb.position),
            rb.position.z,
            info.edge_offset,
            info.edge_count,
            info.sdf_offset,
            rb.sdf_w,
            rb.sdf_h,
            rb.com_local,
        };
    }

    for (const auto& f : world.forces) {
        auto* m = dynamic_cast<Manifold*>(f.get());
        if (!m || pair_count >= MAX_GPU_PAIRS) continue;
        pair_ptr[pair_count++] = { m->bodyA, m->bodyB };
    }

    // GPU manifold-gen pass (synchronous via immediate_submit).
    if (pair_count > 0) {
        *static_cast<uint32_t*>(counter_buf.info.pMappedData) = 0u;

        struct MgenPC { uint32_t pair_count; uint32_t max_contacts; float sdf_scale; }
            pc { pair_count, MAX_GPU_CONTACTS, (float)SDF_SCALE };

        engine_ref->immediate_submit([&, pc](VkCommandBuffer icmd) {
            vkCmdBindPipeline(icmd, VK_PIPELINE_BIND_POINT_COMPUTE, manifold_gen_pl.pipeline);
            vkCmdBindDescriptorSets(icmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                manifold_gen_pl.layout, 0, 1, &manifold_gen_desc, 0, nullptr);
            vkCmdPushConstants(icmd, manifold_gen_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(icmd, (pair_count + 63) / 64, 1, 1);
        });

        // 4. Read back contacts and distribute to precomputed_contacts.
        uint32_t num_contacts = std::min(
            *static_cast<uint32_t*>(counter_buf.info.pMappedData),
            MAX_GPU_CONTACTS);

        auto* contacts = static_cast<GPUContact*>(contact_buf.info.pMappedData);

        // Group raw GPU contacts by pair.
        std::unordered_map<uint64_t, std::vector<ManifoldPoint>> raw;
        for (uint32_t ci = 0; ci < num_contacts; ci++) {
            const GPUContact& gc = contacts[ci];
            uint64_t key = ((uint64_t)gc.body_a << 32) | gc.body_b;
            raw[key].push_back({ gc.world_pt, gc.normal, gc.depth });
        }

        for (auto& [key, pts] : raw) {
            if (pts.empty()) continue;
            world.precomputed_contacts[key] = std::move(pts);
        }

        world.stats.gpu_contacts = num_contacts;
    }

    world.stats.gpu_pairs      = pair_count;
    world.stats.gpu_verts      = next_vert;
    world.stats.gpu_sdf_floats = next_sdf_float;

    // 5. Solve (warm-start + AVBD iterations).
    world.solve(dt);

    // 2. Upload draw data for all active bodies.
    auto* rb_ptr     = static_cast<RigidBodyDrawGPU*>(rb_draw_buf.info.pMappedData);
    auto* active_ptr = static_cast<uint32_t*>(active_indices_buf.info.pMappedData);
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < (uint32_t)world.bodies.size(); i++) {
        if (!world.active[i]) continue;
        if (i >= (uint32_t)body_draw_info.size()) continue;
        const BodyGPUInfo& bdi = body_draw_info[i];
        if (bdi.body_w == 0 || bdi.body_h == 0) continue;

        const RigidBody2& rb = world.bodies[i];
        RigidBodyDrawGPU draw{};
        draw.position         = glm::vec2(rb.position);
        draw.rotation         = rb.position.z;
        draw.total_mass       = rb.mass;
        draw.velocity         = glm::vec2(rb.velocity);
        draw.angular_velocity = rb.velocity.z;
        draw.I_com            = rb.inertia;
        draw.pixel_index      = bdi.pixel_index;
        draw.body_w           = bdi.body_w;
        draw.body_h           = bdi.body_h;
        draw.sdf_offset       = bdi.sdf_offset;
        draw.com_local        = rb.com_local;
        rb_ptr[i] = draw;
        active_ptr[active_count++] = i;
    }

    if (active_count == 0) return;

    // 3. Clear output_screen.
    VkClearColorValue clear_color{};
    VkImageSubresourceRange full_range {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .levelCount = 1,
        .layerCount = 1,
    };
    vkCmdClearColorImage(cmd, output_screen.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clear_color, 1, &full_range);

    // 4. Barrier: clear → draw compute.
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
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &img_clear_barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep1);

    // 5. Draw pass — one workgroup per active body.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, draw_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        draw_pl.layout, 0, 1, &draw_desc, 0, nullptr);
    vkCmdPushConstants(cmd, draw_pl.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &active_count);
    vkCmdDispatch(cmd, active_count, 1, 1);

    // 6. Barrier: draw → gap_fill.
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
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &img_draw_barrier,
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

PhysicsStats PhysicsEngine::get_stats(){
    return world.stats;
}
