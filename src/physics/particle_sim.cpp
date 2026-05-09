#include "particle_sim.h"
#include "physics_engine.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_initializers.h"
#include "../renderer/vk_pipelines.h"
#include <fmt/core.h>
#include <cstring>

static constexpr uint32_t GRID_NODES =
    PHYSICS_WIDTH * PHYSICS_HEIGHT;  // always allocate full-res; cell_size trims usage

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                        VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("ParticleSim: failed to load {}\n", spv);
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

static void full_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                       | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT
                       | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                       | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
                       | VK_ACCESS_2_SHADER_WRITE_BIT
                       | VK_ACCESS_2_TRANSFER_WRITE_BIT,
    };
    VkDependencyInfo dep {
        .sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount  = 1,
        .pMemoryBarriers     = &mb,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ─── init ─────────────────────────────────────────────────────────────────────

void ParticleSim::init(ResiduaEngine* engine, VkImageView output_view)
{
    engine_ref = engine;
    VkDevice device = engine->_device;

    // ── Particle buffer ───────────────────────────────────────────────────────
    particle_buf = engine->create_buffer(
        MAX_PARTICLES * sizeof(Particle),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // ── Grid ping-pong buffers ────────────────────────────────────────────────
    for (int i = 0; i < 2; i++) {
        grid_buf[i] = engine->create_buffer(
            GRID_NODES * 4 * sizeof(int32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    }

    // ── Descriptor pool ───────────────────────────────────────────────────────
    DescriptorAllocator::PoolSizeRatio ratios[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8.f },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2.f },
    };
    desc_alloc.init_pool(device, 8, ratios);

    // ── G2P2G pipeline ────────────────────────────────────────────────────────
    {
        auto& pl = g2p2g_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // particles
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // grid_src
        b.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // grid_dst
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        // Set=1 layout: vel_image for SDF particle push-out (filled in setup_collision_image)
        {
            DescriptorLayoutBuilder bv;
            bv.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
            g2p2g_vel_layout = bv.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
        }

        VkPushConstantRange pc_range {
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(G2P2GPushConstants)
        };
        VkDescriptorSetLayout set_layouts[2] = { pl.desc_layout, g2p2g_vel_layout };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 2,
            .pSetLayouts            = set_layouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/particle_g2p2g.comp.spv", pl.layout);

        // Two descriptor sets: one per ping-pong direction.
        // desc[0]: src=grid[0], dst=grid[1]
        // desc[1]: src=grid[1], dst=grid[0]
        VkDeviceSize grid_sz = GRID_NODES * 4 * sizeof(int32_t);
        VkDeviceSize part_sz = MAX_PARTICLES * sizeof(Particle);

        for (int d = 0; d < 2; d++) {
            g2p2g_desc[d] = desc_alloc.allocate(device, pl.desc_layout);
            DescriptorWriter w;
            w.write_buffer(0, particle_buf.buffer,  part_sz, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.write_buffer(1, grid_buf[d].buffer,   grid_sz, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // src
            w.write_buffer(2, grid_buf[1-d].buffer, grid_sz, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // dst
            w.update_set(device, g2p2g_desc[d]);
        }
    }

    // ── Particle draw pipeline ────────────────────────────────────────────────
    {
        auto& pl = draw_pl;
        DescriptorLayoutBuilder b;
        b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // particles
        b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // output_screen
        pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc_range {
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleDrawPushConstants)
        };
        VkPipelineLayoutCreateInfo li {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &pl.desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pc_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
        pl.pipeline = make_compute_pipeline(device,
            "../shaders/particle_draw.comp.spv", pl.layout);

        draw_desc = desc_alloc.allocate(device, pl.desc_layout);
        DescriptorWriter w;
        w.write_buffer(0, particle_buf.buffer, MAX_PARTICLES * sizeof(Particle),
                       0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_image(1, output_view, VK_NULL_HANDLE,
                      VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(device, draw_desc);
    }

    // Register cleanup
    engine->_mainDeletionQueue.push_function([this, device]() {
        vkDestroyPipeline(device,      g2p2g_pl.pipeline,     nullptr);
        vkDestroyPipelineLayout(device, g2p2g_pl.layout,      nullptr);
        vkDestroyDescriptorSetLayout(device, g2p2g_pl.desc_layout, nullptr);
        vkDestroyPipeline(device,      draw_pl.pipeline,      nullptr);
        vkDestroyPipelineLayout(device, draw_pl.layout,       nullptr);
        vkDestroyDescriptorSetLayout(device, draw_pl.desc_layout,  nullptr);
        desc_alloc.destroy_pool(device);
        vmaDestroyBuffer(engine_ref->_allocator,
                         particle_buf.buffer, particle_buf.allocation);
        for (int i = 0; i < 2; i++)
            vmaDestroyBuffer(engine_ref->_allocator,
                             grid_buf[i].buffer, grid_buf[i].allocation);
    });
}

// ─── setup_collision_image ────────────────────────────────────────────────────

void ParticleSim::setup_collision_image(VkImageView vel_image_view)
{
    VkDevice device = engine_ref->_device;
    auto& pl = bc_pl;

    DescriptorLayoutBuilder b;
    b.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // grid (read/write)
    b.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // vel_image (readonly)
    pl.desc_layout = b.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

    VkPushConstantRange pc_range {
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridBCPushConstants)
    };
    VkPipelineLayoutCreateInfo li {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &pl.desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pc_range,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &li, nullptr, &pl.layout));
    pl.pipeline = make_compute_pipeline(device,
        "../shaders/particle_grid_bc.comp.spv", pl.layout);

    VkDeviceSize grid_sz = GRID_NODES * 4 * sizeof(int32_t);
    // bc_desc[d] targets the dst grid for desc_idx==d, which is grid_buf[1-d]
    for (int d = 0; d < 2; d++) {
        bc_desc[d] = desc_alloc.allocate(device, pl.desc_layout);
        DescriptorWriter w;
        w.write_buffer(0, grid_buf[1-d].buffer, grid_sz, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        w.write_image(1, vel_image_view, VK_NULL_HANDLE,
                      VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        w.update_set(device, bc_desc[d]);
    }

    // G2P2G set=1: vel_image for particle-level SDF push-out
    g2p2g_vel_desc = desc_alloc.allocate(device, g2p2g_vel_layout);
    {
        DescriptorWriter wv;
        wv.write_image(0, vel_image_view, VK_NULL_HANDLE,
                       VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        wv.update_set(device, g2p2g_vel_desc);
    }

    engine_ref->_mainDeletionQueue.push_function([this, device]() {
        vkDestroyPipeline(device,            bc_pl.pipeline,    nullptr);
        vkDestroyPipelineLayout(device,       bc_pl.layout,      nullptr);
        vkDestroyDescriptorSetLayout(device,  bc_pl.desc_layout, nullptr);
        vkDestroyDescriptorSetLayout(device,  g2p2g_vel_layout,  nullptr);
    });
}

// ─── add / clear ──────────────────────────────────────────────────────────────

void ParticleSim::add_particle(const Particle& p)
{
    if (particle_count_ >= MAX_PARTICLES) return;
    auto* buf = static_cast<Particle*>(particle_buf.info.pMappedData);
    buf[particle_count_++] = p;
}

void ParticleSim::clear_particles()
{
    particle_count_ = 0;
}

// ─── dispatch_physics ─────────────────────────────────────────────────────────

void ParticleSim::dispatch_physics(VkCommandBuffer cmd, float dt)
{
    if (particle_count_ == 0) return;

    for (uint32_t iter = 0; iter < PARTICLE_ITERS; ++iter) {
        uint32_t desc_idx = iter % 2;               // which desc set to use
        uint32_t dst_buf  = 1 - desc_idx;           // grid_buf index being written

        // Clear destination grid (all zeros = empty grid node).
        vkCmdFillBuffer(cmd, grid_buf[dst_buf].buffer, 0, VK_WHOLE_SIZE, 0u);

        // Barrier: wait for fill + any prior compute writes (including vel_image from RB draw)
        full_barrier(cmd);

        // ── G2P2G ────────────────────────────────────────────────────────────
        const uint32_t cs       = settings.cell_size;
        const uint32_t pgrid_w  = PHYSICS_WIDTH  / cs;
        const uint32_t pgrid_h  = PHYSICS_HEIGHT / cs;

        G2P2GPushConstants pc {};
        pc.grid_size         = { pgrid_w, pgrid_h };
        pc.particle_count    = particle_count_;
        pc.iteration         = iter;
        pc.iteration_count   = PARTICLE_ITERS;
        pc.delta_time        = dt;
        pc.gravity           = settings.gravity;
        pc.border_friction   = settings.border_friction;
        pc.liquid_relaxation = settings.liquid_relaxation;
        pc.liquid_viscosity  = settings.liquid_viscosity;
        pc.elastic_relaxation= settings.elastic_relaxation;
        pc.elasticity_ratio  = settings.elasticity_ratio;
        pc.friction_angle    = settings.friction_angle;
        pc.plasticity        = settings.plasticity;
        pc.fixed_point_mult  = FIXED_POINT_MULT;
        pc.cell_size         = cs;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g2p2g_pl.pipeline);
        vkCmdPushConstants(cmd, g2p2g_pl.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        {
            VkDescriptorSet sets[2] = { g2p2g_desc[desc_idx], g2p2g_vel_desc };
            uint32_t set_count = (g2p2g_vel_desc != VK_NULL_HANDLE) ? 2u : 1u;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                g2p2g_pl.layout, 0, set_count, sets, 0, nullptr);
        }

        vkCmdDispatch(cmd, (particle_count_ + 63) / 64, 1, 1);
        full_barrier(cmd);

        // ── Grid boundary conditions (rigid body collision) ───────────────────
        if (bc_pl.pipeline != VK_NULL_HANDLE) {
            GridBCPushConstants bc_pc {
                { pgrid_w, pgrid_h },
                FIXED_POINT_MULT,
                settings.gravity * (float)PHYSICS_HEIGHT * dt * dt,
                dt,
                cs,
                0u,
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bc_pl.pipeline);
            vkCmdPushConstants(cmd, bc_pl.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bc_pc), &bc_pc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                bc_pl.layout, 0, 1, &bc_desc[desc_idx], 0, nullptr);
            vkCmdDispatch(cmd, (pgrid_w * pgrid_h + 63) / 64, 1, 1);
            // No trailing barrier needed: next iter's full_barrier (after FillBuffer)
            // or dispatch_render's full_barrier covers this.
        }
    }
}

// ─── dispatch_render ──────────────────────────────────────────────────────────

void ParticleSim::dispatch_render(VkCommandBuffer cmd)
{
    if (particle_count_ == 0) return;

    // Ensure both the last G2P2G and any prior image writes (gap_fill) are
    // visible before the particle draw writes to output_screen.
    full_barrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, draw_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        draw_pl.layout, 0, 1, &draw_desc, 0, nullptr);

    const uint32_t cs = settings.cell_size;
    ParticleDrawPushConstants pc {
        .grid_size      = { PHYSICS_WIDTH / cs, PHYSICS_HEIGHT / cs },
        .particle_count = particle_count_,
        .cell_size      = cs,
    };
    vkCmdPushConstants(cmd, draw_pl.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (particle_count_ + 63) / 64, 1, 1);
}
