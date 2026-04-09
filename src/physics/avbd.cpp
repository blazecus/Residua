#include "avbd.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <fmt/core.h>
#include <cassert>

// ─── Local helpers (same pattern as collision.cpp) ────────────────────────────

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                         VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("avbd: failed to load {}\n", spv);
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

// Full compute-to-compute barrier covering all storage buffers in the pipeline.
// Cheaper to write than per-buffer barriers when many buffers are in flight.
static void compute_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
    };
    VkDependencyInfo dep {
        .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers    = &mb,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ─── AvbdPipeline ─────────────────────────────────────────────────────────────

void AvbdPipeline::init(ResiduaEngine*             engine,
                         const LBvhPipeline&        lbvh,
                         const CollisionPipeline&   collision,
                         const VertexColorPipeline& vertex_color,
                         uint32_t                   max_bodies_in)
{
    max_bodies   = max_bodies_in;
    max_contacts = collision.max_contacts;

    // Cache buffer handles for dispatch barriers
    _bodies_buf   = lbvh.bodies_buf.buffer;
    _contacts_buf = collision.contacts_buf.buffer;
    _ct_count_buf = collision.contact_count_buf.buffer;

    inertia_buf = engine->create_buffer(
        max_bodies * sizeof(InertiaTarget),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    VkDevice dev = engine->_device;
    init_pipelines(dev);
    init_descriptors(dev, lbvh, collision, vertex_color);
}

void AvbdPipeline::init_pipelines(VkDevice dev)
{
    // predict: bodies, inertia  (2 bindings)
    // push: { float dt, float grav_x, float grav_y, uint body_count } = 16 bytes
    predict_pl.desc_layout = make_ssbo_layout(dev, 2);
    predict_pl.layout      = make_pipeline_layout(dev, predict_pl.desc_layout, 16);
    predict_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/avbd_predict.comp.spv", predict_pl.layout);

    // warmstart: contacts, contact_count  (2 bindings)
    // push: { float alpha_gamma, float gamma, float k_start, uint _pad } = 16 bytes
    warmstart_pl.desc_layout = make_ssbo_layout(dev, 2);
    warmstart_pl.layout      = make_pipeline_layout(dev, warmstart_pl.desc_layout, 16);
    warmstart_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/avbd_warmstart.comp.spv", warmstart_pl.layout);

    // solve: bodies, contacts, contact_count, inertia, body_colors  (5 bindings)
    // push: { uint color, uint body_count, uint max_contacts, float dt, float mu } = 20 bytes
    solve_pl.desc_layout = make_ssbo_layout(dev, 5);
    solve_pl.layout      = make_pipeline_layout(dev, solve_pl.desc_layout, 20);
    solve_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/avbd_solve.comp.spv", solve_pl.layout);

    // update_lambda: bodies, contacts, contact_count  (3 bindings)
    // push: { float beta, float mu, uint max_contacts, uint _pad } = 16 bytes
    update_lambda_pl.desc_layout = make_ssbo_layout(dev, 3);
    update_lambda_pl.layout      = make_pipeline_layout(dev, update_lambda_pl.desc_layout, 16);
    update_lambda_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/avbd_update_lambda.comp.spv", update_lambda_pl.layout);

    // integrate: bodies (rw), shapes (r), aabbs (w)  — 3 bindings
    // push: { float dt, uint body_count } = 8 bytes
    integrate_pl.desc_layout = make_ssbo_layout(dev, 3);
    integrate_pl.layout      = make_pipeline_layout(dev, integrate_pl.desc_layout, 8);
    integrate_pl.pipeline    = make_compute_pipeline(
        dev, "../shaders/avbd_integrate.comp.spv", integrate_pl.layout);
}

void AvbdPipeline::init_descriptors(VkDevice                   dev,
                                     const LBvhPipeline&        lbvh,
                                     const CollisionPipeline&   collision,
                                     const VertexColorPipeline& vertex_color)
{
    DescriptorAllocatorGrowable::PoolSizeRatio avbd_ratios[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64.f }};
    desc_alloc.init(dev, 8, avbd_ratios);

    auto wb = [&](DescriptorWriter& w, uint32_t b, VkBuffer buf, size_t sz) {
        w.write_buffer(b, buf, sz, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    const size_t body_sz    = lbvh.max_bodies * sizeof(RigidBody);
    const size_t contact_sz = max_contacts    * sizeof(Contact);
    const size_t inertia_sz = max_bodies      * sizeof(InertiaTarget);
    const size_t color_sz   = max_bodies      * sizeof(uint32_t);

    // predict: bodies (rw), inertia (w)
    {
        predict_desc = desc_alloc.allocate(dev, predict_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.bodies_buf.buffer,             body_sz);
        wb(w, 1, inertia_buf.buffer,                 inertia_sz);
        w.update_set(dev, predict_desc);
    }

    // warmstart: contacts (rw), contact_count (r)
    {
        warmstart_desc = desc_alloc.allocate(dev, warmstart_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, collision.contacts_buf.buffer,       contact_sz);
        wb(w, 1, collision.contact_count_buf.buffer,  sizeof(uint32_t));
        w.update_set(dev, warmstart_desc);
    }

    // solve: bodies (rw), contacts (r), contact_count (r), inertia (r), colors (r)
    {
        solve_desc = desc_alloc.allocate(dev, solve_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.bodies_buf.buffer,                      body_sz);
        wb(w, 1, collision.contacts_buf.buffer,                contact_sz);
        wb(w, 2, collision.contact_count_buf.buffer,           sizeof(uint32_t));
        wb(w, 3, inertia_buf.buffer,                           inertia_sz);
        wb(w, 4, vertex_color.body_colors_buf.buffer,          color_sz);
        w.update_set(dev, solve_desc);
    }

    // update_lambda: bodies (r), contacts (rw), contact_count (r)
    {
        update_lambda_desc = desc_alloc.allocate(dev, update_lambda_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.bodies_buf.buffer,                      body_sz);
        wb(w, 1, collision.contacts_buf.buffer,                contact_sz);
        wb(w, 2, collision.contact_count_buf.buffer,           sizeof(uint32_t));
        w.update_set(dev, update_lambda_desc);
    }

    // integrate: bodies (rw), shapes (r), aabbs (w)
    {
        integrate_desc = desc_alloc.allocate(dev, integrate_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, lbvh.bodies_buf.buffer,  body_sz);
        wb(w, 1, lbvh.shapes_buf.buffer,  lbvh.max_bodies * sizeof(CollisionShape));
        wb(w, 2, lbvh.aabbs_buf.buffer,   lbvh.max_bodies * sizeof(AABB));
        w.update_set(dev, integrate_desc);
    }
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void AvbdPipeline::dispatch(VkCommandBuffer   cmd,
                              uint32_t          body_count,
                              uint32_t          n_colors,
                              float             dt,
                              const AvbdParams& p)
{
    assert(body_count >= 1);

    const uint32_t body_groups    = (body_count   + 63u) / 64u;
    const uint32_t contact_groups = (max_contacts + 63u) / 64u;

    // ── Predict (lines 3-4) ───────────────────────────────────────────────────
    {
        struct { float dt, gx, gy; uint32_t n; } pc {
            dt, p.gravity.x, p.gravity.y, body_count
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, predict_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 predict_pl.layout, 0, 1, &predict_desc, 0, nullptr);
        vkCmdPushConstants(cmd, predict_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, body_groups, 1, 1);
    }

    // ── Warm-start λ and k (lines 5-6) ───────────────────────────────────────
    // No barrier needed before warmstart: it touches contacts_buf (disjoint from predict).
    {
        struct { float ag, g, ks; uint32_t pad; } pc {
            p.alpha * p.gamma, p.gamma, p.k_start, 0u
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, warmstart_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 warmstart_pl.layout, 0, 1, &warmstart_desc, 0, nullptr);
        vkCmdPushConstants(cmd, warmstart_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, contact_groups, 1, 1);
    }

    // Barrier: predict writes bodies + inertia; warmstart writes contacts.
    // Both must be visible to the solve loop.
    compute_barrier(cmd);

    // ── Solver iterations (lines 7-36) ───────────────────────────────────────
    for (uint32_t iter = 0; iter < p.n_iter; iter++) {

        // ── Color loop (lines 8-25) ───────────────────────────────────────────
        for (uint32_t c = 0; c < n_colors; c++) {
            struct { uint32_t color, n, mc; float dt, mu; } pc {
                c, body_count, max_contacts, dt, p.mu
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, solve_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     solve_pl.layout, 0, 1, &solve_desc, 0, nullptr);
            vkCmdPushConstants(cmd, solve_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, body_groups, 1, 1);

            // Each color pass writes bodies; the next color reads them.
            compute_barrier(cmd);
        }

        // ── λ and k update (lines 26-35) ─────────────────────────────────────
        {
            struct { float beta, mu; uint32_t mc, pad; } pc {
                p.beta, p.mu, max_contacts, 0u
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, update_lambda_pl.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     update_lambda_pl.layout, 0, 1, &update_lambda_desc, 0, nullptr);
            vkCmdPushConstants(cmd, update_lambda_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, contact_groups, 1, 1);

            // λ writes must be visible to next iteration's solve.
            compute_barrier(cmd);
        }
    }

    // ── Integrate (line 37) ───────────────────────────────────────────────────
    {
        struct { float dt; uint32_t n; } pc { dt, body_count };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 integrate_pl.layout, 0, 1, &integrate_desc, 0, nullptr);
        vkCmdPushConstants(cmd, integrate_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, body_groups, 1, 1);
        compute_barrier(cmd);
    }
}
