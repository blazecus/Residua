#include "lbvh.h"
#include "../renderer/residua_engine.h"
#include "../renderer/vk_pipelines.h"
#include "../renderer/vk_initializers.h"
#include <fmt/core.h>
#include <algorithm>
#include <cassert>
#include <cmath>

// ─── Helpers (local) ─────────────────────────────────────────────────────────

static VkPipeline make_compute_pipeline(VkDevice device, const char* spv,
                                         VkPipelineLayout layout)
{
    VkShaderModule mod;
    if (!vkutil::load_shader_module(spv, device, &mod)) {
        fmt::print("lbvh: failed to load {}\n", spv);
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

static void transfer_to_compute_barrier(VkCommandBuffer cmd, VkBuffer buf)
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

// Build a VkDescriptorSetLayout from a list of storage-buffer bindings.
static VkDescriptorSetLayout make_ssbo_layout(VkDevice device, uint32_t count)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(count);
    for (uint32_t i = 0; i < count; i++) {
        bindings[i] = {
            .binding         = i,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo ci {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count,
        .pBindings    = bindings.data(),
    };
    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
    return layout;
}

// Build a VkPipelineLayout for a compute shader with one descriptor set and
// push constants of push_size bytes.
static VkPipelineLayout make_pipeline_layout(VkDevice device,
                                              VkDescriptorSetLayout desc,
                                              uint32_t push_size)
{
    VkPushConstantRange pc { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                             .offset = 0, .size = push_size };
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

// ─── LBvhPipeline ─────────────────────────────────────────────────────────────

void LBvhPipeline::init(ResiduaEngine* engine, uint32_t max_b)
{
    max_bodies = max_b;
    VkDevice dev = engine->_device;

    const uint32_t max_blocks   = (max_bodies + 255) / 256;
    const uint32_t node_count   = 2 * max_bodies - 1;   // upper bound

    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags ssbo_td = ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    bodies_buf       = engine->create_buffer(max_bodies * sizeof(RigidBody),
                           ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    shapes_buf       = engine->create_buffer(max_bodies * sizeof(CollisionShape),
                           ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    aabbs_buf        = engine->create_buffer(max_bodies * sizeof(AABB),
                           ssbo | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    morton_buf       = engine->create_buffer(max_bodies * sizeof(uint32_t),  ssbo,     VMA_MEMORY_USAGE_GPU_ONLY);
    indices_buf      = engine->create_buffer(max_bodies * sizeof(uint32_t),  ssbo,     VMA_MEMORY_USAGE_GPU_ONLY);
    morton_tmp_buf   = engine->create_buffer(max_bodies * sizeof(uint32_t),  ssbo,     VMA_MEMORY_USAGE_GPU_ONLY);
    indices_tmp_buf  = engine->create_buffer(max_bodies * sizeof(uint32_t),  ssbo,     VMA_MEMORY_USAGE_GPU_ONLY);
    block_hist_buf   = engine->create_buffer(max_blocks * 256 * sizeof(uint32_t), ssbo, VMA_MEMORY_USAGE_GPU_ONLY);
    bvh_nodes_buf    = engine->create_buffer(node_count * sizeof(BVHNode),   ssbo,     VMA_MEMORY_USAGE_GPU_ONLY);
    refit_flags_buf  = engine->create_buffer(node_count * sizeof(int32_t),   ssbo_td,  VMA_MEMORY_USAGE_GPU_ONLY);

    init_pipelines(dev);
    init_descriptors(dev);
}

void LBvhPipeline::init_pipelines(VkDevice dev)
{
    // ── Morton ────────────────────────────────────────────────────────────────
    morton_pl.desc_layout = make_ssbo_layout(dev, 3);  // aabbs, morton_codes, body_indices
    struct MortonPC { uint32_t n; float wx, wy, iwx, iwy; };
    morton_pl.layout      = make_pipeline_layout(dev, morton_pl.desc_layout, sizeof(MortonPC));
    morton_pl.pipeline    = make_compute_pipeline(dev, "../shaders/lbvh_morton.comp.spv", morton_pl.layout);

    // ── Radix count ───────────────────────────────────────────────────────────
    radix_count_pl.desc_layout = make_ssbo_layout(dev, 2);  // keys, block_hist
    struct RadixCountPC { uint32_t n, bit_shift; };
    radix_count_pl.layout   = make_pipeline_layout(dev, radix_count_pl.desc_layout, sizeof(RadixCountPC));
    radix_count_pl.pipeline = make_compute_pipeline(dev, "../shaders/lbvh_radix_count.comp.spv", radix_count_pl.layout);

    // ── Radix scan ────────────────────────────────────────────────────────────
    radix_scan_pl.desc_layout = make_ssbo_layout(dev, 1);  // block_hist
    radix_scan_pl.layout      = make_pipeline_layout(dev, radix_scan_pl.desc_layout, sizeof(uint32_t));
    radix_scan_pl.pipeline    = make_compute_pipeline(dev, "../shaders/lbvh_radix_scan.comp.spv", radix_scan_pl.layout);

    // ── Radix scatter ─────────────────────────────────────────────────────────
    radix_scatter_pl.desc_layout = make_ssbo_layout(dev, 5); // in_keys, in_vals, out_keys, out_vals, block_hist
    struct RadixScatterPC { uint32_t n, bit_shift; };
    radix_scatter_pl.layout   = make_pipeline_layout(dev, radix_scatter_pl.desc_layout, sizeof(RadixScatterPC));
    radix_scatter_pl.pipeline = make_compute_pipeline(dev, "../shaders/lbvh_radix_scatter.comp.spv", radix_scatter_pl.layout);

    // ── BVH build ─────────────────────────────────────────────────────────────
    build_pl.desc_layout = make_ssbo_layout(dev, 2);  // sorted_keys, nodes
    build_pl.layout      = make_pipeline_layout(dev, build_pl.desc_layout, sizeof(uint32_t));
    build_pl.pipeline    = make_compute_pipeline(dev, "../shaders/lbvh_build.comp.spv", build_pl.layout);

    // ── Refit ─────────────────────────────────────────────────────────────────
    refit_pl.desc_layout = make_ssbo_layout(dev, 4);  // body_aabbs, sorted_indices, nodes, flags
    refit_pl.layout      = make_pipeline_layout(dev, refit_pl.desc_layout, sizeof(uint32_t));
    refit_pl.pipeline    = make_compute_pipeline(dev, "../shaders/lbvh_refit.comp.spv", refit_pl.layout);
}

void LBvhPipeline::init_descriptors(VkDevice dev)
{
    // Pool — estimate max sets needed: 2+2+1+2+2+1+1 = 11
    DescriptorAllocatorGrowable::PoolSizeRatio lbvh_ratios[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64.f }};
    desc_alloc.init(dev, 16, lbvh_ratios);

    auto alloc_set = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet {
        return desc_alloc.allocate(dev, layout);
    };

    auto wb = [&](DescriptorWriter& w, uint32_t binding, VkBuffer buf, size_t size) {
        w.write_buffer(binding, buf, size, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    // morton: aabbs → morton_codes, body_indices
    {
        morton_desc = alloc_set(morton_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, aabbs_buf.buffer,        max_bodies * sizeof(AABB));
        wb(w, 1, morton_buf.buffer,       max_bodies * sizeof(uint32_t));
        wb(w, 2, indices_buf.buffer,      max_bodies * sizeof(uint32_t));
        w.update_set(dev, morton_desc);
    }

    const uint32_t max_blocks = (max_bodies + 255) / 256;
    const size_t hist_size    = max_blocks * 256 * sizeof(uint32_t);

    // radix_count[0]: reads morton_buf (even passes)
    {
        radix_count_desc[0] = alloc_set(radix_count_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, morton_buf.buffer,      max_bodies * sizeof(uint32_t));
        wb(w, 1, block_hist_buf.buffer,  hist_size);
        w.update_set(dev, radix_count_desc[0]);
    }
    // radix_count[1]: reads morton_tmp_buf (odd passes)
    {
        radix_count_desc[1] = alloc_set(radix_count_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, morton_tmp_buf.buffer,  max_bodies * sizeof(uint32_t));
        wb(w, 1, block_hist_buf.buffer,  hist_size);
        w.update_set(dev, radix_count_desc[1]);
    }

    // radix_scan: block_hist in-place
    {
        radix_scan_desc = alloc_set(radix_scan_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, block_hist_buf.buffer, hist_size);
        w.update_set(dev, radix_scan_desc);
    }

    // radix_scatter[0]: real → tmp  (even passes)
    {
        radix_scatter_desc[0] = alloc_set(radix_scatter_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, morton_buf.buffer,      max_bodies * sizeof(uint32_t));
        wb(w, 1, indices_buf.buffer,     max_bodies * sizeof(uint32_t));
        wb(w, 2, morton_tmp_buf.buffer,  max_bodies * sizeof(uint32_t));
        wb(w, 3, indices_tmp_buf.buffer, max_bodies * sizeof(uint32_t));
        wb(w, 4, block_hist_buf.buffer,  hist_size);
        w.update_set(dev, radix_scatter_desc[0]);
    }
    // radix_scatter[1]: tmp → real  (odd passes)
    {
        radix_scatter_desc[1] = alloc_set(radix_scatter_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, morton_tmp_buf.buffer,  max_bodies * sizeof(uint32_t));
        wb(w, 1, indices_tmp_buf.buffer, max_bodies * sizeof(uint32_t));
        wb(w, 2, morton_buf.buffer,      max_bodies * sizeof(uint32_t));
        wb(w, 3, indices_buf.buffer,     max_bodies * sizeof(uint32_t));
        wb(w, 4, block_hist_buf.buffer,  hist_size);
        w.update_set(dev, radix_scatter_desc[1]);
    }

    const uint32_t node_count = 2 * max_bodies - 1;

    // build: sorted_keys, nodes
    {
        build_desc = alloc_set(build_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, morton_buf.buffer,    max_bodies  * sizeof(uint32_t));
        wb(w, 1, bvh_nodes_buf.buffer, node_count  * sizeof(BVHNode));
        w.update_set(dev, build_desc);
    }

    // refit: body_aabbs, sorted_indices, nodes, flags
    {
        refit_desc = alloc_set(refit_pl.desc_layout);
        DescriptorWriter w;
        wb(w, 0, aabbs_buf.buffer,       max_bodies * sizeof(AABB));
        wb(w, 1, indices_buf.buffer,     max_bodies * sizeof(uint32_t));
        wb(w, 2, bvh_nodes_buf.buffer,   node_count * sizeof(BVHNode));
        wb(w, 3, refit_flags_buf.buffer, node_count * sizeof(int32_t));
        w.update_set(dev, refit_desc);
    }
}

// ─── CPU upload helpers ───────────────────────────────────────────────────────

void LBvhPipeline::set_body(uint32_t slot, const RigidBody& body, const CollisionShape& shape)
{
    assert(slot < max_bodies);
    auto* body_ptr  = reinterpret_cast<RigidBody*>(bodies_buf.info.pMappedData);
    auto* shape_ptr = reinterpret_cast<CollisionShape*>(shapes_buf.info.pMappedData);
    auto* aabb_ptr  = reinterpret_cast<AABB*>(aabbs_buf.info.pMappedData);
    body_ptr[slot]  = body;
    shape_ptr[slot] = shape;

    // Compute initial AABB so the first frame's BVH build has valid data
    // (avbd_integrate.comp will keep it updated from frame 1 onward).
    float c = std::cos(body.rotation);
    float s = std::sin(body.rotation);
    glm::vec2 mn( 1e38f,  1e38f);
    glm::vec2 mx(-1e38f, -1e38f);
    for (uint32_t j = 0; j < shape.count; j++) {
        glm::vec2 v = shape.verts[j];
        glm::vec2 w(c * v.x - s * v.y, s * v.x + c * v.y);
        mn = glm::min(mn, w);
        mx = glm::max(mx, w);
    }
    aabb_ptr[slot].min_pt = body.position + mn;
    aabb_ptr[slot].max_pt = body.position + mx;
}

CollisionShape LBvhPipeline::make_shape(const std::vector<glm::vec2>& verts)
{
    CollisionShape s;
    s.count = static_cast<uint32_t>(std::min(verts.size(), (size_t)MAX_SHAPE_VERTS));
    for (uint32_t i = 0; i < s.count; i++) s.verts[i] = verts[i];

    // Precompute outward edge normals in local space.
    // The shape is centred at the origin, so the centroid is ~(0,0).
    // For each edge i→j, the left-perpendicular of the edge direction points either
    // inward or outward; we pick the one with a positive dot product against verts[i],
    // which points away from the origin.
    for (uint32_t i = 0; i < s.count; i++) {
        uint32_t j = (i + 1) % s.count;
        glm::vec2 edge = s.verts[j] - s.verts[i];
        glm::vec2 n    = glm::normalize(glm::vec2(-edge.y, edge.x));
        if (glm::dot(n, s.verts[i]) < 0.0f) n = -n;  // flip if pointing toward origin
        s.normals[i] = n;
    }

    return s;
}

// ─── Build ────────────────────────────────────────────────────────────────────

void LBvhPipeline::build(VkCommandBuffer cmd, uint32_t n)
{
    assert(n >= 2 && n <= max_bodies);
    body_count = n;

    const uint32_t n_blocks   = (n + 255) / 256;
    const uint32_t n_internal = n - 1;
    const uint32_t node_count = 2 * n - 1;

    // Make CPU writes to bodies_buf / shapes_buf visible to the AABB shader.
    VkMemoryBarrier2 host_vis {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    };
    VkDependencyInfo host_dep {
        .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers    = &host_vis,
    };
    vkCmdPipelineBarrier2(cmd, &host_dep);

    // ── 1. Reset refit flags to 0 ─────────────────────────────────────────────
    vkCmdFillBuffer(cmd, refit_flags_buf.buffer, 0,
                    node_count * sizeof(int32_t), 0u);
    transfer_to_compute_barrier(cmd, refit_flags_buf.buffer);

    // ── 2. Compute Morton codes + initialise index array ─────────────────────
    // (AABBs are written by avbd_integrate.comp at end of previous frame,
    //  or by set_body() on first frame — no separate AABB pass needed.)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, morton_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             morton_pl.layout, 0, 1, &morton_desc, 0, nullptr);
    struct MortonPC { uint32_t n; float wx, wy, iwx, iwy; } mpc {
        n,
        0.f, 0.f,
        1.f / PHYSICS_WIDTH,
        1.f / PHYSICS_HEIGHT,
    };
    vkCmdPushConstants(cmd, morton_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), &mpc);
    vkCmdDispatch(cmd, n_blocks, 1, 1);
    buf_barrier(cmd, morton_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    buf_barrier(cmd, indices_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // ── 4. Radix sort (4 passes, 8 bits each) ─────────────────────────────────
    for (uint32_t pass = 0; pass < 4; pass++) {
        const uint32_t src        = pass % 2;
        const uint32_t bit_shift  = pass * 8;
        struct RadixPC { uint32_t n, bit_shift; } rpc { n, bit_shift };

        // Count
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, radix_count_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 radix_count_pl.layout, 0, 1,
                                 &radix_count_desc[src], 0, nullptr);
        vkCmdPushConstants(cmd, radix_count_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(rpc), &rpc);
        vkCmdDispatch(cmd, n_blocks, 1, 1);
        buf_barrier(cmd, block_hist_buf.buffer,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

        // Scan
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, radix_scan_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 radix_scan_pl.layout, 0, 1,
                                 &radix_scan_desc, 0, nullptr);
        vkCmdPushConstants(cmd, radix_scan_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, 4, &n_blocks);
        vkCmdDispatch(cmd, 1, 1, 1);
        buf_barrier(cmd, block_hist_buf.buffer,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // Scatter
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, radix_scatter_pl.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 radix_scatter_pl.layout, 0, 1,
                                 &radix_scatter_desc[src], 0, nullptr);
        vkCmdPushConstants(cmd, radix_scatter_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(rpc), &rpc);
        vkCmdDispatch(cmd, n_blocks, 1, 1);

        // Barrier on whichever buffer was just written (the "destination" of this pass)
        VkBuffer written = (src == 0) ? morton_tmp_buf.buffer : morton_buf.buffer;
        buf_barrier(cmd, written,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        VkBuffer written_idx = (src == 0) ? indices_tmp_buf.buffer : indices_buf.buffer;
        buf_barrier(cmd, written_idx,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }
    // After 4 passes (even count), sorted data is in morton_buf / indices_buf.

    // ── 5. Build BVH tree ─────────────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, build_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             build_pl.layout, 0, 1, &build_desc, 0, nullptr);
    vkCmdPushConstants(cmd, build_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
    vkCmdDispatch(cmd, (n_internal + 255) / 256, 1, 1);
    buf_barrier(cmd, bvh_nodes_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // ── 6. Refit AABBs bottom-up ──────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, refit_pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             refit_pl.layout, 0, 1, &refit_desc, 0, nullptr);
    vkCmdPushConstants(cmd, refit_pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
    vkCmdDispatch(cmd, n_blocks, 1, 1);

    // Refit writes leaf node AABBs and object_ids; broadphase must see them.
    buf_barrier(cmd, bvh_nodes_buf.buffer,
                VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}
