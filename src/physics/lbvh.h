#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <bit>
#include <functional>

#include <glm/gtx/rotate_vector.hpp>

inline const float INF = std::numeric_limits<float>::infinity();

struct AABB {
    glm::vec2 min, max;
};

struct LBVHNode {
    int left, right;
    AABB box;
};

struct LBVH {
    std::vector<AABB> boxes;
    std::vector<int> sorted;
    std::vector<LBVHNode> nodes;
};

inline glm::vec2 AABB_center(const AABB& box) {
    return (box.min + box.max) * 0.5f;
}

inline AABB AABB_merge(const std::vector<AABB>& boxes) {
    AABB merged = {
        glm::vec2(INF),
        glm::vec2(-INF)
    };

    for (const AABB& box : boxes) {
        merged.min = glm::min(merged.min, box.min);
        merged.max = glm::max(merged.max, box.max);
    }

    return merged;
}

inline AABB AABB_offset_ex(const AABB& box, float offset) {
    AABB new_box = AABB{box.min - offset, box.max + offset};
    glm::vec2 center = AABB_center(new_box);
    return AABB_merge({new_box, {center, center}});
}

inline AABB AABB_offset(const AABB& box, float offset){
    return {box.min - offset, box.max + offset};
}

inline bool AABB_intersects(const AABB& a, const AABB& b) {
    return a.max.x > b.min.x && b.max.x > a.min.x &&
           a.max.y > b.min.y && b.max.y > a.min.y;
}

inline bool AABB_contains_point(const AABB& box, glm::vec2 p){
    return box.min.x <= p.x && p.x <= box.max.x && box.min.y <= p.y && p.y <= box.max.y;
}

inline AABB AABB_from_rotated_rect(glm::vec2 position, glm::vec2 size) {
    glm::vec2 c0 = position + glm::vec2(-0.5f * size.x, -0.5f * size.y);
    glm::vec2 c1 = position + glm::vec2( 0.5f * size.x, -0.5f * size.y);
    glm::vec2 c2 = position + glm::vec2(-0.5f * size.x,  0.5f * size.y);
    glm::vec2 c3 = position + glm::vec2( 0.5f * size.x,  0.5f * size.y);

    return {
        glm::min(glm::min(c0, c1), glm::min(c2, c3)),
        glm::max(glm::max(c0, c1), glm::max(c2, c3))
    };
}

inline AABB AABB_rotate(const AABB& box, float angle_radians) {
    glm::vec2 center = 0.5f * (box.min + box.max);
    glm::vec2 half_extents = 0.5f * (box.max - box.min);

    std::array<glm::vec2, 4> corners = {
        glm::vec2(-half_extents.x, -half_extents.y),
        glm::vec2( half_extents.x, -half_extents.y),
        glm::vec2(-half_extents.x,  half_extents.y),
        glm::vec2( half_extents.x,  half_extents.y)
    };

    glm::vec2 new_min( 1e20f,  1e20f);
    glm::vec2 new_max(-1e20f, -1e20f);

    for (auto& c : corners) {
        glm::vec2 rotated = glm::rotate(c, angle_radians) + center;
        new_min = glm::min(new_min, rotated);
        new_max = glm::max(new_max, rotated);
    }

    return {new_min, new_max};
}

inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint32_t morton_code(glm::vec2 p, const AABB& extent) {
    glm::vec2 q = 1024.0f * (p - extent.min) / (extent.max - extent.min);

    uint32_t x = std::clamp(uint32_t(q.x), 0u, 1023u);
    uint32_t y = std::clamp(uint32_t(q.y), 0u, 1023u);

    return (expand_bits(y) << 1) | expand_bits(x);
}

inline void lbvh_build(LBVH& lbvh, std::vector<AABB>& boxes) {
    size_t boxes_size = boxes.size();

    lbvh.sorted.resize(boxes_size);
    lbvh.nodes.resize(boxes_size);
    lbvh.boxes = boxes;

    AABB extent = AABB_merge(boxes);

    lbvh.nodes.clear();

    std::vector<uint32_t> codes(boxes_size);
    std::vector<uint32_t> sorted_codes(boxes_size);

    for(uint32_t i = 0; i < boxes_size; i++) {
        codes[i] = morton_code(AABB_center(boxes[i]), extent);
        lbvh.sorted[i] = i;
    }

    std::sort(lbvh.sorted.begin(), lbvh.sorted.end(),
        [&codes](int a, int b){ return codes[a] < codes[b]; });

    for (size_t j = 0; j < boxes_size; j++) {
        int i = lbvh.sorted[j];
        sorted_codes[i] = codes[j];
    }

    std::function<std::pair<int,AABB>(int,int)> build_recurse;

    build_recurse = [&](int first, int last) -> std::pair<int,AABB> {
        if (first == last) {
            return {-first-1, lbvh.boxes[lbvh.sorted[first]]};
        }

        // find split
        int split = first;
        if (codes[lbvh.sorted[first]] == codes[lbvh.sorted[last]]) {
            split = (first + last) / 2;
        } else {
            uint32_t common_prefix = std::countl_zero(codes[lbvh.sorted[first]] ^ codes[lbvh.sorted[last]]);
            int step = last - first;
            while (step > 1) {
                step = (step + 1) / 2;
                if (split + step < last) {
                    uint32_t split_prefix = std::countl_zero(codes[lbvh.sorted[first]] ^ codes[lbvh.sorted[split]]);
                    if (split_prefix > common_prefix) {
                        split += step;
                    }
                }
            }
        }

        int node_index = (int)lbvh.nodes.size();
        lbvh.nodes.push_back(LBVHNode{}); // empty placeholder

        auto[left_index, box_left] = build_recurse(first, split);
        auto[right_index, box_right] = build_recurse(split+1, last);

        std::vector<AABB> lr_boxes{ box_left, box_right };
        lbvh.nodes[node_index] = { left_index, right_index, AABB_merge(lr_boxes) };
        return {node_index, lbvh.nodes[node_index].box};
    };

    if (!boxes.empty()) {
        build_recurse(0, int(boxes_size)-1);
    }
}


inline void lbvh_query_point(const LBVH& lbvh, const glm::vec2& p, float rq, std::vector<int>& results) {
    results.clear();

    std::function<void(int)> query_recurse;
    query_recurse = [&](int i) {
        if (i < 0) {
            int j = lbvh.sorted[-i-1];
            if (AABB_contains_point(AABB_offset(lbvh.boxes[j], rq), p)) {
                results.push_back(j);
            }
            return;
        }
        const LBVHNode& node = lbvh.nodes[i];
        if (AABB_contains_point(AABB_offset(node.box, rq), p)) {
            query_recurse(node.left);
            query_recurse(node.right);
        }
    };

    if (!lbvh.nodes.empty()) query_recurse(0);
}

inline bool lbvh_ray_vs_aabb(const AABB& box, glm::vec2 origin, glm::vec2 dir, float max_dist)
{
    float t_min = 0.f, t_max = max_dist;
    for (int axis = 0; axis < 2; axis++) {
        float d  = (&dir.x)[axis];
        float o  = (&origin.x)[axis];
        float lo = (&box.min.x)[axis];
        float hi = (&box.max.x)[axis];
        if (std::abs(d) < 1e-9f) {
            if (o < lo || o > hi) return false;
        } else {
            float t1 = (lo - o) / d, t2 = (hi - o) / d;
            if (t1 > t2) std::swap(t1, t2);
            t_min = std::max(t_min, t1);
            t_max = std::min(t_max, t2);
            if (t_min > t_max) return false;
        }
    }
    return true;
}

// Returns box-array indices (map through lbvh_index_map to get body indices).
inline void lbvh_query_ray(const LBVH& lbvh, glm::vec2 origin, glm::vec2 dir, float max_dist,
                            std::vector<int>& results)
{
    results.clear();
    if (lbvh.nodes.empty()) {
        // 0 or 1 bodies — no internal nodes, test leaves directly
        for (int j : lbvh.sorted)
            if (lbvh_ray_vs_aabb(lbvh.boxes[j], origin, dir, max_dist))
                results.push_back(j);
        return;
    }

    std::function<void(int)> recurse;
    recurse = [&](int i) {
        if (i < 0) {
            int j = lbvh.sorted[-i - 1];
            if (lbvh_ray_vs_aabb(lbvh.boxes[j], origin, dir, max_dist))
                results.push_back(j);
            return;
        }
        const LBVHNode& node = lbvh.nodes[i];
        if (lbvh_ray_vs_aabb(node.box, origin, dir, max_dist)) {
            recurse(node.left);
            recurse(node.right);
        }
    };
    recurse(0); // root is always at index 0
}

inline void lbvh_query_AABB(const LBVH& lbvh, const AABB& box, std::vector<int>& results) {
    results.clear();

    std::function<void(int)> query_recurse;
    query_recurse = [&](int i) {
        if (i < 0) {
            int j = lbvh.sorted[-i-1];
            if (AABB_intersects(lbvh.boxes[j], box)) {
                results.push_back(j);
            }
            return;
        }
        const LBVHNode& node = lbvh.nodes[i];
        if (AABB_intersects(node.box, box)) {
            query_recurse(node.left);
            query_recurse(node.right);
        }
    };

    if (!lbvh.nodes.empty()) query_recurse(0);
}
