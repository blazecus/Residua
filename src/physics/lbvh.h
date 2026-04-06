#pragma once

#include <glm/glm.hpp>
#include <limits>
#include <utility>
#include <vector>

struct AABB {
    glm::vec2 min{ std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max() };
    glm::vec2 max{-std::numeric_limits<float>::max(),
                  -std::numeric_limits<float>::max() };

    bool      valid()                 const;
    bool      overlaps(const AABB& o) const;
    glm::vec2 centroid()              const;
    AABB      merge(const AABB& o)    const;
    void      expand(glm::vec2 p);
};

class LBVH {
public:
    void build(const std::vector<AABB>& aabbs);

    void query(const AABB& box, std::vector<int>& out) const;

    void query_pairs(std::vector<std::pair<int, int>>& out) const;

    bool empty() const;

private:
    struct Node {
        AABB aabb;
        int  left  = -1;
        int  right = -1;

        bool is_leaf() const { return right == -1; }
    };

    std::vector<Node> _nodes;
    int               _root = -1;

    int  build_node(const std::vector<AABB>& aabbs,
                    std::vector<int>& ids,
                    const std::vector<uint32_t>& codes,
                    int first, int last);

    int  find_split(const std::vector<int>& ids,
                    const std::vector<uint32_t>& codes,
                    int first, int last) const;

    void query_node(int node, const AABB& box,
                    std::vector<int>& out) const;

    void pairs_nodes(int a, int b,
                     std::vector<std::pair<int, int>>& out) const;

    void self_pairs_node(int node,
                         std::vector<std::pair<int, int>>& out) const;
};
