#include "lbvh.h"
#include <algorithm>
#include <bit>
#include <numeric>

bool AABB::valid() const {
    return min.x <= max.x && min.y <= max.y;
}

bool AABB::overlaps(const AABB& o) const {
    return min.x <= o.max.x && max.x >= o.min.x &&
           min.y <= o.max.y && max.y >= o.min.y;
}

glm::vec2 AABB::centroid() const {
    return (min + max) * 0.5f;
}

AABB AABB::merge(const AABB& o) const {
    return { glm::min(min, o.min), glm::max(max, o.max) };
}

void AABB::expand(glm::vec2 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
}

static uint32_t spread_bits(uint32_t v) {
    v &= 0x0000FFFFu;
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333u;
    v = (v | (v << 1u)) & 0x55555555u;
    return v;
}

static uint32_t morton2d(float x, float y) {
    uint32_t xi = static_cast<uint32_t>(x * 65535.f);
    uint32_t yi = static_cast<uint32_t>(y * 65535.f);
    return spread_bits(xi) | (spread_bits(yi) << 1u);
}

bool LBVH::empty() const { return _nodes.empty(); }

void LBVH::build(const std::vector<AABB>& aabbs) {
    _nodes.clear();
    _root = -1;

    const int n = static_cast<int>(aabbs.size());
    if (n == 0) return;

    _nodes.reserve(2 * n - 1);

    AABB scene;
    for (const AABB& a : aabbs)
        scene.expand(a.centroid());

    glm::vec2 extent = scene.max - scene.min;
    if (extent.x < 1e-6f) extent.x = 1.f;
    if (extent.y < 1e-6f) extent.y = 1.f;

    std::vector<uint32_t> codes(n);
    for (int i = 0; i < n; i++) {
        glm::vec2 c = (aabbs[i].centroid() - scene.min) / extent;
        c = glm::clamp(c, glm::vec2(0.f), glm::vec2(1.f));
        codes[i] = morton2d(c.x, c.y);
    }

    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(),
              [&](int a, int b) { return codes[a] < codes[b]; });

    _root = build_node(aabbs, ids, codes, 0, n - 1);
}

int LBVH::find_split(const std::vector<int>& ids,
                     const std::vector<uint32_t>& codes,
                     int first, int last) const
{
    uint32_t fc = codes[ids[first]];
    uint32_t lc = codes[ids[last]];

    if (fc == lc)
        return (first + last) / 2;

    int      common = std::countl_zero(fc ^ lc);
    uint32_t mask   = 1u << (31 - common);

    int lo = first, hi = last;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if ((codes[ids[mid]] & mask) == 0u)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

int LBVH::build_node(const std::vector<AABB>& aabbs,
                     std::vector<int>& ids,
                     const std::vector<uint32_t>& codes,
                     int first, int last)
{
    int idx = static_cast<int>(_nodes.size());
    _nodes.push_back({});  

    if (first == last) {
        _nodes[idx].aabb  = aabbs[ids[first]];
        _nodes[idx].left  = ids[first];  
        _nodes[idx].right = -1;          
        return idx;
    }

    int split = find_split(ids, codes, first, last);
    int left  = build_node(aabbs, ids, codes, first,     split);
    int right = build_node(aabbs, ids, codes, split + 1, last );

    _nodes[idx].left  = left;
    _nodes[idx].right = right;
    _nodes[idx].aabb  = _nodes[left].aabb.merge(_nodes[right].aabb);
    return idx;
}

void LBVH::query_node(int node, const AABB& box, std::vector<int>& out) const {
    const Node& n = _nodes[node];
    if (!n.aabb.overlaps(box)) return;
    if (n.is_leaf()) { out.push_back(n.left); return; }
    query_node(n.left,  box, out);
    query_node(n.right, box, out);
}

void LBVH::query(const AABB& box, std::vector<int>& out) const {
    if (_root >= 0) query_node(_root, box, out);
}

void LBVH::pairs_nodes(int a, int b,
                       std::vector<std::pair<int, int>>& out) const
{
    const Node& na = _nodes[a];
    const Node& nb = _nodes[b];

    if (!na.aabb.overlaps(nb.aabb)) return;

    if (na.is_leaf() && nb.is_leaf()) {
        int i = na.left, j = nb.left;
        out.push_back(i < j ? std::pair{i, j} : std::pair{j, i});
        return;
    }

    if (na.is_leaf()) {
        pairs_nodes(a, nb.left,  out);
        pairs_nodes(a, nb.right, out);
    } else {
        pairs_nodes(na.left,  b, out);
        pairs_nodes(na.right, b, out);
    }
}

void LBVH::self_pairs_node(int node,
                            std::vector<std::pair<int, int>>& out) const
{
    const Node& n = _nodes[node];
    if (n.is_leaf()) return;

    self_pairs_node(n.left,  out);
    self_pairs_node(n.right, out);
    pairs_nodes(n.left, n.right, out);
}

void LBVH::query_pairs(std::vector<std::pair<int, int>>& out) const {
    if (_root >= 0) self_pairs_node(_root, out);
}
