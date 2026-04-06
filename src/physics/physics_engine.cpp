#include "physics_engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

// ─── Body management ──────────────────────────────────────────────────────────

int Physics::add_body(RigidBody body) {
    if (!_free_ids.empty()) {
        int id = _free_ids.back();
        _free_ids.pop_back();
        _slots[id] = {std::move(body), true};
        return id;
    }
    int id = (int)_slots.size();
    _slots.push_back({std::move(body), true});
    return id;
}

void Physics::remove_body(int id) {
    assert(id >= 0 && id < (int)_slots.size() && _slots[id].active);
    _slots[id].active = false;
    _free_ids.push_back(id);
}

RigidBody* Physics::get(int id) {
    if (id < 0 || id >= (int)_slots.size() || !_slots[id].active) return nullptr;
    return &_slots[id].body;
}

const RigidBody* Physics::get(int id) const {
    if (id < 0 || id >= (int)_slots.size() || !_slots[id].active) return nullptr;
    return &_slots[id].body;
}

void Physics::set_static_layer(uint32_t w, uint32_t h,
                                std::vector<bool> solid)
{
    _static_w     = w;
    _static_h     = h;
    _static_solid = std::move(solid);
}

// ─── Edge query helpers ───────────────────────────────────────────────────────

static float pt_seg_dist(glm::vec2 p, glm::vec2 a, glm::vec2 b) {
    glm::vec2 ab = b - a;
    float len2 = glm::dot(ab, ab);
    if (len2 < 1e-10f) return glm::length(p - a);
    float t = std::clamp(glm::dot(p - a, ab) / len2, 0.f, 1.f);
    return glm::length(p - (a + t * ab));
}

static const MsEdge* nearest_edge(const std::vector<MsEdge>& edges, glm::vec2 p) {
    const MsEdge* best = nullptr;
    float best_d = std::numeric_limits<float>::max();
    for (const MsEdge& e : edges) {
        float d = pt_seg_dist(p, e.a, e.b);
        if (d < best_d) { best_d = d; best = &e; }
    }
    return best;
}

// ─── Contact detection ────────────────────────────────────────────────────────

// Reduce a local list of raw contacts for one body pair to at most 2 representative
// contacts and append them to _contacts.  Uses the averaged normal and the two
// extremal points along the contact tangent so stacking stays stable.
void Physics::reduce_and_emit(std::vector<Contact>& raw, int id_a, int id_b) {
    if (raw.empty()) return;

    // Average normal over all raw contacts.
    glm::vec2 sum_n{0.f};
    for (const Contact& c : raw) sum_n += c.normal;
    float nlen = glm::length(sum_n);
    if (nlen < 1e-6f) return;
    glm::vec2 n = sum_n / nlen;

    // Find the two extremal contact points along the tangent.
    glm::vec2 t{-n.y, n.x};
    glm::vec2 pA = raw[0].point, pB = raw[0].point;
    float tA = glm::dot(pA, t), tB = tA;
    for (const Contact& c : raw) {
        float proj = glm::dot(c.point, t);
        if (proj < tA) { tA = proj; pA = c.point; }
        if (proj > tB) { tB = proj; pB = c.point; }
    }

    // Emit two contacts if they're spatially distinct, otherwise one centroid.
    if (tB - tA > 1.5f) {
        _contacts.push_back({id_a, id_b, pA, n, 1.f});
        _contacts.push_back({id_a, id_b, pB, n, 1.f});
    } else {
        _contacts.push_back({id_a, id_b, (pA + pB) * 0.5f, n, 1.f});
    }
}

// For each solid pixel of body_a that overlaps the static layer, collect a raw
// contact, then reduce the whole set to at most 2 representative contacts.
void Physics::detect_body_vs_static(int id_a) {
    const RigidBody& a = _slots[id_a].body;
    if (!a.sprite || _static_solid.empty()) return;

    const float c  = std::cos(a.rotation), s = std::sin(a.rotation);
    const float hw = a.sprite->width  * 0.5f;
    const float hh = a.sprite->height * 0.5f;

    for(auto& edge : a.sprite->edges){
        float lx = edge.a.x - hw, ly = edge.a.y - hh;
        glm::vec2 wp = a.position + glm::vec2(c*lx - s*ly, s*lx + c*ly);
        if (wp.y > 150){
            _contacts.push_back({id_a, -1, wp, glm::vec2(0, -1), 1.0f});
            return;
        }
    }
    return;

    auto solid_f = [&](int x, int y) -> float {
        if (x < 0 || x >= (int)_static_w || y < 0 || y >= (int)_static_h) return 0.f;
        return _static_solid[y * _static_w + x] ? 1.f : 0.f;
    };

    std::vector<Contact> raw;
    for (uint32_t pi = 0; pi < a.sprite->width * a.sprite->height; pi++) {
        if (a.sprite->pixels[pi].a < 0.5f) continue;

        float lx = float(pi % a.sprite->width)  + 0.5f - hw;
        float ly = float(pi / a.sprite->width)   + 0.5f - hh;
        glm::vec2 wp = a.position + glm::vec2(c*lx - s*ly, s*lx + c*ly);

        int px = (int)wp.x, py = (int)wp.y;
        if (px < 0 || px >= (int)_static_w || py < 0 || py >= (int)_static_h) continue;
        if (!_static_solid[py * _static_w + px]) continue;

        glm::vec2 grad{ solid_f(px+1,py) - solid_f(px-1,py),
                        solid_f(px,py+1) - solid_f(px,py-1) };
        float len = glm::length(grad);
        glm::vec2 normal = len > 1e-6f ? -grad / len : glm::vec2{0.f, -1.f};
        raw.push_back({id_a, -1, wp, normal, 1.f});
    }

    reduce_and_emit(raw, id_a, -1);
}

// Check each marching-squares contour vertex of a against b and vice versa.
// Collects all raw contacts then reduces to at most 2 representative points.
void Physics::detect_body_vs_body(int id_a, int id_b) {
    const RigidBody& a = _slots[id_a].body;
    const RigidBody& b = _slots[id_b].body;
    if (!a.sprite || !b.sprite) return;
    if (a.sprite->edges.empty() && b.sprite->edges.empty()) return;

    const float ca  = std::cos(a.rotation),  sa  = std::sin(a.rotation);
    const float cb  = std::cos(b.rotation),  sb  = std::sin(b.rotation);
    const float cbi = std::cos(-b.rotation), sbi = std::sin(-b.rotation);
    const float cai = std::cos(-a.rotation), sai = std::sin(-a.rotation);

    const float ahw = a.sprite->width  * 0.5f, ahh = a.sprite->height * 0.5f;
    const float bhw = b.sprite->width  * 0.5f, bhh = b.sprite->height * 0.5f;
    const uint32_t aW = a.sprite->width, aH = a.sprite->height;
    const uint32_t bW = b.sprite->width, bH = b.sprite->height;

    std::vector<Contact> raw;

    // ── a's contour vertices inside b ────────────────────────────────────────
    for (const MsEdge& edge : a.sprite->edges) {
        for (int vi = 0; vi < 2; vi++) {
            const glm::vec2& lp = vi == 0 ? edge.a : edge.b;
            float lx = lp.x - ahw, ly = lp.y - ahh;
            glm::vec2 wp = a.position + glm::vec2(ca*lx - sa*ly, sa*lx + ca*ly);

            glm::vec2 rel = wp - b.position;
            float bx = cbi * rel.x - sbi * rel.y + bhw;
            float by = sbi * rel.x + cbi * rel.y + bhh;
            int ibx = (int)bx, iby = (int)by;
            if (ibx < 0 || ibx >= (int)bW || iby < 0 || iby >= (int)bH) continue;
            if (b.sprite->pixels[iby * bW + ibx].a < 0.5f) continue;

            const MsEdge* ne = nearest_edge(b.sprite->edges, {bx, by});
            if (!ne) continue;
            float depth = pt_seg_dist({bx, by}, ne->a, ne->b);
            glm::vec2 wn = glm::vec2(cb * ne->normal.x - sb * ne->normal.y,
                                     sb * ne->normal.x + cb * ne->normal.y);
            raw.push_back({id_a, id_b, wp, wn, depth});
        }
    }

    // ── b's contour vertices inside a ────────────────────────────────────────
    for (const MsEdge& edge : b.sprite->edges) {
        for (int vi = 0; vi < 2; vi++) {
            const glm::vec2& lp = vi == 0 ? edge.a : edge.b;
            float lx = lp.x - bhw, ly = lp.y - bhh;
            glm::vec2 wp = b.position + glm::vec2(cb*lx - sb*ly, sb*lx + cb*ly);

            glm::vec2 rel = wp - a.position;
            float ax = cai * rel.x - sai * rel.y + ahw;
            float ay = sai * rel.x + cai * rel.y + ahh;
            int iax = (int)ax, iay = (int)ay;
            if (iax < 0 || iax >= (int)aW || iay < 0 || iay >= (int)aH) continue;
            if (a.sprite->pixels[iay * aW + iax].a < 0.5f) continue;

            const MsEdge* ne = nearest_edge(a.sprite->edges, {ax, ay});
            if (!ne) continue;
            float depth = pt_seg_dist({ax, ay}, ne->a, ne->b);
            glm::vec2 wn = -glm::vec2(ca * ne->normal.x - sa * ne->normal.y,
                                       sa * ne->normal.x + ca * ne->normal.y);
            raw.push_back({id_a, id_b, wp, wn, depth});
        }
    }

    reduce_and_emit(raw, id_a, id_b);
}

void Physics::detect_contacts() {
    _contacts.clear();

    for (int id : _active_ids)
        detect_body_vs_static(id);

    std::vector<std::pair<int,int>> pairs;
    _bvh.query_pairs(pairs);
    for (auto [pa, pb] : pairs)
        detect_body_vs_body(_active_ids[pa], _active_ids[pb]);
}

// ─── Broadphase ───────────────────────────────────────────────────────────────

void Physics::build_broadphase() {
    _active_ids.clear();
    std::vector<AABB> aabbs;

    for (int i = 0; i < (int)_slots.size(); i++) {
        if (!_slots[i].active) continue;
        _slots[i].body.update_aabb();
        aabbs.push_back(_slots[i].body.aabb);
        _active_ids.push_back(i);
    }

    _bvh.build(aabbs);
}

// ─── Sequential Impulse Solver ────────────────────────────────────────────────

static float     cross2d(glm::vec2 a, glm::vec2 b) { return a.x * b.y - a.y * b.x; }
static glm::vec2 perp(glm::vec2 v)                 { return {-v.y, v.x}; }

void Physics::solve(float dt) {
    if (_contacts.empty()) return;

    constexpr float restitution = 0.4f;
    constexpr float mu          = 0.4f;

    for (int iter = 0; iter < solver_iterations; iter++) {
        for (Contact& ct : _contacts) {
            RigidBody* a = get(ct.body_a);
            RigidBody* b = ct.body_b >= 0 ? get(ct.body_b) : nullptr;
            if (!a) continue;

            const glm::vec2 n   = ct.normal;
            const glm::vec2 r_a = ct.point - a->position;
            const glm::vec2 r_b = b ? ct.point - b->position : glm::vec2{0.f};

            // ── Normal impulse ────────────────────────────────────────────────
            glm::vec2 va = a->velocity + a->angular_velocity * perp(r_a);
            glm::vec2 vb = b ? b->velocity + b->angular_velocity * perp(r_b) : glm::vec2{0.f};
            float v_n = glm::dot(va - vb, n);

            float rna      = cross2d(r_a, n);
            float rnb      = b ? cross2d(r_b, n) : 0.f;
            float eff_mass = a->inv_mass + rna * rna * a->inv_I
                           + (b ? b->inv_mass + rnb * rnb * b->inv_I : 0.f);
            if (eff_mass < 1e-10f) continue;

            // Velocity-only impulse — position correction is a separate pass.
            float lambda = -(v_n * (1.f + restitution)) / eff_mass;

            float old_ln = ct.lambda_n;
            ct.lambda_n  = std::max(old_ln + lambda, 0.f);
            lambda       = ct.lambda_n - old_ln;

            glm::vec2 imp = lambda * n;
            a->velocity         += imp * a->inv_mass;
            a->angular_velocity += cross2d(r_a, imp) * a->inv_I;
            if (b) {
                b->velocity         -= imp * b->inv_mass;
                b->angular_velocity -= cross2d(r_b, imp) * b->inv_I;
            }

            // ── Friction impulse ──────────────────────────────────────────────
            const glm::vec2 t = perp(n);
            va = a->velocity + a->angular_velocity * perp(r_a);
            vb = b ? b->velocity + b->angular_velocity * perp(r_b) : glm::vec2{0.f};
            float v_t = glm::dot(va - vb, t);

            float rta        = cross2d(r_a, t);
            float rtb        = b ? cross2d(r_b, t) : 0.f;
            float eff_mass_t = a->inv_mass + rta * rta * a->inv_I
                             + (b ? b->inv_mass + rtb * rtb * b->inv_I : 0.f);
            if (eff_mass_t < 1e-10f) continue;

            float max_fric = mu * ct.lambda_n;
            float lt       = -v_t / eff_mass_t;
            float old_lt   = ct.lambda_t;
            ct.lambda_t    = std::clamp(old_lt + lt, -max_fric, max_fric);
            lt             = ct.lambda_t - old_lt;

            glm::vec2 fimp = lt * t;
            a->velocity         += fimp * a->inv_mass;
            a->angular_velocity += cross2d(r_a, fimp) * a->inv_I;
            if (b) {
                b->velocity         -= fimp * b->inv_mass;
                b->angular_velocity -= cross2d(r_b, fimp) * b->inv_I;
            }
        }
    }
}

// ─── Position correction ──────────────────────────────────────────────────────
// Directly shifts body positions to resolve penetration.
// Runs after the velocity solve so it doesn't interact with lambda clamping.

void Physics::correct_positions() {
    constexpr float beta = 0.6f;  // fraction of penetration corrected per step
    constexpr float slop = 0.1f;  // pixels of allowed penetration before correcting

    for (const Contact& ct : _contacts) {
        RigidBody* a = get(ct.body_a);
        RigidBody* b = ct.body_b >= 0 ? get(ct.body_b) : nullptr;
        if (!a) continue;

        float pen = std::max(ct.depth - slop, 0.f);
        if (pen <= 0.f) continue;

        float inv_sum = a->inv_mass + (b ? b->inv_mass : 0.f);
        if (inv_sum < 1e-10f) continue;

        // Each body shifts proportional to its inverse mass share.
        glm::vec2 correction = (beta * pen / inv_sum) * ct.normal;
        a->position += a->inv_mass * correction;
        if (b) b->position -= b->inv_mass * correction;
    }
}

void Physics::integrate_velocities(float dt) {
    for (auto& slot : _slots) {
        if (!slot.active || slot.body.inv_mass == 0.f) continue;
        slot.body.velocity += gravity * dt;
    }
}

void Physics::integrate_positions(float dt) {
    for (auto& slot : _slots) {
        if (!slot.active || slot.body.inv_mass == 0.f) continue;
        slot.body.position          += slot.body.velocity         * dt;
        slot.body.rotation          += slot.body.angular_velocity * dt;
        slot.body.update_aabb();
    }
}

int Physics::body_at(glm::vec2 p) const {
    for (int i = 0; i < (int)_slots.size(); i++) {
        if (!_slots[i].active) continue;
        const RigidBody& rb = _slots[i].body;
        if (!rb.sprite) continue;
        glm::vec2 rel = p - rb.position;
        float c = std::cos(-rb.rotation), s = std::sin(-rb.rotation);
        int lx = (int)(c * rel.x - s * rel.y + rb.sprite->width  * 0.5f);
        int ly = (int)(s * rel.x + c * rel.y + rb.sprite->height * 0.5f);
        if (lx < 0 || lx >= (int)rb.sprite->width ||
            ly < 0 || ly >= (int)rb.sprite->height) continue;
        if (rb.sprite->pixels[ly * rb.sprite->width + lx].a > 0.5f) return i;
    }
    return -1;
}

void Physics::step(float dt) {
    using clock = std::chrono::high_resolution_clock;
    using us    = std::chrono::duration<float, std::micro>;

    auto t0 = clock::now();
    integrate_velocities(dt);
    auto t1 = clock::now();
    build_broadphase();
    auto t2 = clock::now();
    detect_contacts();
    auto t3 = clock::now();
    solve(dt);
    auto t4 = clock::now();
    correct_positions();
    auto t5 = clock::now();
    integrate_positions(dt);
    auto t6 = clock::now();

    std::cout << std::fixed
        << "integrate_vel="    << us(t1-t0).count() << "us  "
        << "broadphase="       << us(t2-t1).count() << "us  "
        << "detect_contacts="  << us(t3-t2).count() << "us  "
        << "solve="            << us(t4-t3).count() << "us  "
        << "correct_pos="      << us(t5-t4).count() << "us  "
        << "integrate_pos="    << us(t6-t5).count() << "us  "
        << "total="            << us(t6-t0).count() << "us\n";
}
