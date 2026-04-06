#pragma once

#include "rigidbody.h"
#include "lbvh.h"
#include <glm/glm.hpp>
#include <vector>

// ─── Contact ──────────────────────────────────────────────────────────────────

struct Contact {
    int       body_a{-1};   // slot index in PhysicsWorld
    int       body_b{-1};   // -1 = static world
    glm::vec2 point;        // world-space contact point
    glm::vec2 normal;       // points from b toward a (outward from static)
    float     depth{0.f};   // penetration depth in pixels (> 0 = overlapping)

    float lambda_n{0.f};    // accumulated normal impulse (warm start)
    float lambda_t{0.f};    // accumulated tangent impulse (warm start)
};

// ─── Physics ──────────────────────────────────────────────────────────────────

class Physics {
public:
    glm::vec2 gravity{0.f, 120.f};
    int       solver_iterations{10};

    int  add_body(RigidBody body);
    void remove_body(int id);

    RigidBody*       get(int id);
    const RigidBody* get(int id) const;

    // Set the static world collision layer.
    // solid[y*w+x] = true means that pixel is solid.
    // edges are the marching squares boundary in world pixel space.
    void set_static_layer(uint32_t w, uint32_t h,
                          std::vector<bool> solid);

    void step(float dt);

    // Returns slot id of the first body containing world_pos, or -1.
    int body_at(glm::vec2 world_pos) const;

    // Valid after each step(); maps LBVH leaf index → slot index.
    const std::vector<int>& active_ids() const { return _active_ids; }

    const std::vector<Contact>& last_contacts() const { return _contacts; }

private:
    struct Slot {
        RigidBody body;
        bool      active{false};
    };

    std::vector<Slot> _slots;
    std::vector<int>  _free_ids;

    LBVH             _bvh;
    std::vector<int> _active_ids;

    uint32_t          _static_w{0}, _static_h{0};
    std::vector<bool> _static_solid;

    std::vector<Contact> _contacts;

    void integrate_velocities(float dt);
    void build_broadphase();
    void detect_contacts();
    void detect_body_vs_static(int id_a);
    void detect_body_vs_body(int id_a, int id_b);
    void reduce_and_emit(std::vector<Contact>& raw, int id_a, int id_b);
    void solve(float dt);
    void correct_positions();
    void integrate_positions(float dt);
};
