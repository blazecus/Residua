#pragma once

#include "lbvh.h"
#include <glm/glm.hpp>
#include <vector>

// ─── Marching squares edge ────────────────────────────────────────────────────

struct MsEdge {
    glm::vec2 a, b;      // endpoints in local pixel space (0..width, 0..height)
    glm::vec2 normal;    // outward unit normal in local space
};

// Run marching squares on an alpha field and return boundary edge segments.
// Edges are in pixel-space coordinates of the grid.
std::vector<MsEdge> build_ms_edges(uint32_t W, uint32_t H,
                                    const std::vector<float>& alpha);

// ─── BodySprite ───────────────────────────────────────────────────────────────

struct BodySprite {
    uint32_t               width{}, height{};
    std::vector<glm::vec4> pixels;   // RGBA; alpha = mass density / solid mask
    std::vector<MsEdge>    edges;    // marching squares boundary
};

BodySprite load_body_sprite(const char* path);

// ─── RigidBody ────────────────────────────────────────────────────────────────

struct RigidBody {
    const BodySprite* sprite{nullptr};

    AABB aabb;

    glm::vec2 position{0.f, 0.f};
    glm::vec2 velocity{0.f, 0.f};

    float rotation{0.f};
    float angular_velocity{0.f};

    float mass{0.f};
    float inv_mass{0.f};
    float I_com{0.f};
    float inv_I{0.f};

    void update_aabb();
};

RigidBody make_rigidbody(const BodySprite* sprite,
                          glm::vec2         position,
                          float             rotation = 0.f);
