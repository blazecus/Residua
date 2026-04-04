#ifndef PHYSICS_TYPES_GLSL
#define PHYSICS_TYPES_GLSL

// 40 bytes (std430)
struct RigidBody {
    vec2  position;          // world-space CoM (pixels)
    float rotation;          // radians
    float total_mass;        // Σ(alpha_i); updated each frame by integrator
    vec2  velocity;          // pixels/s
    float angular_velocity;  // radians/s
    float I_com;             // moment of inertia about CoM
    uint  pixel_index;       // base index into PhysicsPixelBuffer / PixelColorBuffer
    uint  _pad;
};

// 32 bytes (std430)
struct PhysicsPixel {
    vec2  total_force;      // Σ F_i
    float force_moment;     // Σ (r × F) about CoM
    float total_mass;       // Σ mass_i  (alpha)
    float inertia_moment;   // Σ (mass_i × |r_i|²) about CoM
    float collision_count;  // Σ (mass_i × weight_i) — penetration depth proxy for Baumgarte
    vec2  contact_normal;   // Σ (weight_i × n_i) — normalize for Baumgarte direction
};

// 40 bytes (std430)
// Written by pixel_physics.comp — one per colliding pixel per occupied collision slot.
struct ContactEntry {
    uint  other_id;     // body ID of the other body
    float weight;       // mass * (1 + approach_speed) — momentum weighting
    vec2  normal;       // weight × outward_normal
    vec2  contact_pos;  // weight × world_pos  (for contact point r)
    vec2  other_vel;    // weight × other_body_velocity
    float other_inv_m;  // weight / other_mass  (0 for static / infinite mass)
    float _pad;
};

// 40 bytes (std430)
// Written by contact_reduction.comp — one per unique colliding body per rigid body.
struct ContactAggregate {
    uint  other_id;
    float collision_count;  // Σ weight
    vec2  contact_normal;   // Σ (weight × n)
    vec2  contact_centroid; // Σ (weight × world_pos)
    vec2  contact_vel;      // Σ (weight × other_vel)
    float contact_inv_mass; // Σ (weight / other_mass)
    float _pad;
};

#define COLLISION_EMPTY  0u
#define COLLISION_STATIC 0xFFFFFFFFu

#define MAX_CONTACTS 4u

// 32 bytes (std430)
struct CollisionPixel {
    uint  body_id;   // COLLISION_EMPTY | COLLISION_STATIC | (tier_index << 24) | (slot + 1)
    uint  body_id2;  // second occupant (for body-vs-body two-way collision)
    float vel_x;     // world velocity of body_id  at this pixel
    float vel_y;
    float vel_x2;    // world velocity of body_id2 at this pixel
    float vel_y2;
    float mass;      // total_mass of body_id  (0 for static/empty)
    float mass2;     // total_mass of body_id2 (0 for static/empty)
};

#endif
