#ifndef PHYSICS_TYPES_GLSL
#define PHYSICS_TYPES_GLSL

// ─── RigidBody ────────────────────────────────────────────────────────────────
// 24 bytes

struct RigidBody {
    vec2  position;          // world space center (pixels)
    float rotation;          // radians
    float _pad;
    vec2  velocity;          // pixels/s
    float angular_velocity;  // radians/s
    uint  pixel_index;       // start of this body's entries in PhysicsPixelBuffer / PixelColorBuffer
};

// ─── PhysicsPixel ─────────────────────────────────────────────────────────────
// 16 bytes

struct PhysicsPixel {
    vec2  total_force;
    float force_moment;    // Σ (pos × F) about world origin — integrator corrects to CoM-relative
    float total_mass;      // Σ mass_i
    float inertia_moment;  // Σ (mass_i × |pos_i|²) about world origin — integrator applies parallel axis theorem
    float _pad;
};

#endif
