#ifndef PHYSICS_TYPES_GLSL
#define PHYSICS_TYPES_GLSL

// 40 bytes (std430)
struct RigidBody {
    vec2  position;          // world-space CoM (pixels)
    float rotation;          // radians
    float total_mass;        // Σ(alpha_i); updated each frame by integrator
    vec2  velocity;          // pixels/s
    float angular_velocity;  // radians/s
    float I_com;             // moment of inertia about CoM; updated each frame by integrator
    uint  pixel_index;       // base index into PhysicsPixelBuffer / PixelColorBuffer
    uint  _pad;
};

// 40 bytes (std430)
struct PhysicsPixel {
    vec2  total_force;        // Σ F_i
    float force_moment;       // Σ (pos × F) about world origin
    float total_mass;         // Σ mass_i  (alpha)
    float inertia_moment;     // Σ (mass_i × |pos_i|²) about world origin
    float collision_count;    // Σ mass_i of pixels in collision (0 = no contact)
    vec2  contact_normal;     // Σ (mass_i × outward_normal_i)  — normalize to get avg
    vec2  contact_centroid;   // Σ (mass_i × world_pos_i) of contact pixels — divide by collision_count for avg
};


#define COLLISION_EMPTY  0u
#define COLLISION_STATIC 0xFFFFFFFFu

// 12 bytes (std430)
struct CollisionPixel {
    uint  body_id;   // COLLISION_EMPTY | COLLISION_STATIC | (body_idx + 1)
    float vel_x;     
    float vel_y;
};

#endif
