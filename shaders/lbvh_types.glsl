#ifndef LBVH_TYPES_GLSL
#define LBVH_TYPES_GLSL

// ─── AABB ─────────────────────────────────────────────────────────────────────
// 16 bytes, std430
struct AABB {
    vec2 min_pt;
    vec2 max_pt;
};

// ─── BVH Node ─────────────────────────────────────────────────────────────────
// 32 bytes, std430
// For N bodies: internal nodes at [0, N-2], leaf nodes at [N-1, 2N-2].
// Root is always node 0.
struct BVHNode {
    vec2 aabb_min;   //  8
    vec2 aabb_max;   //  8
    int  left;       //  4  child index; -1 = unset
    int  right;      //  4
    int  parent;     //  4  -1 for root
    int  object_id;  //  4  leaf: original body index; internal: -1
};

// ─── Rigid body (AVBD) ───────────────────────────────────────────────────────
// 48 bytes, std430
struct RigidBody {
    vec2  position;    //  8  world-space center of mass
    vec2  velocity;    //  8  linear velocity (px/s)
    float rotation;    //  4  radians
    float ang_vel;     //  4  angular velocity (rad/s)
    float inv_mass;    //  4  1/mass; 0 = static/infinite mass
    float inv_inertia; //  4  1/I_com; 0 = static
    vec2  prev_pos;    //  8  position at previous timestep (for AVBD)
    float prev_rot;    //  4  rotation at previous timestep (for AVBD)
    float _pad;        //  4
};

// ─── Collision shape ─────────────────────────────────────────────────────────
// 1040 bytes, std430
// Polygon vertices in local space (body CoM = origin, before rotation).
#define MAX_SHAPE_VERTS 128

struct CollisionShape {
    vec2  verts[MAX_SHAPE_VERTS];    // 1024 bytes  local-space polygon vertices
    vec2  normals[MAX_SHAPE_VERTS];  // 1024 bytes  outward edge normal for edge i→(i+1)%count
    uint  count;                     //    4 bytes
    float _pad[3];                   //   12 bytes
};

// ─── Inertia target (AVBD) ───────────────────────────────────────────────────
// 16 bytes, std430
// y = x^t + h*v + h²*a_ext  (unconstrained predicted position for one body)
struct InertiaTarget {
    vec2  pos;   //  8
    float rot;   //  4
    float _pad;  //  4
};

#endif
