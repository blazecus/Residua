#ifndef COLLISION_TYPES_GLSL
#define COLLISION_TYPES_GLSL

// ─── Candidate pair (broadphase output) ──────────────────────────────────────
// 8 bytes, std430
struct CandidatePair {
    uint body_a;
    uint body_b;
};

// ─── Contact (narrowphase output, AVBD input) ─────────────────────────────────
// 64 bytes, std430
// Convention: normal points FROM body_b TOWARD body_a (outward from B's surface).
struct Contact {
    uint  body_a;    //  4
    uint  body_b;    //  4
    vec2  normal;    //  8   unit normal pointing from B toward A
    float depth;     //  4   penetration depth (> 0 when penetrating)
    uint  valid;     //  4   1 = active contact this frame
    vec2  r_a;       //  8   world-space offset: contact_point - pos_a
    vec2  r_b;       //  8   world-space offset: contact_point - pos_b
    float lambda_n;  //  4   AVBD normal dual variable (warm-start)
    float lambda_t;  //  4   AVBD tangential dual variable
    float rho_n;     //  4   AVBD normal penalty stiffness
    float rho_t;     //  4   AVBD tangential penalty stiffness
    vec2  _pad;      //  8
};
// 4+4+8+4+4+8+8+4+4+4+4+8 = 64 bytes

#define CONTACT_INITIAL_RHO 1000.0

#endif
