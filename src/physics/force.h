#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

// ─── Solver constants ─────────────────────────────────────────────────────────

static constexpr int   AVBD_MAX_ROWS    = 4;
static constexpr float AVBD_PENALTY_MIN = 1.f;
static constexpr float AVBD_PENALTY_MAX = 1e9f;
static constexpr float AVBD_ALPHA       = 0.85f;
static constexpr float AVBD_BETA        = 48000.f; 
static constexpr float AVBD_GAMMA       = 0.99f;  

// ─── Row-major 3×3 matrix (from avbd implementation) ─────────────

struct Mat3 {
    glm::vec3 row[3];
    glm::vec3&       operator[](int i)       { return row[i]; }
    const glm::vec3& operator[](int i) const { return row[i]; }
    glm::vec3 col(int i) const { return { row[0][i], row[1][i], row[2][i] }; }
};

inline Mat3 mat3_diag(float a, float b, float c) {
    return {{ {a,0,0}, {0,b,0}, {0,0,c} }};
}
inline Mat3 mat3_outer(glm::vec3 a, glm::vec3 b) {
    return {{ a.x * b, a.y * b, a.z * b }};
}
inline Mat3& operator+=(Mat3& a, Mat3 b) {
    a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; return a;
}
inline Mat3 operator+(Mat3 a, Mat3 b) {
    return {{ a[0]+b[0], a[1]+b[1], a[2]+b[2] }};
}
inline Mat3 operator*(Mat3 a, float b) {
    return {{ a[0]*b, a[1]*b, a[2]*b }};
}
inline Mat3 operator/(Mat3 a, float b) {
    return {{ a[0]/b, a[1]/b, a[2]/b }};
}
inline glm::vec3 operator*(Mat3 a, glm::vec3 b) {
    return { glm::dot(a[0],b), glm::dot(a[1],b), glm::dot(a[2],b) };
}

// LDL^T solve for 3×3 SPD system (from reference).
inline glm::vec3 mat3_solve(Mat3 a, glm::vec3 b) {
    float D1  = a[0][0];
    float L21 = a[1][0] / D1;
    float L31 = a[2][0] / D1;
    float D2  = a[1][1] - L21 * L21 * D1;
    float L32 = (a[2][1] - L21 * L31 * D1) / D2;
    float D3  = a[2][2] - (L31 * L31 * D1 + L32 * L32 * D2);

    float y1 = b.x;
    float y2 = b.y - L21 * y1;
    float y3 = b.z - L31 * y1 - L32 * y2;

    float z1 = y1 / D1, z2 = y2 / D2, z3 = y3 / D3;

    glm::vec3 x;
    x[2] = z3;
    x[1] = z2 - L32 * x[2];
    x[0] = z1 - L21 * x[1] - L31 * x[2];
    return x;
}

struct PhysicsWorld;

struct Force {
    PhysicsWorld* world;
    uint32_t bodyA, bodyB;

    glm::vec3 J[AVBD_MAX_ROWS]{};
    Mat3      H[AVBD_MAX_ROWS]{};
    float     C[AVBD_MAX_ROWS]{};
    float     fmin[AVBD_MAX_ROWS];
    float     fmax[AVBD_MAX_ROWS];
    float     stiffness[AVBD_MAX_ROWS];
    float     fracture[AVBD_MAX_ROWS];
    float     penalty[AVBD_MAX_ROWS]{};
    float     lambda[AVBD_MAX_ROWS]{};

    Force(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB);
    virtual ~Force() = default;

    virtual int  rows()                              const = 0;
    virtual bool initialize()                              = 0;
    virtual void computeConstraint(float alpha)            = 0;
    virtual void computeDerivatives(uint32_t body_idx)    = 0;
    virtual bool is_contact()                        const { return false; }
};
