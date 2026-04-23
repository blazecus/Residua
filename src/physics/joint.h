#pragma once

#include "force.h"
#include <limits>

struct DistanceJoint : Force {
    glm::vec2 rA_local;
    glm::vec2 rB_local;
    float     bend_stiffness { 0.f };
    float     rest_angle     { 0.f };

    DistanceJoint(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
                  glm::vec2 rA_local, glm::vec2 rB_local,
                  float stiffness      = std::numeric_limits<float>::infinity(),
                  float bend_stiffness = 0.f);

    int  rows()       const override { return bend_stiffness > 0.f ? 3 : 2; }
    bool initialize()                        override;
    void computeConstraint(float alpha)      override;
    void computeDerivatives(uint32_t bi)    override;
};

struct Spring : Force {
    glm::vec2 rA_local;
    glm::vec2 rB_local;
    float     rest_length;

    Spring(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
           glm::vec2 rA_local, glm::vec2 rB_local,
           float rest_length, float stiffness);

    int  rows()       const override { return 1; }
    bool initialize()                        override;
    void computeConstraint(float alpha)      override;
    void computeDerivatives(uint32_t bi)    override;
};
