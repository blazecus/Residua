#pragma once

#include "force.h"

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
