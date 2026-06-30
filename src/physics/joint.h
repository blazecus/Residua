#pragma once

#include "force.h"
#include <limits>

struct DistanceJoint : Force {
    glm::vec2 rA_local;
    glm::vec2 rB_local;
    float     bend_stiffness       { 0.f };
    float     rest_angle           { 0.f };
    bool      angular_reaction    { true };  
    float     torqueArm      { 1.f };
    glm::vec3 C0             { 0.f };

    DistanceJoint(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
                  glm::vec2 rA_local, glm::vec2 rB_local,
                  float stiffness      = std::numeric_limits<float>::infinity(),
                  float bend_stiffness = 0.f);

    int  rows()       const override { return bend_stiffness > 0.f ? 3 : 2; }
    bool initialize()                        override;
    void computeConstraint(float alpha)      override;
    void computeDerivatives(uint32_t bi)    override;
};

// One-directional positional joint: follower (bodyB) is pulled to match leader (bodyA),
// but no reaction force is applied to the leader.
struct LeaderJoint : Force {
    glm::vec2 rA_local;
    glm::vec2 rB_local;

    LeaderJoint(PhysicsWorld* world, uint32_t leader, uint32_t follower,
                glm::vec2 rA_local, glm::vec2 rB_local,
                float stiffness = std::numeric_limits<float>::infinity());

    int  rows() const override { return 2; }
    bool initialize()                        override;
    void computeConstraint(float alpha)      override;
    void computeDerivatives(uint32_t bi)    override;
};

// Interactive mouse drag — constrains a body's local point to a world-space target.
// bodyB is unused (UINT32_MAX sentinel); update target each frame before stepping.
struct MouseDrag : Force {
    glm::vec2 r_local;
    glm::vec2 target;

    MouseDrag(PhysicsWorld* world, uint32_t bodyA,
              glm::vec2 r_local, glm::vec2 target, float stiffness);

    int  rows() const override { return 2; }
    bool initialize()                    override;
    void computeConstraint(float alpha)  override;
    void computeDerivatives(uint32_t bi) override;
};
