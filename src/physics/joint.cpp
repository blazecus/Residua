#include "joint.h"
#include "physics_world.h"
#include <glm/gtx/rotate_vector.hpp>
#include <limits>

DistanceJoint::DistanceJoint(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
                             glm::vec2 rA_local, glm::vec2 rB_local,
                             float stiffness, float bend_stiffness)
    : Force(world, bodyA, bodyB), rA_local(rA_local), rB_local(rB_local),
      bend_stiffness(bend_stiffness)
{
    float pos_start = std::isinf(stiffness) ? AVBD_PENALTY_MAX * 0.01f : stiffness;
    for (int i = 0; i < 2; i++) {
        this->stiffness[i] = stiffness;
        this->penalty[i]   = pos_start;
    }

    if (bend_stiffness > 0.f) {
        this->stiffness[2] = bend_stiffness;
        this->penalty[2]   = bend_stiffness;
        float tA = (bodyA < world->bodies.size()) ? world->bodies[bodyA].position.z : 0.f;
        float tB = (bodyB < world->bodies.size()) ? world->bodies[bodyB].position.z : 0.f;
        rest_angle = tA - tB;
    }
}

bool DistanceJoint::initialize() {
    return world->active[bodyA] && world->active[bodyB];
}

void DistanceJoint::computeConstraint(float /*alpha*/) {
    const RigidBody2& ba = world->bodies[bodyA];
    const RigidBody2& bb = world->bodies[bodyB];

    glm::vec2 rAw = glm::rotate(rA_local, ba.position.z);
    glm::vec2 rBw = glm::rotate(rB_local, bb.position.z);

    glm::vec2 pA = glm::vec2(ba.position) + rAw;
    glm::vec2 pB = glm::vec2(bb.position) + rBw;

    C[0] = pA.x - pB.x;
    C[1] = pA.y - pB.y;

    if (bend_stiffness > 0.f)
        C[2] = (ba.position.z - bb.position.z) - rest_angle;
}

void DistanceJoint::computeDerivatives(uint32_t bi) {
    if (bi == bodyA) {
        glm::vec2 rAw = glm::rotate(rA_local, world->bodies[bodyA].position.z);

        J[0] = {  1.f,  0.f, -rAw.y };
        J[1] = {  0.f,  1.f,  rAw.x };
        H[0] = {}; H[0].row[2].z = -rAw.x;
        H[1] = {}; H[1].row[2].z = -rAw.y;

        if (bend_stiffness > 0.f) { J[2] = { 0.f, 0.f, 1.f }; H[2] = {}; }
    } else {
        glm::vec2 rBw = glm::rotate(rB_local, world->bodies[bodyB].position.z);

        J[0] = { -1.f,  0.f,  rBw.y };
        J[1] = {  0.f, -1.f, -rBw.x };
        H[0] = {}; H[0].row[2].z = -rBw.x;
        H[1] = {}; H[1].row[2].z = -rBw.y;

        if (bend_stiffness > 0.f) { J[2] = { 0.f, 0.f, -1.f }; H[2] = {}; }
    }
}

Spring::Spring(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
               glm::vec2 rA_local, glm::vec2 rB_local,
               float rest_length, float stiffness)
    : Force(world, bodyA, bodyB), rA_local(rA_local), rB_local(rB_local),
      rest_length(rest_length)
{
    this->stiffness[0] = stiffness;
    this->penalty[0]   = stiffness; // start at full stiffness, no soft warm-up
}

bool Spring::initialize() {
    return world->active[bodyA] && world->active[bodyB];
}

void Spring::computeConstraint(float /*alpha*/) {
    const RigidBody2& ba = world->bodies[bodyA];
    const RigidBody2& bb = world->bodies[bodyB];

    glm::vec2 rAw = glm::rotate(rA_local, ba.position.z);
    glm::vec2 rBw = glm::rotate(rB_local, bb.position.z);

    glm::vec2 diff = glm::vec2(ba.position) + rAw - glm::vec2(bb.position) - rBw;
    float len = glm::length(diff);

    C[0] = (len < 1e-6f) ? -rest_length : len - rest_length;
}

void Spring::computeDerivatives(uint32_t bi) {
    const RigidBody2& ba = world->bodies[bodyA];
    const RigidBody2& bb = world->bodies[bodyB];

    glm::vec2 rAw  = glm::rotate(rA_local, ba.position.z);
    glm::vec2 rBw  = glm::rotate(rB_local, bb.position.z);
    glm::vec2 diff = glm::vec2(ba.position) + rAw - glm::vec2(bb.position) - rBw;
    float len = glm::length(diff);

    H[0] = {};

    if (len < 1e-6f) {
        J[0] = {};
        return;
    }

    glm::vec2 d = diff / len;

    if (bi == bodyA) {
        J[0] = { d.x, d.y, rAw.x * d.y - rAw.y * d.x };
    } else {
        J[0] = { -d.x, -d.y, -(rBw.x * d.y - rBw.y * d.x) };
    }
}
