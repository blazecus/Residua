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

    glm::vec2 sA = (bodyA < world->bodies.size())
        ? glm::vec2(world->bodies[bodyA].sprite.width, world->bodies[bodyA].sprite.height)
        : glm::vec2(0.f);
    glm::vec2 sB = (bodyB < world->bodies.size())
        ? glm::vec2(world->bodies[bodyB].sprite.width, world->bodies[bodyB].sprite.height)
        : glm::vec2(0.f);
    glm::vec2 sum = sA + sB;
    torqueArm = glm::dot(sum, sum);
    if (torqueArm < 1.f) torqueArm = 1.f;

    if (bend_stiffness > 0.f) {
        this->stiffness[2] = bend_stiffness;
        this->penalty[2]   = bend_stiffness;
        float tA = (bodyA < world->bodies.size()) ? world->bodies[bodyA].position.z : 0.f;
        float tB = (bodyB < world->bodies.size()) ? world->bodies[bodyB].position.z : 0.f;
        rest_angle = tA - tB;
    }
}

bool DistanceJoint::initialize() {
    if (!world->active[bodyA] || !world->active[bodyB]) return false;

    const RigidBody& ba = world->bodies[bodyA];
    const RigidBody& bb = world->bodies[bodyB];

    glm::vec2 pA = glm::vec2(ba.position) + glm::rotate(rA_local, ba.position.z);
    glm::vec2 pB = glm::vec2(bb.position) + glm::rotate(rB_local, bb.position.z);
    C0[0] = pA.x - pB.x;
    C0[1] = pA.y - pB.y;
    if (bend_stiffness > 0.f)
        C0[2] = (ba.position.z - bb.position.z - rest_angle) * torqueArm;

    return true;
}

void DistanceJoint::computeConstraint(float alpha) {
    const RigidBody& ba = world->bodies[bodyA];
    const RigidBody& bb = world->bodies[bodyB];

    glm::vec2 rAw = glm::rotate(rA_local, ba.position.z);
    glm::vec2 rBw = glm::rotate(rB_local, bb.position.z);

    glm::vec2 pA = glm::vec2(ba.position) + rAw;
    glm::vec2 pB = glm::vec2(bb.position) + rBw;

    float Cn[3];
    Cn[0] = pA.x - pB.x;
    Cn[1] = pA.y - pB.y;
    Cn[2] = (ba.position.z - bb.position.z - rest_angle) * torqueArm;

    for (int i = 0; i < rows(); i++)
        C[i] = std::isinf(stiffness[i]) ? Cn[i] - C0[i] * alpha : Cn[i];
}

void DistanceJoint::computeDerivatives(uint32_t bi) {
    if (bi == bodyA) {
        glm::vec2 rAw = glm::rotate(rA_local, world->bodies[bodyA].position.z);

        J[0] = {  1.f,  0.f, -rAw.y };
        J[1] = {  0.f,  1.f,  rAw.x };
        J[2] = { 0.f, 0.f, torqueArm };
        H[0] = {}; H[0].row[2].z = -rAw.x;
        H[1] = {}; H[1].row[2].z = -rAw.y;
        H[2] = {};

    } else {
        glm::vec2 rBw = glm::rotate(rB_local, world->bodies[bodyB].position.z);

        J[0] = { -1.f,  0.f,  rBw.y };
        J[1] = {  0.f, -1.f, -rBw.x };
        J[2] = { 0.f, 0.f, -torqueArm };
        H[0] = {}; H[0].row[2].z = rBw.x;
        H[1] = {}; H[1].row[2].z = rBw.y;
        H[2] = {};
    }
}

// ─── MouseDrag ────────────────────────────────────────────────────────────────

MouseDrag::MouseDrag(PhysicsWorld* world, uint32_t bodyA,
                     glm::vec2 r_local, glm::vec2 target, float stiffness)
    : Force(world, bodyA, ~0u), r_local(r_local), target(target)
{
    this->stiffness[0] = this->stiffness[1] = stiffness;
    this->penalty[0]   = this->penalty[1]   = stiffness;
}

bool MouseDrag::initialize() {
    return bodyA < world->bodies.size() && world->active[bodyA];
}

void MouseDrag::computeConstraint(float /*alpha*/) {
    const RigidBody& ba = world->bodies[bodyA];
    glm::vec2 rAw = glm::rotate(r_local, ba.position.z);
    glm::vec2 pA  = glm::vec2(ba.position) + rAw;
    C[0] = pA.x - target.x;
    C[1] = pA.y - target.y;
}

void MouseDrag::computeDerivatives(uint32_t bi) {
    if (bi != bodyA) return;
    glm::vec2 rAw = glm::rotate(r_local, world->bodies[bodyA].position.z);
    J[0] = {  1.f, 0.f, -rAw.y };
    J[1] = {  0.f, 1.f,  rAw.x };
    H[0] = {}; H[0].row[2].z = -rAw.x;
    H[1] = {}; H[1].row[2].z = -rAw.y;
}
