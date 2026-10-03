#include "joint.h"
#include "physics_world.h"
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <cmath>

DistanceJoint::DistanceJoint(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB,
                             glm::vec2 rA_local, glm::vec2 rB_local,
                             float stiffness, float bend_stiffness, float max_torque,
                             float bend_damping)
    : Force(world, bodyA, bodyB), rA_local(rA_local), rB_local(rB_local),
      bend_stiffness(bend_stiffness), bend_damping(bend_damping), max_torque(max_torque)
{
    float pos_start = std::isinf(stiffness) ? AVBD_PENALTY_MAX * 0.01f : stiffness;
    for (int i = 0; i < 2; i++) {
        this->stiffness[i] = stiffness;
        this->penalty[i]   = pos_start;
    }

    torqueArm = 1.f;

    if (bend_stiffness > 0.f) {
        this->stiffness[2] = bend_stiffness;
        this->penalty[2]   = bend_stiffness;
        this->fmax[2]      =  max_torque;
        this->fmin[2]      = -max_torque;
        float tA = (bodyA < world->bodies.size()) ? world->bodies[bodyA].position.z : 0.f;
        float tB = (bodyB < world->bodies.size()) ? world->bodies[bodyB].position.z : 0.f;
        rest_angle = tA - tB;
    }
}

void DistanceJoint::configureBendMotor(float hz, float dampingRatio, float maxAngularAccel) {
    float invIa = (angular_reaction && bodyA < world->bodies.size())
        ? world->bodies[bodyA].inv_inertia : 0.f;
    float invIb = (bodyB < world->bodies.size()) ? world->bodies[bodyB].inv_inertia : 0.f;
    float invISum = invIa + invIb;
    float effInertia = (invISum > 1e-8f) ? 1.f / invISum : 0.f;

    constexpr float PI = 3.14159265f;
    float omega = 2.f * PI * hz;
    bend_stiffness = effInertia * omega * omega;
    bend_damping   = 2.f * effInertia * dampingRatio * omega;
    max_torque     = effInertia * maxAngularAccel;

    stiffness[2] = bend_stiffness;
    penalty[2]   = bend_stiffness;
    fmax[2]      =  max_torque;
    fmin[2]      = -max_torque;
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

    float rel_angle = ba.position.z - bb.position.z;

    float Cn[3];
    Cn[0] = pA.x - pB.x;
    Cn[1] = pA.y - pB.y;

    if (bend_stiffness > 0.f) {
        float diff = rel_angle - rest_angle;
        diff = std::atan2(std::sin(diff), std::cos(diff));

        float omega_rel = ba.velocity.z - bb.velocity.z;
        float bias = bend_damping * omega_rel / bend_stiffness;
        Cn[2] = (diff + bias) * torqueArm;
    }

    for (int i = 0; i < rows(); i++)
        C[i] = std::isinf(stiffness[i]) ? Cn[i] - C0[i] * alpha : Cn[i];
}

void DistanceJoint::computeDerivatives(uint32_t bi) {
    if (bi == bodyA) {
        glm::vec2 rAw = glm::rotate(rA_local, world->bodies[bodyA].position.z);

        J[0] = {  1.f,  0.f, -rAw.y };
        J[1] = {  0.f,  1.f,  rAw.x };
        H[0] = {}; H[0].row[2].z = -rAw.x;
        H[1] = {}; H[1].row[2].z = -rAw.y;
        J[2] = angular_reaction ? glm::vec3{ 0.f, 0.f, torqueArm } : glm::vec3{ 0.f };
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

// ─── LeaderJoint ─────────────────────────────────────────────────────────────

LeaderJoint::LeaderJoint(PhysicsWorld* world, uint32_t leader, uint32_t follower,
                         glm::vec2 rA_local, glm::vec2 rB_local, float stiffness)
    : Force(world, leader, follower), rA_local(rA_local), rB_local(rB_local)
{
    float pos_start = std::isinf(stiffness) ? AVBD_PENALTY_MAX * 0.01f : stiffness;
    this->stiffness[0] = this->stiffness[1] = stiffness;
    this->penalty[0]   = this->penalty[1]   = pos_start;
}

bool LeaderJoint::initialize() {
    return world->active[bodyA] && world->active[bodyB];
}

void LeaderJoint::computeConstraint(float /*alpha*/) {
    const RigidBody& ba = world->bodies[bodyA];
    const RigidBody& bb = world->bodies[bodyB];

    glm::vec2 pA = glm::vec2(ba.position) + glm::rotate(rA_local, ba.position.z);
    glm::vec2 pB = glm::vec2(bb.position) + glm::rotate(rB_local, bb.position.z);

    C[0] = pB.x - pA.x;
    C[1] = pB.y - pA.y;
}

void LeaderJoint::computeDerivatives(uint32_t bi) {
    if (bi != bodyB) return;
    glm::vec2 rBw = glm::rotate(rB_local, world->bodies[bodyB].position.z);
    J[0] = {  1.f, 0.f, -rBw.y };
    J[1] = {  0.f, 1.f,  rBw.x };
    H[0] = {}; H[0].row[2].z = -rBw.x;
    H[1] = {}; H[1].row[2].z = -rBw.y;
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

AngleAnchor::AngleAnchor(PhysicsWorld* world, uint32_t bodyA)
    : Force(world, bodyA, ~0u)
{
    stiffness[0] = 0.f;
    penalty[0]   = 0.f;
    fmax[0]      =  std::numeric_limits<float>::infinity();
    fmin[0]      = -std::numeric_limits<float>::infinity();
}

bool AngleAnchor::initialize() {
    return bodyA < world->bodies.size() && world->active[bodyA];
}

void AngleAnchor::computeConstraint(float /*alpha*/) {
    const RigidBody& ba = world->bodies[bodyA];
    float diff = ba.position.z - rest_angle;
    diff = std::atan2(std::sin(diff), std::cos(diff));

    float bias = (bend_stiffness > 0.f) ? bend_damping * ba.velocity.z / bend_stiffness : 0.f;
    C[0] = diff + bias;
}

void AngleAnchor::computeDerivatives(uint32_t bi) {
    if (bi != bodyA) return;
    J[0] = { 0.f, 0.f, 1.f };
    H[0] = {};
}

void AngleAnchor::configureBendMotor(float hz, float dampingRatio, float maxAngularAccel) {
    float invIa       = (bodyA < world->bodies.size()) ? world->bodies[bodyA].inv_inertia : 0.f;
    float effInertia  = (invIa > 1e-8f) ? 1.f / invIa : 0.f;

    constexpr float PI = 3.14159265f;
    float omega = 2.f * PI * hz;
    bend_stiffness = effInertia * omega * omega;
    bend_damping   = 2.f * effInertia * dampingRatio * omega;
    max_torque     = effInertia * maxAngularAccel;

    stiffness[0] = bend_stiffness;
    penalty[0]   = bend_stiffness;
    fmax[0]      =  max_torque;
    fmin[0]      = -max_torque;
}
