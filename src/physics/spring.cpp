#include "spring.h"
#include "physics_world.h"
#include <glm/gtx/rotate_vector.hpp>

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
    const RigidBody& ba = world->bodies[bodyA];
    const RigidBody& bb = world->bodies[bodyB];

    glm::vec2 rAw = glm::rotate(rA_local, ba.position.z);
    glm::vec2 rBw = glm::rotate(rB_local, bb.position.z);

    glm::vec2 diff = glm::vec2(ba.position) + rAw - glm::vec2(bb.position) - rBw;
    float len = glm::length(diff);

    C[0] = (len < 1e-6f) ? -rest_length : len - rest_length;
}

void Spring::computeDerivatives(uint32_t bi) {
    const RigidBody& ba = world->bodies[bodyA];
    const RigidBody& bb = world->bodies[bodyB];

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
