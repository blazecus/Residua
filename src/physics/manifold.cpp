#include "manifold.h"
#include "physics_world.h"
#include "collision.h"
#include <glm/gtx/rotate_vector.hpp>
#include <cmath>
#include <limits>

Manifold::Manifold(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB)
    : Force(world, bodyA, bodyB)
{
    for (int i = 0; i < MANIFOLD_MAX_CONTACTS; i++) {
        fmax[i * 2 + 0] = 0.f;
        fmin[i * 2 + 0] = -std::numeric_limits<float>::infinity();
    }
}

bool Manifold::initialize() {
    const RigidBody2& ba = world->bodies[bodyA];
    const RigidBody2& bb = world->bodies[bodyB];

    friction = std::sqrt(ba.friction * bb.friction);

    Contact oldContacts[MANIFOLD_MAX_CONTACTS];
    float   oldPenalty[MANIFOLD_MAX_CONTACTS * 2];
    float   oldLambda [MANIFOLD_MAX_CONTACTS * 2];
    bool    oldStick  [MANIFOLD_MAX_CONTACTS];
    int     oldNum = numContacts;

    for (int i = 0; i < oldNum; i++) {
        oldContacts[i]         = contacts[i];
        oldPenalty [i * 2 + 0] = penalty[i * 2 + 0];
        oldPenalty [i * 2 + 1] = penalty[i * 2 + 1];
        oldLambda  [i * 2 + 0] = lambda [i * 2 + 0];
        oldLambda  [i * 2 + 1] = lambda [i * 2 + 1];
        oldStick   [i]         = contacts[i].stick;
    }

    // Use GPU-precomputed contacts if available, else fall back to CPU.
    std::vector<ManifoldPoint> pts;
    uint64_t key = ((uint64_t)bodyA << 32) | (uint64_t)bodyB;
    auto it = world->precomputed_contacts.find(key);
    if (it != world->precomputed_contacts.end()) {
        pts = it->second;
    }
    else {
        //pts = build_manifold(ba, bb);
        /*
        if (!pts.empty())
            printf("cpu fallback: %u contacts for pair (%u,%u)\n",
                   (uint32_t)pts.size(), bodyA, bodyB);
                   */
    }
    pts = reduce_manifold(std::move(pts), MANIFOLD_MAX_CONTACTS);
    numContacts = (int)pts.size();

    if (numContacts == 0) {
        for (int j = 0; j < oldNum; j++) {
            retained_penalty = std::max(retained_penalty, oldPenalty[j * 2 + 0]);
            retained_penalty = std::max(retained_penalty, oldPenalty[j * 2 + 1]);
        }
        retained_penalty = std::max(AVBD_PENALTY_MIN, retained_penalty * AVBD_GAMMA);
        return true;
    }

    // Floor retained_penalty at the inertial stiffness of the heavier body so the
    // constraint is immediately strong enough to resist it on the first contact frame.
    // TODO: remove?
    {
        float mass_a = ba.inv_mass > 1e-10f ? 1.f / ba.inv_mass : 0.f;
        float mass_b = bb.inv_mass > 1e-10f ? 1.f / bb.inv_mass : 0.f;
        float ref_mass = std::max(mass_a, mass_b);
        float dt = world->last_dt;
        if (ref_mass > 0.f && dt > 1e-10f)
            retained_penalty = std::max(retained_penalty, ref_mass / (dt * dt));
    }

    for (int i = 0; i < numContacts; i++) {
        penalty[i * 2 + 0] = retained_penalty;
        penalty[i * 2 + 1] = retained_penalty;
        lambda [i * 2 + 0] = 0.f;
        lambda [i * 2 + 1] = 0.f;
        contacts[i].stick = false;
    }

    for (int i = 0; i < numContacts; i++) {
        const ManifoldPoint& mp = pts[i];

        contacts[i].rA     = glm::rotate(mp.position - glm::vec2(ba.position), -ba.position.z);
        contacts[i].rB     = glm::rotate(mp.position - glm::vec2(bb.position), -bb.position.z);
        contacts[i].normal = mp.normal;

        float best = std::numeric_limits<float>::max();
        int   bestJ = -1;
        for (int j = 0; j < oldNum; j++) {
            glm::vec2 d = contacts[i].rA - oldContacts[j].rA;
            float dist = glm::dot(d, d);
            if (dist < best) { best = dist; bestJ = j; }
        }

        if (bestJ >= 0) {
            penalty[i * 2 + 0] = oldPenalty[bestJ * 2 + 0];
            penalty[i * 2 + 1] = oldPenalty[bestJ * 2 + 1];
            lambda [i * 2 + 0] = oldLambda [bestJ * 2 + 0];
            lambda [i * 2 + 1] = oldLambda [bestJ * 2 + 1];
            contacts[i].stick  = oldStick  [bestJ];

            if (oldStick[bestJ]) {
                contacts[i].rA = oldContacts[bestJ].rA;
                contacts[i].rB = oldContacts[bestJ].rB;
            }
        }
    }

    for (int i = 0; i < numContacts; i++) {
        glm::vec2 n = contacts[i].normal;
        glm::vec2 t = { n.y, -n.x };  // tangent

        glm::vec2 rAW = glm::rotate(contacts[i].rA, ba.position.z);
        glm::vec2 rBW = glm::rotate(contacts[i].rB, bb.position.z);

        contacts[i].JAn = { -n.x, -n.y, -(rAW.x * n.y - rAW.y * n.x) };
        contacts[i].JBn = {  n.x,  n.y,   rBW.x * n.y - rBW.y * n.x  };
        contacts[i].JAt = { -t.x, -t.y, -(rAW.x * t.y - rAW.y * t.x) };
        contacts[i].JBt = {  t.x,  t.y,   rBW.x * t.y - rBW.y * t.x  };

        glm::vec2 sep = glm::vec2(ba.position) + rAW - glm::vec2(bb.position) - rBW;
        contacts[i].C0 = { -pts[i].depth + COLLISION_MARGIN, glm::dot(sep, t) };
    }

    return numContacts > 0;
}

void Manifold::computeConstraint(float alpha) {
    const RigidBody2& ba = world->bodies[bodyA];
    const RigidBody2& bb = world->bodies[bodyB];

    for (int i = 0; i < numContacts; i++) {
        glm::vec3 dpA = ba.position - ba.initial;
        glm::vec3 dpB = bb.position - bb.initial;

        C[i * 2 + 0] = contacts[i].C0.x * (1.f - alpha)
                      + glm::dot(contacts[i].JAn, dpA)
                      + glm::dot(contacts[i].JBn, dpB);

        C[i * 2 + 1] = contacts[i].C0.y * (1.f - alpha)
                      + glm::dot(contacts[i].JAt, dpA)
                      + glm::dot(contacts[i].JBt, dpB);

        float frictionBound = std::abs(lambda[i * 2 + 0]) * friction;
        fmax[i * 2 + 1] =  frictionBound;
        fmin[i * 2 + 1] = -frictionBound;

        contacts[i].stick = std::abs(lambda[i * 2 + 1]) < frictionBound
                         && std::abs(contacts[i].C0.y)  < STICK_THRESH;
    }
}

void Manifold::computeDerivatives(uint32_t body_idx) {
    for (int i = 0; i < numContacts; i++) {
        if (body_idx == bodyA) {
            J[i * 2 + 0] = contacts[i].JAn;
            J[i * 2 + 1] = contacts[i].JAt;
        } else {
            J[i * 2 + 0] = contacts[i].JBn;
            J[i * 2 + 1] = contacts[i].JBt;
        }
    }
}
