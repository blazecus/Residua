#pragma once

#include "force.h"

static constexpr int   MANIFOLD_MAX_CONTACTS = 2;
static constexpr float COLLISION_MARGIN      = 0.0005f;
static constexpr float STICK_THRESH          = 0.01f;

struct Manifold : Force {
    struct Contact {
        glm::vec2 rA, rB;          
        glm::vec2 normal;           
        glm::vec3 JAn, JBn;       
        glm::vec3 JAt, JBt;       
        glm::vec2 C0;               
        bool      stick{ false };
    };

    Contact contacts[MANIFOLD_MAX_CONTACTS]{};
    int     numContacts{ 0 };
    float   friction{ 0.f };
    float   retained_penalty{ AVBD_PENALTY_MIN }; 

    Manifold(PhysicsWorld* world, uint32_t bodyA, uint32_t bodyB);

    int  rows()       const override { return numContacts * 2; }
    bool is_contact() const override { return true; }
    bool initialize()                           override;
    void computeConstraint(float alpha)         override;
    void computeDerivatives(uint32_t body_idx) override;
};
