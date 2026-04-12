#pragma once

#include "rigid_body.h"
#include <vector>

struct ManifoldPoint {
    glm::vec2 position;  // world space contact point
    glm::vec2 normal;    // unit vector from A into B, specific to this contact
    float     depth;     // penetration depth (>= 0)
};

std::vector<ManifoldPoint> build_manifold(const RigidBody2& a, const RigidBody2& b);

std::vector<ManifoldPoint> reduce_manifold(std::vector<ManifoldPoint> points, int n);
