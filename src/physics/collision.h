#pragma once

#include "rigid_body.h"
#include <vector>

struct ManifoldPoint {
    glm::vec2 position; 
    glm::vec2 normal; 
    float     depth; 
};

std::vector<ManifoldPoint> build_manifold(const RigidBody2& a, const RigidBody2& b);

std::vector<ManifoldPoint> reduce_manifold(std::vector<ManifoldPoint> points, int n);
