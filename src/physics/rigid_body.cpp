#include "../physics/rigid_body.h"
#include "../physics/shape_gen.h"

AABB RigidBody::generate_AABB() {
    glm::vec2 min = {INF, INF};
    glm::vec2 max = {-INF, -INF};
    for (glm::vec2 vert : shape) {
        min = glm::min(min, vert);
        min = glm::max(max, vert);
    }

    return {min, max};
}

void RigidBody::generate_shape(){
    shape = generate_shape(sprite.width, sprite.height, sprite.pixels);
}