#include "../physics/physics_engine.h"
#include "../physics/lbvh.h"
#include <string>

struct RigidBody {
    LoadedBodyImage sprite;
    std::vector<glm::vec2> shape; 
    AABB unrotated_AABB;

    glm::vec2 position;
    glm::vec2 velocity;
    float rotation;

    void generate_shape();

    void load_body(std::string& path);

    AABB generate_AABB();
}
