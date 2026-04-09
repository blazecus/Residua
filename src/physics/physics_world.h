#include "../physics/lbvh.h"
#include 

struct PhysicsWorld{
    std::vector<RigidBody> bodies;
    std::vector<AABB> boxes;

    std::vector<uint32_t> open_slots;

    void add_body(RigidBody& new_body);

    void remove_body(uint32_t index);

    RigidBody get_body(uint32_t index);

    void step();
}