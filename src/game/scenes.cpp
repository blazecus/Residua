#include "scenes.h"
#include <src/physics/joint.h>
#include <src/physics/spring.h>
#include <src/renderer/residua_engine.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <fmt/core.h>

void SceneManager::init(PhysicsEngine* phys, ResiduaEngine* eng,
                        const LoadedBodyImage* ball, const LoadedBodyImage* cs,
                        const LoadedBodyImage* st)
{
    physics = phys; engine = eng;
    ball_image = ball; cshape = cs; star = st;
}

void SceneManager::clear() {
    physics->clear();
    current_scene = -1;
}

void SceneManager::load(int idx) {
    clear();
    current_scene = idx;
    load_file(SCENE_LIST[idx].path);
}

uint32_t SceneManager::spawn_body(const LoadedBodyImage& img, glm::vec2 world_pos) {
    RigidBody rb;
    rb.sprite = img;
    rb.compute_mass_properties();
    rb.generate_shape();
    rb.generate_sdf();
    rb.position = glm::vec3(world_pos, 0.f);
    return physics->add_body(engine, std::move(rb));
}

static LoadedBodyImage make_rect_image(uint32_t w, uint32_t h) {
    LoadedBodyImage img;
    img.width = w; img.height = h;
    img.pixels.assign(w * h, glm::vec4(1.f));
    return img;
}

static RigidBody make_body(const LoadedBodyImage& img, float x, float y,
                            bool is_static, float density = 1.f) {
    RigidBody rb;
    rb.sprite   = img;
    rb.density  = density;
    rb.compute_mass_properties(density);
    rb.generate_shape();
    rb.generate_sdf();
    rb.position = glm::vec3(x, y, 0.f);
    if (is_static)
        rb.mass = rb.inertia = rb.inv_mass = rb.inv_inertia = 0.f;
    return rb;
}

using Fields = std::unordered_map<std::string, std::string>;

static Fields parse_fields(std::istringstream& ss) {
    Fields f;
    std::string token;
    while (ss >> token) {
        auto eq = token.find('=');
        if (eq != std::string::npos)
            f[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return f;
}

static float getf(const Fields& f, const char* key, float def = 0.f) {
    auto it = f.find(key);
    if (it == f.end()) return def;
    try { return std::stof(it->second); } catch (...) { return def; }
}

static uint32_t getu(const Fields& f, const char* key, uint32_t def = 0u) {
    auto it = f.find(key);
    if (it == f.end()) return def;
    try { return (uint32_t)std::stoul(it->second, nullptr, 0); } catch (...) { return def; }
}

static std::string gets(const Fields& f, const char* key, const char* def = "") {
    auto it = f.find(key);
    return it != f.end() ? it->second : def;
}

void SceneManager::load_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fmt::print("SceneManager: could not open '{}'\n", path);
        return;
    }

    std::vector<uint32_t> body_idx;
    std::unordered_map<std::string, LoadedBodyImage> img_cache;

    std::string line;
    while (std::getline(file, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);

        std::istringstream ss(line);
        std::string kw;
        if (!(ss >> kw)) continue;

        if (kw == "rect") {
            auto f = parse_fields(ss);
            std::string type = gets(f, "type", "rb");
            float cx   = getf(f, "cx");
            float cy   = getf(f, "cy");
            float w    = getf(f, "w", 1.f);
            float h    = getf(f, "h", 1.f);
            uint32_t layer  = getu(f, "layer",   1u);
            uint32_t mask   = getu(f, "mask",    0xFFFFFFFFu);
            float density   = getf(f, "density", 1.f);
            float ang_damp  = getf(f, "ang_damp", 1.f);

            auto img = make_rect_image(std::max(1u, (uint32_t)std::round(w)),
                                       std::max(1u, (uint32_t)std::round(h)));
            bool is_static = (type == "sb");
            RigidBody rb = make_body(img, cx, cy, is_static, density);
            rb.collision_layer = layer;
            rb.collision_mask  = mask;
            rb.angular_damping = ang_damp;
            body_idx.push_back(physics->add_body(engine, std::move(rb)));
        }
        else if (kw == "body") {
            auto f = parse_fields(ss);
            std::string type     = gets(f, "type", "rb");
            float x              = getf(f, "x");
            float y              = getf(f, "y");
            std::string img_path = gets(f, "img");
            uint32_t layer  = getu(f, "layer",   1u);
            uint32_t mask   = getu(f, "mask",    0xFFFFFFFFu);
            float density   = getf(f, "density", 1.f);
            float ang_damp  = getf(f, "ang_damp", 1.f);

            if (img_path.empty()) continue;
            auto it = img_cache.find(img_path);
            if (it == img_cache.end())
                it = img_cache.emplace(img_path, load_body_image(img_path.c_str())).first;

            bool is_static = (type == "sb");
            RigidBody rb = make_body(it->second, x, y, is_static, density);
            rb.collision_layer = layer;
            rb.collision_mask  = mask;
            rb.angular_damping = ang_damp;
            body_idx.push_back(physics->add_body(engine, std::move(rb)));
        }
        else if (kw == "wall") {
            auto f = parse_fields(ss);
            float cx = getf(f, "cx");
            float cy = getf(f, "cy");
            float w  = getf(f, "w",  1.f);
            float h  = getf(f, "h",  1.f);
            body_idx.push_back(physics->add_static_rect(engine, {cx, cy}, w, h));
        }
        else if (kw == "spring") {
            auto f = parse_fields(ss);
            int iA = (int)getu(f, "bodyA");
            int iB = (int)getu(f, "bodyB");
            float rAx       = getf(f, "rAx");
            float rAy       = getf(f, "rAy");
            float rBx       = getf(f, "rBx");
            float rBy       = getf(f, "rBy");
            float rest      = getf(f, "rest");
            float stiffness = getf(f, "stiffness", 1000.f);
            if (iA < 0 || iA >= (int)body_idx.size()) continue;
            if (iB < 0 || iB >= (int)body_idx.size()) continue;

            physics->world.add_force(
                std::make_unique<Spring>(&physics->world,
                    body_idx[iA], body_idx[iB],
                    glm::vec2(rAx, rAy), glm::vec2(rBx, rBy),
                    rest, stiffness));
        }
        else if (kw == "grid") {
            // soft body generation
            // TODO: move to separate function 
            auto f = parse_fields(ss);
            int   rows     = (int)getu(f, "rows",    10u);
            int   cols     = (int)getu(f, "cols",    10u);
            float cx       = getf(f, "cx",      240.f);
            float cy       = getf(f, "cy",      135.f);
            float spacing  = getf(f, "spacing",  12.f);
            float bw       = getf(f, "w",         8.f);
            float bh       = getf(f, "h",         8.f);
            uint32_t layer = getu(f, "layer",      4u);
            uint32_t mask  = getu(f, "mask",       1u);
            float density  = getf(f, "density",   0.5f);
            float ang_damp = getf(f, "ang_damp",  0.95f);
            float stiffness= getf(f, "stiffness",3000.f);
            bool static_top= getu(f, "static_top", 0u) != 0u;
            bool shear     = getu(f, "shear",      1u) != 0u;
            float shear_k  = getf(f, "shear_stiffness", stiffness * 0.5f);

            float x0 = cx - (cols - 1) * spacing * 0.5f;
            float y0 = cy - (rows - 1) * spacing * 0.5f;

            uint32_t base = (uint32_t)body_idx.size();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    bool is_static = static_top && (r == 0);
                    auto img = make_rect_image(
                        std::max(1u, (uint32_t)std::round(bw)),
                        std::max(1u, (uint32_t)std::round(bh)));
                    RigidBody rb = make_body(img, x0 + c * spacing, y0 + r * spacing,
                                              is_static, density);
                    rb.collision_layer = layer;
                    rb.collision_mask  = mask;
                    rb.angular_damping = ang_damp;
                    body_idx.push_back(physics->add_body(engine, std::move(rb)));
                }
            }

            auto gi = [&](int r, int c) -> uint32_t {
                return body_idx[base + (uint32_t)(r * cols + c)];
            };

            float rest_s = spacing;
            float rest_d = spacing * 1.4142135f;

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols - 1; c++)
                    physics->world.add_force(std::make_unique<Spring>(
                        &physics->world, gi(r,c), gi(r,c+1),
                        glm::vec2(0.f), glm::vec2(0.f), rest_s, stiffness));

            for (int r = 0; r < rows - 1; r++)
                for (int c = 0; c < cols; c++)
                    physics->world.add_force(std::make_unique<Spring>(
                        &physics->world, gi(r,c), gi(r+1,c),
                        glm::vec2(0.f), glm::vec2(0.f), rest_s, stiffness));

            if (shear) {
                for (int r = 0; r < rows - 1; r++) {
                    for (int c = 0; c < cols - 1; c++) {
                        physics->world.add_force(std::make_unique<Spring>(
                            &physics->world, gi(r,c), gi(r+1,c+1),
                            glm::vec2(0.f), glm::vec2(0.f), rest_d, shear_k));
                        physics->world.add_force(std::make_unique<Spring>(
                            &physics->world, gi(r,c+1), gi(r+1,c),
                            glm::vec2(0.f), glm::vec2(0.f), rest_d, shear_k));
                    }
                }
            }
        }
        else if (kw == "joint") {
            auto f = parse_fields(ss);
            int iA = (int)getu(f, "bodyA");
            int iB = (int)getu(f, "bodyB");
            float rAx  = getf(f, "rAx");
            float rAy  = getf(f, "rAy");
            float rBx  = getf(f, "rBx");
            float rBy  = getf(f, "rBy");
            float bend = getf(f, "bend", 0.f);
            if (iA < 0 || iA >= (int)body_idx.size()) continue;
            if (iB < 0 || iB >= (int)body_idx.size()) continue;

            physics->world.add_force(
                std::make_unique<DistanceJoint>(&physics->world,
                    body_idx[iA], body_idx[iB],
                    glm::vec2(rAx, rAy), glm::vec2(rBx, rBy),
                    std::numeric_limits<float>::infinity(), bend));
        }
    }
}
