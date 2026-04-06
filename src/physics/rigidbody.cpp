#include "rigidbody.h"
#include <stb_image.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

// ─── Marching squares ─────────────────────────────────────────────────────────

// Each row: cell-edge indices to connect into segments.
// Edges: T=0, R=1, B=2, L=3.  -1 = unused.
// Cases 6 and 9 are saddle cases producing two segments.
static const int8_t ms_table[16][4] = {
    {-1,-1,-1,-1}, // 0:  all empty
    { 0, 3,-1,-1}, // 1:  TL          → T-L
    { 0, 1,-1,-1}, // 2:  TR          → T-R
    { 1, 3,-1,-1}, // 3:  TL+TR       → R-L
    { 2, 3,-1,-1}, // 4:  BL          → B-L
    { 0, 2,-1,-1}, // 5:  TL+BL       → T-B
    { 0, 1, 2, 3}, // 6:  TR+BL       → T-R and B-L  (saddle)
    { 1, 2,-1,-1}, // 7:  TL+TR+BL    → R-B
    { 1, 2,-1,-1}, // 8:  BR          → R-B
    { 0, 3, 1, 2}, // 9:  TL+BR       → T-L and R-B  (saddle)
    { 0, 2,-1,-1}, // 10: TR+BR       → T-B
    { 2, 3,-1,-1}, // 11: TL+TR+BR    → B-L
    { 1, 3,-1,-1}, // 12: BL+BR       → R-L
    { 0, 1,-1,-1}, // 13: TL+BL+BR    → T-R
    { 0, 3,-1,-1}, // 14: TR+BL+BR    → T-L
    {-1,-1,-1,-1}, // 15: all solid
};

static float lerp_t(float a0, float a1) {
    float d = a1 - a0;
    if (std::abs(d) < 1e-6f) return 0.5f;
    return std::clamp((0.5f - a0) / d, 0.f, 1.f);
}

static glm::vec2 edge_pt(int edge, int cx, int cy, const float c[4]) {
    switch (edge) {
        case 0: return { cx + lerp_t(c[0], c[1]), (float)cy       }; // top:   TL→TR
        case 1: return { (float)(cx+1), cy + lerp_t(c[1], c[3])   }; // right: TR→BR
        case 2: return { cx + lerp_t(c[2], c[3]), (float)(cy+1)   }; // bot:   BL→BR
        case 3: return { (float)cx,  cy + lerp_t(c[0], c[2])      }; // left:  TL→BL
        default: return {};
    }
}

// Outward normal: alpha gradient points into solid, so negate it.
static glm::vec2 seg_normal(const std::vector<float>& alpha, uint32_t W, uint32_t H,
                              glm::vec2 mid)
{
    auto a = [&](int x, int y) -> float {
        x = std::clamp(x, 0, (int)W - 1);
        y = std::clamp(y, 0, (int)H - 1);
        return alpha[y * W + x];
    };
    int mx = (int)mid.x, my = (int)mid.y;
    glm::vec2 g{ a(mx+1, my) - a(mx-1, my),
                 a(mx, my+1) - a(mx, my-1) };
    float len = glm::length(g);
    return len > 1e-6f ? -g / len : glm::vec2{0.f, -1.f};
}

std::vector<MsEdge> build_ms_edges(uint32_t W, uint32_t H,
                                    const std::vector<float>& alpha)
{
    std::vector<MsEdge> edges;

    for (int y = 0; y < (int)H - 1; y++) {
        for (int x = 0; x < (int)W - 1; x++) {
            float c[4] = {
                alpha[ y      * W + x    ],   // TL
                alpha[ y      * W + x + 1],   // TR
                alpha[(y + 1) * W + x    ],   // BL
                alpha[(y + 1) * W + x + 1],   // BR
            };
            int cas = (c[0]>0.5f?1:0)|(c[1]>0.5f?2:0)|(c[2]>0.5f?4:0)|(c[3]>0.5f?8:0);
            const int8_t* row = ms_table[cas];
            if (row[0] < 0) continue;

            glm::vec2 pa = edge_pt(row[0], x, y, c);
            glm::vec2 pb = edge_pt(row[1], x, y, c);
            edges.push_back({ pa, pb, seg_normal(alpha, W, H, (pa+pb)*0.5f) });

            if (row[2] >= 0) {  // saddle: second segment
                glm::vec2 pc = edge_pt(row[2], x, y, c);
                glm::vec2 pd = edge_pt(row[3], x, y, c);
                edges.push_back({ pc, pd, seg_normal(alpha, W, H, (pc+pd)*0.5f) });
            }
        }
    }

    return edges;
}

// ─── BodySprite ───────────────────────────────────────────────────────────────

BodySprite load_body_sprite(const char* path) {
    int w, h, channels;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    assert(data && "load_body_sprite: failed to open file");

    BodySprite sprite;
    sprite.width  = (uint32_t)w;
    sprite.height = (uint32_t)h;

    const uint32_t n = sprite.width * sprite.height;
    sprite.pixels.resize(n);
    std::vector<float> alpha(n);

    for (uint32_t i = 0; i < n; i++) {
        sprite.pixels[i] = glm::vec4(
            data[i*4+0] / 255.f, data[i*4+1] / 255.f,
            data[i*4+2] / 255.f, data[i*4+3] / 255.f);
        alpha[i] = sprite.pixels[i].a;
    }
    stbi_image_free(data);

    sprite.edges = build_ms_edges(sprite.width, sprite.height, alpha);
    return sprite;
}

// ─── RigidBody ────────────────────────────────────────────────────────────────

void RigidBody::update_aabb() {
    if (!sprite) return;

    const float hw = sprite->width  * 0.5f;
    const float hh = sprite->height * 0.5f;
    const float c  = std::cos(rotation);
    const float s  = std::sin(rotation);

    glm::vec2 corners[4] = {
        { c*-hw - s*-hh,  s*-hw + c*-hh },
        { c* hw - s*-hh,  s* hw + c*-hh },
        { c*-hw - s* hh,  s*-hw + c* hh },
        { c* hw - s* hh,  s* hw + c* hh },
    };

    aabb = {};
    for (const glm::vec2& corner : corners)
        aabb.expand(position + corner);
}

RigidBody make_rigidbody(const BodySprite* sprite,
                          glm::vec2         position,
                          float             rotation)
{
    assert(sprite);

    float mass = 0.f, I_com = 0.f;
    for (uint32_t i = 0; i < sprite->width * sprite->height; i++) {
        const float m  = sprite->pixels[i].a;
        const float lx = float(i % sprite->width)  - sprite->width  * 0.5f + 0.5f;
        const float ly = float(i / sprite->width)   - sprite->height * 0.5f + 0.5f;
        mass  += m;
        I_com += m * (lx*lx + ly*ly);
    }

    RigidBody body;
    body.sprite   = sprite;
    body.position = position;
    body.rotation = rotation;
    body.mass     = mass;
    body.inv_mass = mass  > 0.f ? 1.f / mass  : 0.f;
    body.I_com    = I_com;
    body.inv_I    = I_com > 0.f ? 1.f / I_com : 0.f;
    body.update_aabb();
    return body;
}
