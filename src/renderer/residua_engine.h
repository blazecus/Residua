#pragma once

#include <deque>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>

#include "vk_types.h"
#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_descriptors.h"
#include "vk_pipelines.h"

#include "../physics/physics_engine.h"
#include "../physics/rigidbody.h"

#include <vk_mem_alloc.h>
#include <chrono>

static constexpr uint32_t PHYSICS_WIDTH  = 480;
static constexpr uint32_t PHYSICS_HEIGHT = 270;

// CPU-driven renderer for rigidbodies.
// Each frame: paints body pixels into a CPU buffer, uploads to a GPU image.
struct BodyRenderer {
    AllocatedImage  output;   // PHYSICS_WIDTH × PHYSICS_HEIGHT RGBA8, blitted to swapchain
    AllocatedBuffer staging;  // persistently-mapped CPU buffer, same dimensions

    std::vector<uint32_t> background; // packed RGBA8 static layer, size = W*H (may be empty)

    void init(struct ResiduaEngine* engine);
    void set_background(uint32_t w, uint32_t h, const std::vector<glm::vec4>& pixels);
    void draw(const Physics& world, VkCommandBuffer cmd);
};

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function)
    {
        deletors.push_back(function);
    }

    void flush()
    {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); // call functors
        }

        deletors.clear();
    }
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
    glm::vec2 playerPos;
};

struct ComputeEffect {
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct FrameData {
    VkSemaphore _swapchainSemaphore, _renderSemaphore;
    VkFence _renderFence;

    DescriptorAllocatorGrowable _frameDescriptors;
    DeletionQueue _deletionQueue;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct EngineStats {
    float frametime;
    int triangle_count;
    int drawcall_count;
    float mesh_draw_time;
};

struct TextureID {
    uint32_t Index;
};

struct TextureCache {

    std::vector<VkDescriptorImageInfo> Cache;
    std::unordered_map<std::string, TextureID> NameMap;
    TextureID AddTexture(const VkImageView& image, VkSampler sampler);
};

class ResiduaEngine {
public:
    bool _isInitialized { false };
    int _frameNumber { 0 };

    VkExtent2D _windowExtent { 1700, 900 };
    struct SDL_Window* _window { nullptr };

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    FrameData _frames[FRAME_OVERLAP];

    VkSurfaceKHR _surface;
    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;
    VkExtent2D _swapchainExtent;
    VkExtent2D _drawExtent;
    VkDescriptorPool _descriptorPool;

    DescriptorAllocator globalDescriptorAllocator;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    DeletionQueue _mainDeletionQueue;

    VmaAllocator _allocator; // vma lib allocator

    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    // draw resources
    AllocatedImage _drawImage;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{ 0 };

    TextureCache texCache;

    EngineStats stats;

    // singleton style getter.multiple engines is not supported
    static ResiduaEngine& Get();

    // initializes everything in the engine
    void init(
     VkExtent2D windowExtent,
     struct SDL_Window* window
    );

    // shuts down the engine
    void cleanup();

    // draw loop
    void draw();
    void draw_main(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

    void update_scene();

    FrameData& get_current_frame();
    FrameData& get_last_frame();

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    void destroy_image(const AllocatedImage& img);
    void destroy_buffer(const AllocatedBuffer& buffer);

    bool resize_requested { false };

    Physics*     cpu_physics { nullptr };
    BodyRenderer body_renderer;
    float _dt { 0.016f };
    bool freeze_rendering { false };

    void set_window(SDL_Window* window, VkExtent2D& extent);

    void process_renderer_input(SDL_Event& e);

    void process_player_update(glm::vec2 position);

    void draw_frame();
    //void add_render_object()

private:
    void init_vulkan();

    void init_swapchain();

    void create_swapchain(uint32_t width, uint32_t height);

    void resize_swapchain();

    void destroy_swapchain();

    void init_commands();

    void init_descriptors();

    void init_sync_structures();

    void init_imgui();

    void init_default_data();

    void init_pipelines();

    void init_background_pipelines();
};
