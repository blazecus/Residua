#pragma once

#include "vk_types.h"
#include "vk_pipelines.h"

struct ComputeEffect {
    const char* name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    // ComputePushConstants data;
};

void initialize_compute_effects() {

}
