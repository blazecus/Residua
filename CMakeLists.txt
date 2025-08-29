# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Waka")

find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")


add_executable (Waka
  src/main.cpp
  src/game/input_manager.h
  src/game/game.cpp
  src/game/game.h
  src/game/client.cpp
  src/game/client.h
  src/renderer/waka_engine.cpp
  src/renderer/waka_engine.h
  src/renderer/vk_types.h
  src/renderer/vk_images.cpp
  src/renderer/vk_images.h
  src/renderer/vk_initializers.cpp
  src/renderer/vk_initializers.h
  src/renderer/compute_effects.h
  src/renderer/vk_pipelines.cpp
  src/renderer/vk_pipelines.h
  src/renderer/vk_descriptors.cpp
  src/renderer/vk_descriptors.h
  src/renderer/vk_pipelines.cpp
  src/renderer/vk_pipelines.h
)

set_property(TARGET Waka PROPERTY CXX_STANDARD 20)
target_compile_definitions(Waka PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_include_directories(Waka PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(Waka PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui)

target_precompile_headers(Waka PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <glm/mat4x4.hpp>  <glm/vec4.hpp> <vulkan/vulkan.h>)

add_custom_command(TARGET Waka POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Waka> $<TARGET_FILE_DIR:Waka>
  COMMAND_EXPAND_LISTS
  )

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  message(STATUS COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )
