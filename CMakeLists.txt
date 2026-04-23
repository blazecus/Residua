# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Residua")

find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")


add_executable (Residua
  src/main.cpp
  src/game/input_manager.h
  src/game/game.cpp
  src/game/game.h
  src/game/scenes.cpp
  src/game/scenes.h
  src/game/ui.cpp
  src/game/ui.h
  src/game/client.cpp
  src/game/client.h
  src/renderer/residua_engine.cpp
  src/renderer/residua_engine.h
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
  src/physics/body_image.cpp
  src/physics/body_image.h
  src/physics/lbvh.h
  src/physics/physics_world.cpp
  src/physics/physics_world.h
  src/physics/shape_gen.cpp
  src/physics/shape_gen.h
  src/physics/rigid_body.cpp
  src/physics/rigid_body.h
  src/physics/collision.cpp
  src/physics/collision.h
  src/physics/manifold.cpp
  src/physics/manifold.h
  src/physics/force.h
  src/physics/joint.cpp
  src/physics/joint.h
  src/physics/physics_engine.cpp
  src/physics/physics_engine.h
)

set_property(TARGET Residua PROPERTY CXX_STANDARD 20)
target_compile_definitions(Residua PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_include_directories(Residua PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(Residua PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui)

target_precompile_headers(Residua PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <glm/mat4x4.hpp>  <glm/vec4.hpp> <vulkan/vulkan.h>)

add_custom_command(TARGET Residua POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Residua> $<TARGET_FILE_DIR:Residua>
  COMMAND_EXPAND_LISTS
  )

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
    )

file(GLOB_RECURSE GLSL_HEADER_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.glsl"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  message(STATUS "BUILDING SHADER")
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  message(STATUS ${GLSL})
  message(STATUS COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -I${PROJECT_SOURCE_DIR}/shaders -o ${SPIRV})
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.3 ${GLSL} -I${PROJECT_SOURCE_DIR}/shaders -o ${SPIRV}
    DEPENDS ${GLSL} ${GLSL_HEADER_FILES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders
    DEPENDS ${SPIRV_BINARY_FILES}
    )

add_dependencies(Residua Shaders)
