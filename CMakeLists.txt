cmake_minimum_required (VERSION 3.8)

project ("Residua")

find_package(Vulkan REQUIRED)

add_subdirectory(third_party)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install    OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# ── Sources shared between the exe and the Python extension ──────────────────
set(RESIDUA_COMMON_SOURCES
  src/game/input_manager.h
  src/game/game.cpp
  src/game/game.h
  src/game/scenes.cpp
  src/game/scenes.h
  src/game/ui.cpp
  src/game/ui.h
  src/game/debug_draw.cpp
  src/game/debug_draw.h
  src/game/client.cpp
  src/game/client.h
  src/game/settings.cpp
  src/game/settings.h
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
  src/physics/spring.cpp
  src/physics/spring.h
  src/physics/physics_engine.cpp
  src/physics/physics_engine.h
  src/physics/particle_sim.cpp
  src/physics/particle_sim.h
  src/physics/physics_api.cpp
  src/physics/fracture.cpp
  src/physics/neighborhood_disconnector.cpp
  src/physics/neighborhood_disconnector.h
  src/game/player/character_config.h
  src/game/player/character_config.cpp
  src/game/player/animation_clip.h
  src/game/player/animation_clip.cpp
  src/game/player/player_character.h
  src/game/player/player_character.cpp
  src/game/player/movement_state.h
  src/game/player/movement_state.cpp
  src/game/player/player_api.h
  src/game/player/player_api.cpp
)

# ── Game executable ───────────────────────────────────────────────────────────
add_executable(Residua
  src/main.cpp
  ${RESIDUA_COMMON_SOURCES}
)

set_property(TARGET Residua PROPERTY CXX_STANDARD 20)
target_compile_definitions(Residua PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_include_directories(Residua PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(Residua PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui nlohmann_json::nlohmann_json)
target_precompile_headers(Residua PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <glm/mat4x4.hpp> <glm/vec4.hpp> <vulkan/vulkan.h>)

add_custom_command(TARGET Residua POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Residua> $<TARGET_FILE_DIR:Residua>
  COMMAND_EXPAND_LISTS
)

# ── Python extension module ───────────────────────────────────────────────────
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter Development)

if(Python3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(residua_py
    src/game/py_env.h
    src/game/py_env.cpp
    src/game/residua_py.cpp
    ${RESIDUA_COMMON_SOURCES}
  )

  set_property(TARGET residua_py PROPERTY CXX_STANDARD 20)
  target_compile_definitions(residua_py PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
  target_include_directories(residua_py PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(residua_py PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui nlohmann_json::nlohmann_json)
  target_precompile_headers(residua_py REUSE_FROM Residua)

  add_custom_command(TARGET residua_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:residua_py> $<TARGET_FILE_DIR:residua_py>
    COMMAND_EXPAND_LISTS
  )

  add_dependencies(residua_py Shaders)
else()
  message(STATUS "Python not found — skipping residua_py. Re-run cmake with -DPython3_EXECUTABLE=<path/to/python.exe>")
endif()

# ── Shader compilation ────────────────────────────────────────────────────────
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
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.3 ${GLSL} -I${PROJECT_SOURCE_DIR}/shaders -o ${SPIRV}
    DEPENDS ${GLSL} ${GLSL_HEADER_FILES})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(Shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(Residua Shaders)
