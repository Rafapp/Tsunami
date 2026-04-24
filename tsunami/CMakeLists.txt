set(TSUNAMI_CORE_SOURCES
    src/main.cpp
    src/app/app.cpp
    src/app/scene_catalog.cpp
    src/audio/audio_input_utils.cpp
    src/audio/microphone_input.cpp
    src/audio/reactive_audio_controller.cpp
    src/camera/camera.cpp
    src/camera/fly_camera.cpp
    src/core/window.cpp
    src/light/pointlight.cpp
    src/materials/material.cpp
    src/scene/scene.cpp
    src/scene/transform.cpp
    src/shader/slang_shader_utils.cpp
    src/shapes/box.cpp
    src/shapes/mesh.cpp
    src/shapes/quad.cpp
    src/shapes/sphere.cpp
    src/simulation/water_audio_gate.cpp
    src/simulation/water_scene_support.cpp
    src/simulation/water_surface_simulation.cpp
    src/texture/texture.cpp
)

set(TSUNAMI_UI_SOURCES
    src/ui/audience_control_panel.cpp
    src/ui/audience_overlay.cpp
    src/ui/imgui_runtime.cpp
    src/ui/selection_panel.cpp
)

set(TSUNAMI_VULKAN_SOURCES
    src/vulkan/floating_policy.cpp
    src/vulkan/frame_policy.cpp
    src/vulkan/vulkan.cpp
    src/vulkan/water_surface_bridge.cpp
)

add_executable(tsunami)

target_sources(tsunami PRIVATE
    ${TSUNAMI_CORE_SOURCES}
    ${TSUNAMI_UI_SOURCES}
    ${TSUNAMI_VULKAN_SOURCES}
)

# === includes ===
target_include_directories(tsunami PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# === defines ===
target_compile_definitions(tsunami PRIVATE
    PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    SHADERS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    VENDORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../vendors"
)

# === links ===
target_link_libraries(tsunami PRIVATE
    Vulkan::Vulkan
    vk-bootstrap
    VulkanMemoryAllocator
    glfw
    imgui
    miniaudio
    volk
    glm
    assimp
    openpbr-bsdf
    stb
)

# === compile definitions ===
target_compile_definitions(tsunami PRIVATE VK_NO_PROTOTYPES)

# === slang (optional - requires ./build.sh to download binary) ===
if(TARGET slang)
    target_link_libraries(tsunami PRIVATE slang)
endif()
