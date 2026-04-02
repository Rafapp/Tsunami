# === vk-bootstrap ===
add_subdirectory(vk-bootstrap)

# === Vulkan Memory Allocator (header-only) ===
add_library(VulkanMemoryAllocator INTERFACE)
target_include_directories(VulkanMemoryAllocator SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator/include)

# === GLFW ===
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

# === volk ===
add_subdirectory(volk)

# === Dear ImGui ===
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)
target_compile_definitions(imgui PUBLIC
    IMGUI_IMPL_VULKAN_USE_VOLK
)
target_link_libraries(imgui PUBLIC
    Vulkan::Vulkan
    glfw
    volk
)

# === miniaudio ===
set(MINIAUDIO_INSTALL OFF CACHE BOOL "" FORCE)
set(MINIAUDIO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MINIAUDIO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MINIAUDIO_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(miniaudio EXCLUDE_FROM_ALL)

# === Slang (prebuilt binary, downloaded by setup.sh) ===
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/include/slang.h)
    add_library(slang SHARED IMPORTED GLOBAL)
    target_include_directories(slang INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/include)
    if(WIN32)
        set_target_properties(slang PROPERTIES
            IMPORTED_IMPLIB   ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/lib/slang.lib
            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/bin/slang.dll)
    elseif(APPLE)
        set_target_properties(slang PROPERTIES
            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/lib/libslang.dylib)
    else()
        set_target_properties(slang PROPERTIES
            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/slang-bin/lib/libslang.so)
    endif()
else()
    message(WARNING "Slang not found — run ./setup.sh first")
endif()

# === GLM (header-only) ===
add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/glm)
