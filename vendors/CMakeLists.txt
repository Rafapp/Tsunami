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