file(GLOB_RECURSE TSUNAMI_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE TSUNAMI_HEADERS CONFIGURE_DEPENDS include/*.h include/*.hpp)

add_executable(tsunami ${TSUNAMI_SOURCES} ${TSUNAMI_HEADERS})

# === includes ===
target_include_directories(tsunami PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
)

# === compile definitions ===
target_compile_definitions(tsunami PRIVATE VK_NO_PROTOTYPES)

# === slang (optional — requires ./setup.sh to download binary) ===
if(TARGET slang)
    target_link_libraries(tsunami PRIVATE slang)
endif()
