file(GLOB_RECURSE TSUNAMI_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE TSUNAMI_HEADERS CONFIGURE_DEPENDS include/*.h include/*.hpp)

add_executable(tsunami ${TSUNAMI_SOURCES} ${TSUNAMI_HEADERS})

target_include_directories(tsunami PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tsunami PRIVATE
    Vulkan::Vulkan
    vk-bootstrap
    VulkanMemoryAllocator
    glfw
    volk
)

if(TARGET slang)
    target_link_libraries(tsunami PRIVATE slang)
endif()