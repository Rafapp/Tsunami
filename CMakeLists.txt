cmake_minimum_required(VERSION 3.24...4.1)
project(tsunami VERSION 0.1.0 LANGUAGES CXX)

# === C++ standard ===
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === output directories ===
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# === Vulkan ===
find_package(Vulkan REQUIRED)

# === vendors ===
add_subdirectory(vendors)

# === tsunami ===
add_subdirectory(tsunami)

# === Copy Slang DLL to bin (Windows only) ===
if(WIN32)
    # Create a small custom target to copy the Slang DLL into the runtime folder.
    # Using a separate target avoids referencing the `tsunami` target before
    # or across directory boundaries which can trigger CMake errors.
    add_custom_target(copy_slang ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/vendors/slang-bin/bin/slang.dll
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/slang.dll
        COMMENT "Copying slang.dll to bin"
    )
    add_dependencies(tsunami copy_slang)
endif()

# === Shaders ===
add_custom_target(shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/tsunami/shaders
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders
    COMMENT "Copying shaders to build folder"
)
add_dependencies(tsunami shaders)

# === Resources ===
add_custom_target(resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources
    COMMENT "Copying resources to build folder"
)
add_dependencies(tsunami resources)