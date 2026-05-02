# Tsunami

> A compact wavefront path tracer built on Vulkan for physically-based interactive rendering and simulation experiments.

Tsunami is a real-time GPU-accelerated renderer featuring progressive path tracing, an audio-reactive water surface simulation, buoyancy-driven floating object physics, and a full material editor, all running on a modern Vulkan + Slang pipeline.

---

## Features

| Feature | Description |
|---|---|
| **Path Tracing** | Naive PT and HiPR (Hierarchical Progressive Rendering) modes |
| **Water Simulation** | GPU wave propagation with audio-reactive ripples |
| **Floating Objects** | Buoyancy-based physics interacting with the water surface |
| **Audio Reactivity** | Microphone input drives water and material parameters in real time |
| **OpenPBR Materials** | Adobe OpenPBR BSDF with texture LUT evaluation |
| **Interactive Editor** | ImGui panels for materials, lighting, camera, and scene control |
| **Slang Shaders** | All shaders written in Slang, compiled to SPIR-V at runtime |

---

## Dependencies

### Required

| Dependency | Install |
|---|---|
| **CMake** ≥ 3.24 | [cmake.org/download](https://cmake.org/download/) or `winget install Kitware.CMake` |
| **C++20 compiler** | Windows: [Visual Studio 2022](https://visualstudio.microsoft.com/) (select "Desktop development with C++") / macOS: `xcode-select --install` / Linux: `sudo apt install build-essential` |
| **Vulkan SDK** | [vulkan.lunarg.com](https://vulkan.lunarg.com/sdk/home); install for your platform, then reopen your terminal so `VULKAN_SDK` is set |
| **clang-format** | Windows: `winget install LLVM.LLVM` / macOS: `brew install clang-format` / Linux: `sudo apt install clang-format` |

### Vendored (fetched automatically by CMake)

| Library | Purpose |
|---|---|
| [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) | Vulkan instance / device setup |
| [volk](https://github.com/zeux/volk) | Vulkan function loader |
| [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory management |
| [Slang](https://github.com/shader-slang/slang) 2026.4.2 | Shader language + SPIR-V compiler |
| [Assimp](https://github.com/assimp/assimp) | glTF / GLB / OBJ model loading |
| [GLFW](https://www.glfw.org/) | Window and input |
| [GLM](https://github.com/g-truc/glm) | Math (header-only) |
| [Dear ImGui](https://github.com/ocornut/imgui) | UI panels |
| [miniaudio](https://miniaud.io/) | Microphone capture |
| [stb](https://github.com/nothings/stb) | Image loading (header-only) |

### Platform-Specific

| Platform | Extra Requirement |
|---|---|
| **macOS** | [MoltenVK](https://vulkan.lunarg.com/sdk/home#mac) via the Vulkan SDK |
| **Windows** | Vulkan SDK for Windows ([LunarG](https://vulkan.lunarg.com/sdk/home#windows)) |
| **Linux** | `vulkan-sdk` package from your distro or LunarG |

---

## Building

```bat
build.bat
```

On first run, CMake will automatically download Slang and other vendored dependencies. An internet connection is required.

---

## Running

```bash
./tsunami [scene]
```

Built-in scenes:

| Scene | Description |
|---|---|
| `pool` | Water pool with floating objects (default) |
| `chess` | Chess set scene |
| `cornell` | Cornell box |
| `cornellsimple` | Simplified Cornell box |
| `sponza` | Sponza atrium |

To load a custom scene, pass the full path to a `.glb` or `.gltf` file:

```bash
./tsunami C:/path/to/scene.glb
```

---

## Controls

### Camera

| Input | Action |
|---|---|
| **RMB** (hold) | Capture mouse / enter fly-cam mode |
| **ESC** | Release mouse capture |
| **W / S** | Move forward / backward |
| **A / D** | Strafe left / right |
| **Space** | Move up |
| **Ctrl** | Move down |
| **Shift** | 4× speed multiplier |
| **Q** | Halve movement speed |
| **E** | Double movement speed |

### UI

| Input | Action |
|---|---|
| **F1** | Toggle all UI panels |
| **F11** | Toggle fullscreen |
| **LMB** (viewport) | Select object under cursor |

---

## Code Style

A pre-push Git hook runs `clang-format` on all changed C++ files and auto-amends them into the last commit so the push continues without interruption.

To install the hook:

```bash
python scripts/install_cppformat.py
```

---

## License

See [LICENSE](LICENSE).
