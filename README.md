# Tsunami

Tsunami is a Vulkan ray-traced renderer for interactive physically based rendering experiments, real-time scene diagnostics, and simulation-driven visuals.

The core innovation in this project is **HiPR**: a targeted path-tracing strategy that prioritizes sampling on the objects that matter most to the currently selected object.

## Why HiPR matters

Traditional uniform path tracing spreads budget across the full frame every update. In interactive workflows, this can waste samples on regions that are not currently important for the user's focus object.

**HiPR (Hierarchical Progressive Rendering )** addresses this by ranking objects according to their estimated contribution to the selected object and then spending a configurable sample budget on that ranked set.

At a high level, HiPR in Tsunami does the following:

1. Runs a visibility pass to cache per-pixel object IDs.
2. Collects influence statistics for objects that affect the selected object (secondary and shadow events).
3. Computes a per-object influence score and forms a Top-K ranked order.
4. Schedules focused sampling over that ranked object set.
5. Continues with full-scene accumulation while preserving focused-sample history.

Current scoring signal:

```text
score(object) = (secondary_hits + 0.5 * shadow_hits) / max(visible_hits, 1)
```

The ranking can run in full recompute mode or in an incremental/stable mode with score blending and gentle local swaps to reduce disruptive rank reshuffles.

## Features

- Vulkan compute path tracing with hardware ray tracing acceleration.
- HiPR and Naive debug render modes for side-by-side behavior comparison.
- HiPRVis diagnostic mode to visualize influence and ranking behavior.
- Interactive object selection and rich OpenPBR material editing.
- Voice-reactive material control path for live parameter modulation.
- Audience/water simulation overlays and floating-object integration.
- Shader hot reload for rapid iteration.

## Renderer Modes

In the Object Inspector panel, you can switch between:

- `HiPR`: ranked focused sampling pipeline.
- `Obj ID`: object ID debug visualization.
- `HiPR Vis`: influence/ranking visualization view.
- `Naive`: baseline uniform path tracing behavior.

## Build and setup

### Prerequisites

- CMake 3.24+
- Ninja
- Vulkan SDK (or MoltenVK on macOS)
- Python 3 (used by formatting hook installer)
- A compiler toolchain with C++20 support

Windows:

- Visual Studio 2022 (Desktop development with C++) is recommended.

macOS:

- Install MoltenVK from the Vulkan SDK:
	https://vulkan.lunarg.com/sdk/home#mac

### One-command setup scripts

From the repository root:

- Windows (Release):

	```bat
	build.bat
	```

- Windows (Debug):

	```bat
	build.bat --debug
	```

- Linux/macOS (Release):

	```bash
	./build.sh
	```

- Linux/macOS (Debug):

	```bash
	./build.sh --debug
	```

These scripts initialize submodules, set up Slang binaries, install the formatting hook, and build with CMake presets.

### Manual CMake flow

Release:

```bash
cmake --preset default
cmake --build --preset default
```

Debug:

```bash
cmake --preset debug
cmake --build --preset debug
```

## Running

The executable is generated in the preset binary folder under `bin/`:

- Release: `build/bin/tsunami` (or `build/bin/tsunami.exe` on Windows)
- Debug: `build-debug/bin/tsunami` (or `build-debug/bin/tsunami.exe` on Windows)

Run with an optional scene alias or explicit `.gltf/.glb` path:

```bash
tsunami [pool|chess|cornell|cornellsimple|sponza|<path/to/scene.gltf|.glb>]
```

If no argument is provided, Tsunami defaults to the Cornell scene.

## Controls

- `F1`: toggle all GUI panels
- `F6`: hot-reload shaders
- `F11`: toggle fullscreen
- `RMB`: capture/release mouse for fly camera
- `ESC`: release mouse capture
- `W/A/S/D`: move camera
- `Space` / `Left Ctrl`: move up/down
- `Shift`: speed boost
- `Q` / `E`: decrease/increase fly speed
- `LMB` (while cursor is free): select mesh under cursor

## HiPR tuning guide

Useful controls in `Object Inspector > HiPR Debug` and `Path Tracing`:

- `Top-K objects`: size of the ranked set sampled by HiPR.
- `Frames per object`: focused accumulation budget per ranked object.
- `SPP`: base samples per pixel.
- `Max bounces`: bounce depth limit.
- `HiPR water SPP override`: extra SPP for water object IDs (`0` disables override).
- `Influence tint` / `Heatmap tint` / `Tint strength` (HiPRVis mode): debug visibility.

Recommended workflow:

1. Select the object you care about.
2. Keep renderer mode on `HiPR` while tuning Top-K and Frames-per-object.
3. Use `HiPR Vis` to verify ranking behavior and influence concentration.
4. Compare with `Naive` mode when evaluating quality/performance tradeoffs.

## Project layout

- `tsunami/src/`: renderer, scene, simulation, UI, and Vulkan runtime code.
- `tsunami/shaders/`: Slang compute shaders, including `hipr.slang` and `naivept.slang`.
- `resources/`: scenes and meshes used for demos/experiments.
- `vendors/`: third-party dependencies (GLFW, ImGui, VMA, vk-bootstrap, OpenPBR, etc.).

## Notes

- This repository is research-oriented and evolves quickly.
- HiPR parameters and scheduling behavior are intended to be interactive and experimental.

If you are extending the renderer, start with `hipr.slang` and the Vulkan runtime dispatch flow in `tsunami/src/vulkan/vulkan.cpp`.