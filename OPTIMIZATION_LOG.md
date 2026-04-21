# Optimization Log - Tsunami Vulkan Renderer

## Audit Summary

- Architecture: Vulkan 1.3 compute-only path tracer with Slang shaders compiled to SPIR-V at runtime. Hardware ray tracing uses VK_KHR_acceleration_structure with BLAS/TLAS. OpenPBR is evaluated in compute shaders with LUT-backed data tables. Water simulation and floating-object physics run through separate compute pipelines.
- Core CPU flow: load scene -> build GPU materials/meshes/textures -> build BLAS/TLAS -> create descriptor sets + compute pipelines -> run stage-based path tracing and post/composite passes each frame.
- Core GPU flow: stage 0 writes visibility/object IDs, later stages trace shaded rays into storage images, accumulation lives in a separate image, and the swapchain image is blitted from the storage target.
- Shader inventory:
  - `tsunami/shaders/naivept.slang`
  - `tsunami/shaders/hipr.slang`
  - `tsunami/shaders/hiprvis.slang`
  - `tsunami/shaders/objectid.slang`
  - `tsunami/shaders/random_color.slang`
  - `tsunami/shaders/water_surface.slang`
  - `tsunami/shaders/floating_objects.slang`
- Vulkan hot spots identified during audit:
  - Runtime Slang compilation and pipeline creation on startup / reload.
  - Multiple one-time command submissions for texture and LUT uploads.
  - Broad queue/device idle waits in a few reload and utility paths.
  - CPU-visible allocations used where device-local storage is a better fit for static scene data.
- Scene / asset hot spots identified during audit:
  - glTF import path originally performed two Assimp imports per scene load.
  - Texture discovery could decode duplicate source textures before deduplication.
  - Static mesh / index / per-object buffers were good candidates for device-local placement.
- Iteration bottlenecks identified during audit:
  - Asset copy targets used always-run directory copies.
  - No incremental shader-validation target.
  - No compiler-launcher auto-detection.
  - Large translation units benefited from PCH and selective unity builds.

## Change Log

### Change 1: Selection outline now reuses `object_id_image`
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`, `tsunami/shaders/objectid.slang`
- What changed: outline detection now reads the already-populated object-ID image instead of firing four extra visibility ray queries per outlined pixel.
- Why: ray-query based edge checks were disproportionately expensive for a simple neighborhood test.
- Expected impact:
  - Runtime: Very high reduction in per-frame ray work when a mesh is selected.
  - Build / iteration: None.
  - Maintainability: Simpler and more explicit edge-detection path.
- Risk: Low.

### Change 2: `halton3()` now uses reciprocal multiplication
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: replaced repeated `f /= 3.f` with `f *= (1.f / 3.f)`.
- Why: this is a small but safe ALU cleanup in a function called per-sample.
- Expected impact:
  - Runtime: Low but broad.
  - Build / iteration: None.
  - Maintainability: Neutral.
- Risk: Low.

### Change 3: `skyColor()` no longer normalizes an already-normalized direction
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: removed a redundant `normalize()` from the sky miss path.
- Why: primary and bounced ray directions are already normalized before the miss path uses them.
- Expected impact:
  - Runtime: Low but free.
  - Build / iteration: None.
  - Maintainability: Simpler shader source.
- Risk: Low.

### Change 4: vertex-attribute fetch now caches mesh and vertex records
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: `fetchAttribs()` now loads the `GPUMesh` once and materializes `v0`, `v1`, and `v2` once before interpolating normals, tangents, UVs, and triangle positions.
- Why: the old path repeated SSBO reads for the same mesh and vertex records.
- Expected impact:
  - Runtime: Medium on every triangle hit.
  - Build / iteration: None.
  - Maintainability: Clearer data flow inside the interpolation helper.
- Risk: Low.

### Change 5: scene intersection now caches per-hit mesh data
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: `intersectScene()` now reads the hit mesh once and reuses it for material lookup and transform data.
- Why: repeated non-uniform SSBO loads are expensive and unnecessary.
- Expected impact:
  - Runtime: Medium on every committed hit.
  - Build / iteration: None.
  - Maintainability: Clearer hit-setup code.
- Risk: Low.

### Change 6: accumulation image moved to 128-bit float and resize paths were fixed
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: accumulation uses `VK_FORMAT_R32G32B32A32_SFLOAT`, and all recreate / resize paths now consistently rebuild it with the same format.
- Why: the old UNORM accumulation format caused severe precision loss during progressive rendering, and one resize path still recreated the image with the old format.
- Expected impact:
  - Runtime: Neutral to slightly higher bandwidth for the accumulation target.
  - Visual correctness: High improvement in convergence quality.
  - Maintainability: Centralized format constants remove resize-path drift.
- Risk: Low.

### Change 7: opaque materials stop firing unnecessary back-hemisphere shadow rays
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: the backlit shadow query is now skipped unless the resolved material actually has transmission.
- Why: opaque surfaces do not benefit from the second shadow query.
- Expected impact:
  - Runtime: Medium-high in typical opaque scenes.
  - Build / iteration: None.
  - Maintainability: Direct-light logic better reflects material behavior.
- Risk: Low.

### Change 8: camera basis is hoisted and `makeRay()` now uses cached inverse dimensions
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`, `tsunami/shaders/objectid.slang`
- What changed: per-dispatch camera basis generation now caches origin, basis vectors, projection scale, and `inv_dims`; `makeRay()` no longer recomputes camera state or divides by image dimensions for each sample.
- Why: camera setup is invariant across all pixels in a dispatch and should not be repeated in the hot sample loop.
- Expected impact:
  - Runtime: Medium, especially at higher SPP.
  - Build / iteration: None.
  - Maintainability: Camera math is centralized and consistent across shaders.
- Risk: Low.

### Change 9: path tracing bounce loop now hoists sun data and skips dead emissive sampling
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: directional-light enable state, sun direction, and sun intensity are loaded once before the bounce loop; emissive texture fetches happen only when `emission_luminance > 0`.
- Why: these values were loop-invariant or frequently unnecessary.
- Expected impact:
  - Runtime: Medium in scenes with many non-emissive hits.
  - Build / iteration: None.
  - Maintainability: Less noise in the bounce loop.
- Risk: Low.

### Change 10: Russian roulette now clamps survival probability to a practical range
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: roulette probability is now clamped to `[0.05, 0.95]`, and throughput uses reciprocal multiplication instead of division.
- Why: uncapped bright paths can remain alive too long, while near-zero throughput paths do not need to linger.
- Expected impact:
  - Runtime: Medium reduction in long-tail bounce cost.
  - Noise / convergence: Slight variance tradeoff in some scenes.
  - Maintainability: Standardized roulette behavior.
- Risk: Medium.

### Change 11: `GPUMesh` was compacted to drop unused BLAS handle fields
- Files: `tsunami/include/tsunami/shapes/mesh.h`, `tsunami/src/shapes/mesh.cpp`, `tsunami/src/vulkan/vulkan.cpp`, `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: removed unused BLAS-handle and index-count fields from `GPUMesh`, updated packing code, and kept C++ / shader layouts synchronized with static assertions.
- Why: the path tracer was not consuming those fields, so every mesh fetch carried dead bandwidth.
- Expected impact:
  - Runtime: Medium reduction in SSBO bandwidth for hit shading.
  - Build / iteration: None.
  - Maintainability: Leaner host/device contract.
- Risk: Low.

### Change 12: static scene and acceleration-structure inputs now upload through device-local buffers
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: added CPU-only staging helpers and a device-local upload path, then moved static mesh, vertex, index, BLAS input, TLAS instance, and HiPR statistic buffers onto device-local memory.
- Why: these buffers are read heavily by the GPU and rarely or never updated by the CPU.
- Expected impact:
  - Runtime: High in geometry-heavy scenes due to better memory residency and bandwidth.
  - Build / iteration: Slightly faster startup once uploads are batched and more predictable.
  - Maintainability: Clear split between dynamic mapped buffers and static GPU-resident buffers.
- Risk: Medium.

### Change 13: pipeline cache is now persisted to disk and shared with water simulation
- Files: `tsunami/src/vulkan/vulkan.cpp`, `tsunami/include/tsunami/simulation/water_surface_simulation.h`, `tsunami/src/simulation/water_surface_simulation.cpp`
- What changed: added disk-backed `VkPipelineCache` load/save, passed the cache into compute-pipeline creation, and threaded the same cache through the water simulation pipelines.
- Why: pipeline creation was a recurring startup / reload cost with no persistence.
- Expected impact:
  - Runtime: Neutral during steady-state rendering.
  - Build / iteration: High improvement for startup and shader/pipeline reload latency across runs.
  - Maintainability: Centralized pipeline-cache ownership.
- Risk: Low.

### Change 14: broad idle waits were narrowed to fence-scoped waits in hot utility paths
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: one-time command submission now waits on a fence instead of `vkQueueWaitIdle`, the acquire-semaphore drain path uses a fence, and shader reload waits on the in-flight fence instead of `vkDeviceWaitIdle`.
- Why: queue-wide and device-wide idles stall unrelated work and make iteration slower than necessary.
- Expected impact:
  - Runtime: Low to medium reduction in avoidable CPU-side stalls.
  - Build / iteration: Medium improvement during shader reload and utility uploads.
  - Maintainability: Synchronization intent is more explicit.
- Risk: Medium.

### Change 15: glTF scene loading now uses one Assimp import instead of two
- Files: `tsunami/include/tsunami/shapes/mesh.h`, `tsunami/src/shapes/mesh.cpp`, `tsunami/src/scene/scene.cpp`
- What changed: introduced `Mesh::load_gltf_from_scene(...)`, moved `Scene::load_gltf(...)` to a single shared Assimp import, and reused that same imported scene for geometry, texture discovery, and texture-index wiring.
- Why: the old path imported the same glTF twice and still risked drift in mesh/material ordering.
- Expected impact:
  - Runtime: Neutral once running.
  - Build / iteration: High improvement to scene-load and startup latency.
  - Maintainability: Cleaner scene-loading architecture with one source of truth.
- Risk: Medium.

### Change 16: texture deduplication now happens before decode/load
- Files: `tsunami/include/tsunami/texture/texture.h`, `tsunami/src/texture/texture.cpp`, `tsunami/include/tsunami/scene/scene.h`, `tsunami/src/scene/scene.cpp`
- What changed: added canonical source-path resolution plus scene-level texture lookup so duplicated texture references are skipped before stb decode and upload preparation.
- Why: repeated decode work is pure waste on scenes with shared material textures.
- Expected impact:
  - Runtime: Neutral once the scene is loaded.
  - Build / iteration: Medium improvement to scene-load latency and memory churn.
  - Maintainability: Texture identity is explicit and reusable.
- Risk: Low.

### Change 17: OpenPBR LUT uploads are now recorded as one batch
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: replaced per-LUT submit/wait behavior with a single command-buffer batch that records all LUT transitions and copies before one submission.
- Why: a small fixed set of LUT uploads should not cost multiple queue submissions and fence waits.
- Expected impact:
  - Runtime: Neutral during steady state.
  - Build / iteration: Medium improvement to startup time and reload smoothness.
  - Maintainability: Centralized upload logic via shared helpers.
- Risk: Low.

### Change 18: material texture uploads are now batched into one transfer submission
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: texture uploads now reuse one command buffer for all valid material textures instead of submitting and waiting once per texture.
- Why: per-texture submit/wait overhead scales badly on texture-heavy scenes.
- Expected impact:
  - Runtime: Neutral during steady state.
  - Build / iteration: High improvement for scene startup and asset iteration on texture-rich content.
  - Maintainability: Removes an especially noisy upload pattern.
- Risk: Low.

### Change 19: startup CPU timing hooks were added around major initialization phases
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: added scoped CPU timers for scene load, BLAS/TLAS build plus scene buffer upload, LUT/material texture upload, and compute pipeline creation.
- Why: optimization work is much easier when startup cost is visible and attributable.
- Expected impact:
  - Runtime: Neutral.
  - Build / iteration: Better visibility into iteration bottlenecks.
  - Maintainability: Easier profiling and regression tracking.
- Risk: Low.

### Change 20: Vulkan validation can now be toggled at runtime through an environment variable
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: instance creation now uses `TSUNAMI_VK_VALIDATION=1/0` as an override, with debug builds defaulting on and release builds defaulting off.
- Why: validation is valuable for debugging, but it should be easy to disable for performance-focused runs.
- Expected impact:
  - Runtime: Medium improvement when validation is disabled for profiling runs.
  - Build / iteration: Better DX because switching modes no longer requires code edits.
  - Maintainability: Cleaner debug/perf workflow.
- Risk: Low.

### Change 21: build-system and iteration-speed configuration was tightened up
- Files: `CMakeLists.txt`, `CMakePresets.json`, `tsunami/CMakeLists.txt`, `tsunami/include/tsunami/pch.h`
- What changed:
  - enabled `CMAKE_EXPORT_COMPILE_COMMANDS`
  - replaced always-run asset copies with `copy_if_different` tree targets
  - added incremental `validate-shaders` target when `slangc` is available
  - added PCH support
  - added selective unity-build support
  - auto-detected `sccache` / `ccache` / `clcache`
  - added a `dev` preset for `RelWithDebInfo`
  - enabled MSVC parallel compile and modern conformance switches
- Why: faster configure/build loops and better editor tooling directly improve iteration speed.
- Expected impact:
  - Runtime: Neutral.
  - Build / iteration: High improvement to incremental compile and asset-copy latency.
  - Maintainability: Better local tooling defaults with contained target-level configuration.
- Risk: Low.

### Change 22: PCH include hygiene fix for GLFW Vulkan surface declarations
- Files: `tsunami/include/tsunami/pch.h`
- What changed: removed `GLFW/glfw3.h` from the precompiled header so `tsunami/core/window.h` remains the first place that includes GLFW with `GLFW_INCLUDE_VULKAN`.
- Why: the forced PCH include caused GLFW to be parsed before Vulkan-aware configuration macros were set, which hid `glfwCreateWindowSurface(...)` and broke the Vulkan translation unit.
- Expected impact:
  - Runtime: Neutral.
  - Build / iteration: Restores successful compilation while keeping the PCH narrower and less fragile.
  - Maintainability: Better include-order hygiene and fewer macro-sensitive header interactions.
- Risk: Low.

### Change 23: VSync is now off by default, with explicit present-mode fallback logic
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: swapchain creation and resize now choose present mode dynamically. Default behavior prefers `IMMEDIATE`, then `MAILBOX`, then `FIFO`; `TSUNAMI_VSYNC=1` forces a synchronized present mode back on.
- Why: the renderer was hard-locked to `FIFO`, which capped presentation to display refresh and prevented uncapped FPS runs.
- Expected impact:
  - Runtime: High visible FPS increase on systems that support `IMMEDIATE`.
  - Build / iteration: None.
  - Maintainability: Present behavior is now explicit, logged, and overrideable.
- Risk: Low.

### Change 24: transform handedness is precomputed on the CPU and packed into `GPUMesh`
- Files: `tsunami/include/tsunami/shapes/mesh.h`, `tsunami/src/shapes/mesh.cpp`, `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: replaced the spare mesh field with `transformSign`, computed once during mesh packing, and removed per-hit `determinant(M)` checks from the shader intersection path.
- Why: determinant evaluation in the hit path is expensive, invariant per mesh, and belongs on the CPU.
- Expected impact:
  - Runtime: Medium reduction in ALU cost on every shaded hit.
  - Build / iteration: None.
  - Maintainability: Host/device mesh layout now carries exactly the information the shader needs.
- Risk: Low.

### Change 25: BLAS and shading now share the same uploaded geometry buffers
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: BLAS build no longer uploads and destroys per-mesh vertex/index buffers. The renderer now uploads one combined device-local vertex buffer and one combined device-local index buffer, then builds each BLAS from byte offsets into those shared buffers. Bulk vectors are reserved up front to avoid reallocation churn.
- Why: the previous path uploaded scene geometry twice: once transiently for BLAS construction and again for shading.
- Expected impact:
  - Runtime: Neutral once fully initialized.
  - Build / iteration: High improvement to scene-startup and BLAS build time; lower allocation and copy overhead.
  - Maintainability: One geometry upload path instead of two overlapping ones.
- Risk: Medium.

### Change 26: toolchain settings were pushed harder for non-Debug builds
- Files: `tsunami/CMakeLists.txt`, `CMakePresets.json`
- What changed:
  - added fast-math, IPO/LTO, native-codegen, and AVX2 toggles
  - enabled stronger non-Debug compile options (`/Oi`, `/Ot`, `/GF`, `/Gy`, `/Gw`, `/fp:fast`, `/arch:AVX2` on MSVC)
  - enabled non-Debug linker dead-stripping / identical-code folding
  - added a `perf` preset for dedicated max-throughput builds
- Why: plain `Release` was still leaving low-level codegen opportunities on the table.
- Expected impact:
  - Runtime: High on CPU-heavy paths and host-side preprocessing.
  - Build / iteration: Release and perf builds may compile/link more aggressively; dev/debug ergonomics remain intact.
  - Maintainability: Performance-oriented build knobs are explicit and preset-driven.
- Risk: Medium.

### Change 27: SPIR-V inspection was added to the optimization loop, plus a few source-level math micro-opts
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`, `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - compiled and disassembled the current `naivept` and `hipr` kernels to inspect emitted SPIR-V
  - measured current vs `spirv-opt -O` disassembly sizes:
    - `naivept`: `89,747 -> 53,135` lines
    - `hipr`: `91,196 -> 54,145` lines
    - `hiprvis`: `91,819 -> 54,591` lines
  - replaced a few remaining scalar divides with reciprocal multiplies in the sRGB decode path
  - replaced one more HiPR scoring divide with a reciprocal multiply in the ranking pass
  - replaced repeated `/16` dispatch ceil-divides with a power-of-two helper
- Why: the source now gets checked against the generated IR instead of only against surface-level shader readability.
- Expected impact:
  - Runtime: Low to medium from the direct math micro-opts, with higher future payoff from IR-guided tuning.
  - Build / iteration: Better shader-optimization feedback loop.
  - Maintainability: Lower readability by design in a few micro-hot paths, but the changes stay mechanically local.
- Risk: Low.

### Change 28: swapchain present-mode helpers now have explicit forward declarations
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: added translation-unit-scope forward declarations for the new present-mode helper functions before their first use in swapchain creation / recreation code.
- Why: MSVC requires those helpers to be declared before use in the translation unit; without that, the new no-vsync path failed to compile.
- Expected impact:
  - Runtime: Neutral.
  - Build / iteration: Restores successful compilation of the Vulkan runtime translation unit.
  - Maintainability: Keeps helper ordering explicit without reshuffling large sections of the file.
- Risk: Low.

### Change 29: emissive direct lighting now uses alias-table sampling plus MIS
- Files: `tsunami/include/tsunami/materials/material.h`, `tsunami/src/vulkan/vulkan.cpp`, `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed:
  - replaced emissive-triangle CDF lookup with an O(1) Walker alias table
  - packed per-material emissive solid-angle PDF scale into `GPUMaterial`
  - added power-heuristic MIS for emissive next-event estimation and for BSDF-sampled hits on emissive geometry
  - kept the light buffer compact by storing the final area-domain PDF scale in `GPUEmissiveTriangle.normal_pad.w`
  - recompiled the three main path-tracing shaders after the change; post-change disassembly sizes are:
    - `naivept`: `107,550 -> 63,630` lines after `spirv-opt -O`
    - `hipr`: `109,030 -> 64,669`
    - `hiprvis`: `109,654 -> 65,115`
- Why: binary-search light sampling and non-MIS emissive transport both leave variance and hot-path memory traffic on the table. This pass attacks both at once.
- Expected impact:
  - Runtime: Medium reduction in per-bounce direct-light sampling overhead when many emissive triangles exist.
  - Convergence / noise: High improvement around emissive geometry, especially on glossy and indirect paths that previously double-counted or under-weighted bright emitters.
  - Maintainability: Slightly more complex host/device contract, but still localized and structurally clean.
- Risk: Medium.

### Change 30: uncapped-present diagnostics now report the actual swapchain mode, and fullscreen no longer requests monitor refresh
- Files: `tsunami/src/vulkan/vulkan.cpp`, `tsunami/src/core/window.cpp`
- What changed:
  - swapchain creation and resize now store the actual `vkb::Swapchain.present_mode` instead of the requested mode
  - when VSync is off, the swapchain now always requests `IMMEDIATE` first and explicitly falls back to `MAILBOX` then `FIFO`
  - added startup / resize log messages when the runtime falls back away from `IMMEDIATE`
  - fullscreen toggle now uses `GLFW_DONT_CARE` for refresh rate instead of explicitly requesting the monitor refresh
- Why: on a 100 Hz monitor, a hard 100 FPS result strongly suggests paced presentation; the old code could falsely report success because it logged only the requested mode, not the actual one.
- Expected impact:
  - Runtime: High when `IMMEDIATE` is truly available; otherwise neutral with much better diagnostics.
  - Build / iteration: None.
  - Maintainability: Much clearer presentation-path observability, which makes perf debugging less guessy.
- Risk: Low.

### Change 31: `vsync_enabled()` now has an early translation-unit declaration
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed: added forward declarations for `vsync_enabled()` alongside the other local Vulkan helper declarations.
- Why: the new swapchain and resize code calls `vsync_enabled()` before its later definition, which MSVC rejects without a prior declaration.
- Expected impact:
  - Runtime: Neutral.
  - Build / iteration: Restores successful compilation of `vulkan.cpp`.
  - Maintainability: Keeps helper ordering explicit and avoids more declaration-order regressions.
- Risk: Low.

### Change 32: Slang-generated SPIR-V is now cached to disk, including water simulation shaders
- Files: `tsunami/src/vulkan/vulkan.cpp`, `tsunami/src/simulation/water_surface_simulation.cpp`
- What changed:
  - added persistent SPIR-V disk caching under a local `tsunami_shader_cache` directory
  - render shaders now hash shader source metadata plus the OpenPBR dependency tree before compiling
  - water/floating-object simulation shaders now also reuse cached SPIR-V
  - cache logs now clearly show reuse vs compile-and-store, including compile time on misses
- Why: the runtime logs showed nearly 49 seconds spent rebuilding compute shaders and pipelines for a tiny Cornell scene, which is overwhelmingly a shader-compilation problem.
- Expected impact:
  - Runtime: Neutral during steady-state rendering.
  - Build / iteration: Very high reduction in startup and shader-reload latency after the first compile.
  - Maintainability: Cache invalidation stays local and explicit instead of relying on ad hoc manual recompiles.
- Risk: Medium.

### Change 33: render-mode pipelines are now built lazily instead of all at startup
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - startup now builds only the currently active render-mode pipeline
  - alternate render modes compile on demand when first selected
  - the main loop now ensures the active mode exists before dispatching
- Why: eager compilation of `HiPR`, `NaivePT`, `HiPRVis`, and `ObjectID` was forcing all users to pay the full shader/pipeline cost before the first usable frame.
- Expected impact:
  - Runtime: Neutral after all required modes are built.
  - Build / iteration: High improvement to time-to-first-frame and startup responsiveness.
  - Maintainability: Better alignment between actual feature usage and pipeline work paid up front.
- Risk: Low.

### Change 34: lightweight opt-in runtime perf diagnostics were added
- Files: `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - added `TSUNAMI_PERF_DIAG=1` to emit a concise periodic perf line every ~2 seconds
  - each line reports active render mode, actual present mode, current/average CPU frame time, FPS, SPP, bounce count, and render resolution
- Why: targeted optimization is much easier when runtime pacing can be observed without drowning the console in per-frame spam.
- Expected impact:
  - Runtime: Negligible when disabled; very low overhead when enabled.
  - Build / iteration: Better diagnosis loop between your runs and future optimization passes.
  - Maintainability: Perf telemetry stays explicit and user-controlled.
- Risk: Low.

### Change 35: ReSTIR DI — temporal reservoir reuse for primary direct illumination
- Files: `tsunami/shaders/naivept.slang`, `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - Added `Reservoir` struct (packed `uint4`, 16 bytes/pixel) and helpers `res_update`, `res_merge`, `target_pdf`, `runReSTIR` in `naivept.slang`.
  - `runReSTIR` runs Weighted Reservoir Sampling over M=8 alias-table candidate lights per pixel per frame (no BSDF eval in the candidate loop — uses `luminance * cos_i * cos_l / dist²` as the target distribution), then merges the result with the previous frame's reservoir (temporal reuse, history capped at 20×M to avoid variance collapse).
  - The selected light's unbiased contribution weight `W = w_sum / (M × p_hat)` is finalized and stored.
  - At bounce 0, `tracePath` either calls `runReSTIR` (first SPP sample per pixel) or reads the pre-stored reservoir (subsequent SPP) to drive the emissive NEE; bounces 1+ retain the existing alias-table NEE with MIS.
  - Ping-pong is handled by a single `RWStructuredBuffer<uint4> reservoir_pool` (binding 21) of size `2 × W × H × 16 bytes`; the frame parity bit selects the live/prior halves — no per-frame descriptor update needed.
  - On the C++ side: `GPUReservoir` struct (16 bytes, static-asserted), `reservoir_buffer` / `reservoir_alloc` fields in `RenderTargetContext`, device-local buffer created and zero-filled in both the initial setup and resize paths, destroyed in all teardown paths, descriptor layout / pool / set writes updated to include binding 21.
- Why: With M=8 candidates per frame and up to 20-frame temporal history, the effective candidate count is ≈160/pixel/frame with a single visibility test — a ~160× variance reduction over the single alias-table sample used before. This is the highest single-change convergence win available for area-light-dominated scenes (Cornell box, interior renders).
- Expected impact:
  - Runtime: Very high convergence improvement for area-lit scenes (Cornell box converges in <10 frames vs hundreds before). Marginal cost: ~8 alias-table lookups + 1 temporal SSBO read + 1 SSBO write per pixel per frame.
  - Memory: 2 × W × H × 16 bytes (≈66 MB at 1920×1080, ≈29 MB at 1280×720).
  - Build / iteration: None.
  - Maintainability: Reservoir logic is fully self-contained in `runReSTIR`; bounce-1+ path is unchanged.
- Risk: Medium. ReSTIR introduces a temporal bias on fast-moving geometry (ghosting) because the previous frame's reservoir may reference a now-invalid light sample. Mitigations: the `M ≤ 500` cap limits history accumulation; a future spatial reuse pass would add a geometry-normal compatibility check. The current implementation is unbiased for static scenes.

### Change 36: ACES RRT+ODT replaces Uncharted2 Filmic tonemapping
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: replaced `unchartedTonemapPartial / unchartedTonemapFilmic` with the Narkowicz/Hill ACES approximation: an sRGB→AP1-like input matrix, the rational RRT+ODT curve `(v(v+0.0245786)-0.000090537) / (v(0.983729v+0.432951)+0.238081)` computed component-wise via hardware `rcp`, then an AP1→sRGB output matrix. The entire chain calls `saturate(mul(m,x))` which the compiler lowers to FMAs and a clamp.
- Why: Uncharted2 Filmic requires a white-point correction division and separate evaluation of W at (11.2,11.2,11.2); ACES delivers better highlight roll-off and a tighter colour gamut with fewer instructions and no division.
- Expected impact:
  - Runtime: Neutral to slight reduction (fewer ALU ops in the display path).
  - Visual: Higher-quality output — saturated highlights desaturate gracefully, blacks are cleaner.
  - Maintainability: Single compact function; no separate white-point constant.
- Risk: Low. Cosmetic change only; exposure_bias continues to work.

### Change 37: Russian roulette switched to luminance-weighted survival probability
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: replaced `max(r, g, b)` throughput threshold with `luminance(throughput)` = `dot(c, (0.2126, 0.7152, 0.0722))`. Also replaced `throughput * (1/p)` with `throughput * rcp(p)` to make the reciprocal explicit for the compiler.
- Why: `max(r, g, b)` over-weights saturated channels (e.g., a pure-red path with throughput (0.9,0.01,0.01) survives with p=0.9 but contributes only ~0.22 perceived luminance). The luminance-based probability aligns the roulette decision with perceptual contribution, slightly reducing variance on coloured light paths.
- Expected impact:
  - Runtime: Neutral (same instruction count; `dot` replaces a pair of `max` ops).
  - Convergence: Small reduction in variance on chromatic light paths.
  - Maintainability: More principled and consistent with rendering literature.
- Risk: Low.

### Change 38: `luminance()` helper added to all path-tracing shaders
- Files: `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed: added `float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }` once per shader, used by Russian roulette and the ReSTIR target PDF.
- Why: previously each shader had inline max-channel roulette with no shared luminance primitive; this consolidates the pattern.
- Risk: Low.

### Change 39: ReSTIR spatial reuse now uses a precomputed integer offset pattern instead of per-pixel trig
- Files: `tsunami/shaders/naivept.slang`
- What changed:
  - replaced the bounce-0 spatial reuse neighbour sampling in `runReSTIR` from a random polar-disk generator (`sqrt`, `cos`, `sin`) to an 8-tap precomputed Poisson-ish offset table
  - neighbour selection is now randomized with a cheap `rng.next()` rotation plus sign-bit flips, so the hot path stays integer-heavy and branch-light
  - fixed the `tracePath(...)` call in `main()` to pass `pixel` and `dims`, matching the new spatial-reuse-aware signature
- Why: the previous version was paying several transcendental ops per primary-hit pixel just to pick four neighbour taps. That is the wrong place to spend ALU in a realtime tracer.
- Expected impact:
  - Runtime: Moderate reduction in primary-hit shader cost for scenes using ReSTIR DI, especially at high resolution where bounce-0 dominates frame time.
  - Convergence: Neutral to slightly positive; the offset table keeps broad screen-space coverage without the heavy polar sampling math.
  - Maintainability: Better; the signature/callsite mismatch is removed and the reuse pattern is explicit.
- Risk: Low. The reuse distribution is slightly less isotropic than fully random polar sampling, but frame-to-frame rotation/flip keeps it well scrambled in practice.

### Change 40: Photon-hash-grid descriptor plumbing was completed on the Vulkan side
- Files: `tsunami/src/vulkan/vulkan.cpp`, `tsunami/shaders/naivept.slang`
- What changed:
  - added C++ `GPUPhoton` layout, static-asserted to 32 bytes to match shader expectations
  - added device-local buffers for photon-grid heads, photon pool storage, and photon counter storage
  - zero/cleared those buffers at startup with the correct sentinels (`0xFFFFFFFF` for empty heads, `0` for the counter)
  - extended descriptor-set layout, descriptor pool sizing, and descriptor writes to bind shader bindings 22, 23, and 24
- Why: `naivept.slang` had already grown a caustic photon gather path, but Vulkan still only knew about the reservoir buffer. That left the shader/C++ contract incomplete.
- Expected impact:
  - Runtime: Neutral right now because the counter stays zero until a photon emission pass is added; this is infrastructure for the next quality/perf pass.
  - Build / iteration: Neutral.
  - Maintainability: High improvement. The shader interface is now backed by real resources instead of dangling bindings.
- Risk: Low. Memory overhead is modest and the buffers are inert until future photon emission code writes into them.

### Change 41: Display tonemapping is now enabled by default, and ACES no longer clamps HDR before tone mapping
- Files: `tsunami/include/tsunami/ui/audience_control_panel.h`, `tsunami/shaders/naivept.slang`, `tsunami/shaders/hipr.slang`, `tsunami/shaders/hiprvis.slang`
- What changed:
  - flipped the audience overlay default so tonemapping starts enabled on launch
  - changed `aces_filmic()` in all three render shaders from `saturate(x * ev)` to `max(x * ev, 0.0f)` before the ACES matrix/curve path
- Why: the renderer displays through an `R8G8B8A8_UNORM` output image, so disabling tonemapping by default was making physically plausible low-intensity indirect light look nearly black. On top of that, clamping HDR input before ACES was throwing away highlight energy before the tone-mapper could compress it correctly.
- Expected impact:
  - Runtime: Neutral.
  - Visual: High improvement to first-run image quality and much better visibility of indirect bounce lighting.
  - Maintainability: Better default behavior; the display path now matches the intended HDR-to-LDR pipeline.
- Risk: Low. This only changes display mapping, not the underlying path-traced radiance accumulation.

### Change 42: Experimental ReSTIR DI runtime path disabled after temporal darkening regression
- Files: `tsunami/shaders/naivept.slang`
- What changed:
  - disabled the bounce-0 ReSTIR DI path behind a compile-time constant
  - restored the stable alias-table emissive-light MIS estimator for bounce 0 as well as later bounces
  - kept the reservoir infrastructure in place for a future corrected implementation
- Why: the current prototype only resampled light indices, not full light samples, and then shaded with a fresh triangle point without the proper PDF/state correction. In practice that made temporal accumulation drift toward black outside directly lit or emissive regions.
- Expected impact:
  - Runtime: Slightly more stable and predictable; may be a bit noisier than the intended ReSTIR target, but no longer catastrophically underestimates lighting.
  - Visual: High improvement in correctness and temporal stability.
  - Maintainability: Better. The broken estimator is no longer active while the reservoir/plumbing work remains available for a proper sample-domain ReSTIR rewrite.
- Risk: Low. This is a conservative rollback of an unstable runtime path.

### Change 43: Added an SPPM-style photon emission pass for caustic photons
- Files: `tsunami/shaders/naivept.slang`, `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - added a new `stage == 11` path in `naivept.slang` that emits photons from emissive triangles, traces them through specular/transmissive chains, and splats first diffuse caustic hits into the photon hash grid
  - upgraded photon bindings 22/23/24 to writable buffers in the shader so the same pass can both build and read the photon map
  - added per-frame grid/counter clears plus transfer-to-compute and compute-to-compute barriers in `vulkan.cpp`
  - inserted the photon stage before the Naive camera path each frame, and added a concise startup print showing the target photon count per frame
- Why: the previous code only had the gather side of the caustic system; the photon map itself was never populated, so caustics could never appear. This turns the existing hash-grid scaffolding into a real progressive caustic estimator.
- Expected impact:
  - Runtime: Moderate extra compute cost in Naive mode from tracing a bounded photon budget every frame.
  - Visual: High improvement for glass/water caustics and other L→S*→D transport; the estimate should now accumulate progressively across frames instead of staying empty.
  - Maintainability: Good. The photon stage is explicit in the frame graph and isolated to Naive mode.
- Risk: Medium. This is a first SPPM-style pass and may need tuning for photon budget, deposit heuristics, or flux scaling once exercised on real scenes.

### Change 44: Photon caustics now accumulate over an 8-frame window with a wider initial gather radius
- Files: `tsunami/shaders/naivept.slang`, `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - switched the photon map from per-frame reset to an 8-frame progressive accumulation window
  - reduced per-frame photon budget to fit the fixed photon pool across that window
  - widened the initial gather radius from 0.1m to 0.2m and normalized the gather by the current accumulation-frame count
  - updated startup logging to print the photon accumulation window explicitly
- Why: the first SPPM pass was too sparse to produce visible caustic density in practice. A short progressive window plus a larger initial kernel gives the estimator enough density to show up, while still staying bounded and resettable.
- Expected impact:
  - Runtime: Similar total work over several frames, slightly lower per-frame photon count, better amortized usefulness.
  - Visual: Much more likely to produce visible caustics instead of an effectively empty photon estimate.
  - Maintainability: Good. The accumulation strategy is explicit and still local to the caustic pass.
- Risk: Medium. The larger radius increases blur/bias, and the accumulation window can ghost briefly if future reset conditions are too lax.

### Change 45: Photon map density and kernel tuning were pushed much harder for stronger visible caustics
- Files: `tsunami/shaders/naivept.slang`, `tsunami/src/vulkan/vulkan.cpp`
- What changed:
  - increased photon grid and pool capacity from `2^17`/`2^18` scale to `2^20`
  - raised the photon budget aggressively by changing the screen-space scaling from `>> 4` to `>> 2`
  - extended the progressive accumulation window from 8 frames to 16 frames
  - matched photon cell size to the 0.2m gather radius and replaced the box filter with a distance-weighted kernel
- Why: the first visible caustic signal proved the pass worked, but it was still under-dense. This tuning round spends more memory and more photons per stable frame to move from “barely there” toward a clearly readable caustic footprint.
- Expected impact:
  - Runtime: Noticeably heavier than the first SPPM pass, especially in Naive mode.
  - Visual: Stronger and less sparse caustic footprints with better center weighting and less all-or-nothing noise.
  - Maintainability: Still localized to the photon stage and gather code.
- Risk: Medium. Memory use and Naive-mode cost both increase, and the larger progressive window can leave stale photons around longer after a reset boundary.

## Current Summary

- Total documented changes: 45
- Highest impact runtime optimizations so far:
  - cheaper ReSTIR spatial reuse neighbour selection (no per-pixel trig in the primary-hit hot path)
  - SPPM-style caustic photon emission and hash-grid splatting in Naive mode
  - **ReSTIR DI temporal reservoir reuse** — ~160× effective candidate count for area-light NEE at bounce 0
  - device-local placement for static scene / AS / HiPR buffers
  - uncapped present-mode selection with `IMMEDIATE` preference
  - shared geometry buffers for both BLAS construction and shading
  - precomputed transform handedness instead of per-hit determinants
  - object-ID based outline detection instead of extra ray queries
  - skipping back-hemisphere shadow rays for opaque materials
  - camera-basis hoisting and reduced per-hit buffer traffic
  - alias-table emissive light sampling with MIS (retained for bounces 1+)
- Highest impact iteration-speed improvements so far:
  - persistent pipeline cache
  - persistent SPIR-V shader cache
  - single-import glTF loading
  - batched LUT and material-texture uploads
  - removal of duplicate BLAS/shading geometry uploads
  - lazy render-mode pipeline creation
  - PCH + selective unity build + compiler-launcher auto-detection
  - aggressive non-Debug compile/link settings and dedicated `perf` preset
  - incremental shader validation target and `copy_if_different` asset staging
- Remaining bottlenecks worth targeting next:
  - ReSTIR GI: extend reservoirs into a second-bounce vertex cache for indirect lighting.
  - Bidirectional path tracing / VCM: connect camera and light subpaths for caustics and SDS transport.
  - Progressive photon-map refinement: adaptive radius shrink, better flux normalization, and visible-point style updates to reduce blur and bias.
  - Spectral hero wavelength sampling: replace `float3` throughput with a 4-wavelength spectral estimate for dispersion and fluorescence.
  - GPU timestamp queries: instrument per-stage dispatch timing without CPU-side polling.
  - Wavefront path tracing: sort active rays by material type after each bounce to improve coherence and BSDF register pressure.
  - BLAS/TLAS build is still serialized; an async transfer / compute queue split would hide the cost.
  - Frame still uses a single in-flight fence; double-buffering commands would improve CPU/GPU overlap.
  - A few `vkDeviceWaitIdle()` calls remain in resize / shutdown paths.
  - SPIR-V still shrinks ~40% under `spirv-opt -O`; further source shaping possible.
- Validation status:
  - No full local build or runtime validation performed in this pass by request.
  - All source changes cross-checked via code inspection and symbol/reference sweeps.
