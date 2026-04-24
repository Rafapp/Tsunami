// Purpose: Implements the core Vulkan renderer runtime: init, resources, pipelines, and main loop.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#ifndef VK_NO_PROTOTYPES
#	define VK_NO_PROTOTYPES
#endif

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "tsunami/camera/fly_camera.h"
#include "tsunami/core/window.h"
#include "tsunami/scene/scene.h"
#include "vk_mem_alloc.h"

#include "app/internal/scene_catalog.h"
#include "audio/internal/audio_input_utils.h"
#include "shader/internal/slang_shader_utils.h"
#include "tsunami/vulkan/vulkan.h"

using vulkan::Runtime;

// ============================================================
// === OpenPBR LUT data — C++ array mode
// ============================================================
// The shader uses OPENPBR_USE_TEXTURE_LUTS=1. Here on the CPU side we include
// openpbr in C++ + array mode (OPENPBR_USE_TEXTURE_LUTS=0) so the raw LUT
// data arrays are accessible as plain C++ statics for GPU upload.
#define OPENPBR_LANGUAGE_TARGET_CPP 1
#define OPENPBR_USE_TEXTURE_LUTS 0
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0        // uint32 for broadest platform compat
#include "openpbr.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// Raw data headers — these define the C arrays we will upload.
// HOW TO USE:
//   1. Open openpbr_data_constants.h     → table resolution constants
//   2. Open impl/data/openpbr_energy_arrays.h → the 7 energy table uint32 arrays
//   3. Open impl/data/openpbr_ltc_array.h     → the float3 LTC table array
// Then fill in upload_openpbr_luts() below.
#include "impl/data/openpbr_energy_arrays.h"
#include "impl/data/openpbr_ltc_array.h"
#include "openpbr_data_constants.h"
// Remove macros so they don't leak.
#undef OPENPBR_LANGUAGE_TARGET_CPP
#undef OPENPBR_USE_TEXTURE_LUTS
#undef OPENPBR_ENERGY_TABLES_USE_UINT16

// ============================================================
// === LUT / texture structs
// ============================================================
struct LutTexture2D {
	VkImage       image = VK_NULL_HANDLE;
	VmaAllocation alloc = VK_NULL_HANDLE;
	VkImageView   view  = VK_NULL_HANDLE;
	uint32_t      width = 0, height = 0;
};
struct LutTexture3D {
	VkImage       image = VK_NULL_HANDLE;
	VmaAllocation alloc = VK_NULL_HANDLE;
	VkImageView   view  = VK_NULL_HANDLE;
	uint32_t      width = 0, height = 0, depth = 0;
};
static constexpr uint32_t NUM_LUTS              = 8;
static constexpr uint32_t MAX_MATERIAL_TEXTURES = 256;
static constexpr uint32_t HIPR_TOP_K            = 16;

// =======================
// === Context structs ===
// =======================
struct SceneContext {
	void*         camera_mapped               = nullptr;
	VkBuffer      camera_buffer               = VK_NULL_HANDLE;
	VmaAllocation camera_alloc                = VK_NULL_HANDLE;
	void*         material_mapped             = nullptr;
	VkBuffer      material_buffer             = VK_NULL_HANDLE;
	VmaAllocation material_alloc              = VK_NULL_HANDLE;
	uint32_t      material_count              = 0;
	void*         mesh_mapped                 = nullptr;
	VkBuffer      mesh_buffer                 = VK_NULL_HANDLE;
	VmaAllocation mesh_alloc                  = VK_NULL_HANDLE;
	VkBuffer      vertex_buffer               = VK_NULL_HANDLE;
	VmaAllocation vertex_alloc                = VK_NULL_HANDLE;
	VkBuffer      index_buffer                = VK_NULL_HANDLE;
	VmaAllocation index_alloc                 = VK_NULL_HANDLE;
	uint32_t      mesh_count                  = 0;
	uint32_t      hipr_top_k                  = HIPR_TOP_K;
	VkBuffer      hipr_visible_count_buffer   = VK_NULL_HANDLE;
	VmaAllocation hipr_visible_count_alloc    = VK_NULL_HANDLE;
	VkBuffer      hipr_secondary_count_buffer = VK_NULL_HANDLE;
	VmaAllocation hipr_secondary_count_alloc  = VK_NULL_HANDLE;
	VkBuffer      hipr_shadow_count_buffer    = VK_NULL_HANDLE;
	VmaAllocation hipr_shadow_count_alloc     = VK_NULL_HANDLE;
	VkBuffer      hipr_score_buffer           = VK_NULL_HANDLE;
	VmaAllocation hipr_score_alloc            = VK_NULL_HANDLE;
	VkBuffer      hipr_order_buffer           = VK_NULL_HANDLE;
	VmaAllocation hipr_order_alloc            = VK_NULL_HANDLE;
} scene_ctx;
#include "tsunami/audio/microphone_input.h"
#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_scene_support.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"
#include "tsunami/ui/audience_overlay.h"
#include "tsunami/ui/imgui_runtime.h"
#include "tsunami/ui/selection_panel.h"
#include "tsunami/vulkan/water_surface_bridge.h"
#include "vulkan/internal/floating_policy.h"
#include "vulkan/internal/frame_policy.h"

struct VulkanContext {
	vkb::Instance       instance{};
	vkb::PhysicalDevice phys_device{};
	vkb::Device         log_device{};
	uint32_t            scratch_alignment;
	float               timestamp_period_ns   = 0.0f;
	bool                supports_timestamps   = false;
	VkDevice            device                = VK_NULL_HANDLE;
	VkSurfaceKHR        surface               = VK_NULL_HANDLE;
	VkQueue             graphics_queue        = VK_NULL_HANDLE;
	uint32_t            graphics_queue_family = 0;
} vulkan_ctx{};

struct SwapchainContext {
	vkb::Swapchain           swapchain{};
	std::vector<VkImage>     images;
	std::vector<VkImageView> image_views;
	std::vector<bool>        image_initialized;
	VkFormat                 image_format = VK_FORMAT_UNDEFINED;
	VkExtent2D               extent{};
} swapchain_ctx{};

struct RenderResourcesContext {
	VmaAllocator allocator = nullptr;
} render_resources_ctx{};

struct RenderTargetContext {
	VmaAllocator                       allocator;
	VkExtent2D                         extent = {};
	VkImage                            storage_image;
	VmaAllocation                      storage_image_alloc;
	VkImageView                        storage_image_view;
	bool                               storage_image_initialized     = false;
	VkImage                            object_id_image               = VK_NULL_HANDLE;
	VmaAllocation                      object_id_image_alloc         = VK_NULL_HANDLE;
	VkImageView                        object_id_image_view          = VK_NULL_HANDLE;
	VkImage                            object_id_history_image       = VK_NULL_HANDLE;
	VmaAllocation                      object_id_history_image_alloc = VK_NULL_HANDLE;
	VkImageView                        object_id_history_image_view  = VK_NULL_HANDLE;
	VkImage                            accum_image;
	VmaAllocation                      accum_image_alloc;
	VkImageView                        accum_image_view;
	VkImage                            accum_history_image       = VK_NULL_HANDLE;
	VmaAllocation                      accum_history_image_alloc = VK_NULL_HANDLE;
	VkImageView                        accum_history_image_view  = VK_NULL_HANDLE;
	VkImage                            dummy_image_2d;
	VmaAllocation                      dummy_image_2d_alloc;
	VkImageView                        dummy_image_2d_view;
	VkImage                            dummy_image_3d;
	VmaAllocation                      dummy_image_3d_alloc;
	VkImageView                        dummy_image_3d_view;
	VkSampler                          lut_sampler      = VK_NULL_HANDLE;
	VkSampler                          material_sampler = VK_NULL_HANDLE;
	std::array<LutTexture2D, NUM_LUTS> lut_textures_2d;
	std::array<LutTexture3D, NUM_LUTS> lut_textures_3d;
	std::vector<VkImage>               mat_images;
	std::vector<VmaAllocation>         mat_allocs;
	std::vector<VkImageView>           mat_views;
	void*                              water_surface_params_mapped = nullptr;
	VkBuffer                           water_surface_params_buffer = VK_NULL_HANDLE;
	VmaAllocation                      water_surface_params_alloc  = VK_NULL_HANDLE;
	VkDescriptorSetLayout              descriptor_set_layout;
	VkDescriptorPool                   descriptor_pool;
	VkDescriptorSet                    descriptor_set;
} render_target_ctx;

struct ComputePipelineContext {
	VkPipelineLayout          pipeline_layout = VK_NULL_HANDLE;
	std::array<VkPipeline, 4> pipelines{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
	                                    VK_NULL_HANDLE};
	std::array<bool, 4>       compiled{false, false, false, false};
} compute_ctx{};

struct CommandContext {
	VkCommandPool   command_pool   = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
} command_ctx{};

struct SyncContext {
	std::vector<VkSemaphore> image_available;
	std::vector<VkSemaphore> render_finished;
	VkFence                  in_flight = VK_NULL_HANDLE;
} sync_ctx{};

struct OverlayContext {
	VkRenderPass                  render_pass = VK_NULL_HANDLE;
	std::vector<VkFramebuffer>    framebuffers;
	ui::AudienceControlPanelState controls{};
	ui::AudienceDiagnostics       diagnostics{};
	bool                          show_control_panel = true;
} overlay_ctx{};

struct GpuTimingContext {
	VkQueryPool query_pool              = VK_NULL_HANDLE;
	bool        has_submission          = false;
	float       simulation_pass_time_ms = 0.0f;
	float       render_pass_time_ms     = 0.0f;
} gpu_timing_ctx{};

struct PathTracerPushConstants {
	uint32_t  frame               = 0;
	uint32_t  material_count      = 0;
	int32_t   selected_mesh_index = -1;
	uint32_t  outline_width       = 1;
	int32_t   debug_view_mode     = static_cast<int32_t>(ui::RenderDebugViewMode::HiPR);
	uint32_t  stage               = 0;        // 0 = visibility pass, 1 = path trace selected object
	uint32_t  spp                 = 1;
	uint32_t  max_bounces         = 8;
	glm::vec4 outline_color       = glm::vec4(1.0f, 0.65f, 0.15f, 1.0f);
	uint32_t  enable_tonemapping  = 1;
	float     exposure_bias       = 2.0f;
	uint32_t  hipr_object_count   = 0;
	uint32_t  hipr_top_k          = HIPR_TOP_K;
	int32_t   hipr_render_rank    = -1;
	uint32_t  hipr_incremental_sort     = 1;
	uint32_t  hipr_clear_order          = 1;
	uint32_t  hipr_vis_enable_tint      = 1;
	uint32_t  hipr_vis_rainbow_tint     = 1;
	uint32_t  hipr_reserved0            = 0;
	uint32_t  hipr_frames_per_object    = 10;
	float     hipr_score_blend          = 0.25f;
	float     hipr_vis_tint_strength    = 0.5f;
	uint32_t  skybox_enabled            = 1;
	uint32_t  directional_light_enabled = 1;
	float     sun_dir_x                 = 0.0f;
	float     sun_dir_y                 = 1.0f;
	float     sun_dir_z                 = 0.0f;
	float     sun_intensity             = 10.0f;
};

static_assert(sizeof(PathTracerPushConstants) == 124);
static constexpr uint32_t kWaterPauseBit   = 0x80000000u;
static constexpr uint32_t kSceneDynamicBit = 0x40000000u;
static constexpr uint32_t kWaterSppMask    = 0x0000FFFFu;
static_assert(sizeof(vulkan::waterbridge::WaterSurfaceParamsGpu) == 96);

struct Bounds3 {
	glm::vec3 min   = glm::vec3(0.0f);
	glm::vec3 max   = glm::vec3(0.0f);
	bool      valid = false;
};

simulation::WaterSurfaceRenderPlacement          water_surface_render_ctx{};
std::vector<glm::mat4>                           mesh_pose_transforms;
std::vector<glm::mat4>                           mesh_base_transforms;
std::vector<float>                               mesh_user_scales;
std::vector<glm::vec3>                           mesh_user_translations;
std::vector<glm::vec3>                           mesh_user_rotations_deg;
std::vector<glm::vec3>                           mesh_user_rotation_pivots;
std::vector<int>                                 mesh_floating_group_index;
std::vector<vulkan::floating::FloatingMeshGroup> floating_mesh_groups;
std::vector<float>                               floating_group_user_scales;
std::vector<glm::vec3>                           floating_group_user_translations;
std::vector<glm::vec3>                           floating_group_user_rotations_deg;
std::vector<simulation::FloatingObjectSettings>  floating_group_base_settings;
std::vector<simulation::FloatingObjectSettings>  floating_simulation_settings;
bool                                             floating_settings_dirty = false;
bool                                             tlas_update_pending     = false;

constexpr float kEditorTranslationMin = -2.0f;
constexpr float kEditorTranslationMax = 2.0f;
constexpr float kEditorRotationMinDeg = -180.0f;
constexpr float kEditorRotationMaxDeg = 180.0f;

struct FrameTimingHistory {
	static constexpr size_t kSampleWindow = 120;

	std::array<float, kSampleWindow> frame_times_ms{};
	size_t                           next_index       = 0;
	size_t                           sample_count     = 0;
	float                            current_fps      = 0.0f;
	float                            average_fps      = 0.0f;
	float                            min_fps          = 0.0f;
	float                            max_fps          = 0.0f;
	float                            current_frame_ms = 0.0f;
	float                            average_frame_ms = 0.0f;
	float                            min_frame_ms     = 0.0f;
	float                            max_frame_ms     = 0.0f;

	void pushFrame(float delta_time_seconds) {
		const float clamped_delta_time = std::max(delta_time_seconds, 1.0e-6f);
		current_frame_ms               = clamped_delta_time * 1000.0f;
		current_fps                    = 1.0f / clamped_delta_time;

		frame_times_ms[next_index] = current_frame_ms;
		next_index                 = (next_index + 1) % frame_times_ms.size();
		sample_count               = std::min(sample_count + 1, frame_times_ms.size());

		float total_ms = 0.0f;
		min_frame_ms   = frame_times_ms[0];
		max_frame_ms   = frame_times_ms[0];
		for (size_t index = 0; index < sample_count; ++index) {
			total_ms += frame_times_ms[index];
			min_frame_ms = std::min(min_frame_ms, frame_times_ms[index]);
			max_frame_ms = std::max(max_frame_ms, frame_times_ms[index]);
		}

		average_frame_ms = total_ms / static_cast<float>(sample_count);
		min_fps          = 1000.0f / std::max(max_frame_ms, 1.0e-6f);
		max_fps          = 1000.0f / std::max(min_frame_ms, 1.0e-6f);
		average_fps      = 1000.0f / std::max(average_frame_ms, 1.0e-6f);
	}
} frame_timing_history{};

void check_vk_result(VkResult result) {
	if (result == VK_SUCCESS) {
		return;
	}

	std::cerr << "[Vulkan] VkResult = " << result << "\n";
	if (result < 0) {
		throw std::runtime_error("Vulkan call failed");
	}
}

ui::ImGuiRendererInitInfo makeImGuiRendererInitInfo() {
	ui::ImGuiRendererInitInfo init_info{};
	init_info.instance              = vulkan_ctx.instance.instance;
	init_info.physical_device       = vulkan_ctx.phys_device.physical_device;
	init_info.device                = vulkan_ctx.device;
	init_info.graphics_queue_family = vulkan_ctx.graphics_queue_family;
	init_info.graphics_queue        = vulkan_ctx.graphics_queue;
	init_info.image_count           = static_cast<uint32_t>(swapchain_ctx.images.size());
	init_info.render_pass           = overlay_ctx.render_pass;
	init_info.check_vk_result       = check_vk_result;
	return init_info;
}

vulkan::waterbridge::WaterSurfaceParamsBufferContext makeWaterSurfaceParamsBufferContext() {
	return vulkan::waterbridge::WaterSurfaceParamsBufferContext{
	    .params_mapped = render_target_ctx.water_surface_params_mapped,
	    .allocator     = render_target_ctx.allocator,
	    .params_alloc  = render_target_ctx.water_surface_params_alloc,
	};
}

struct CpuRay {
	glm::vec3 origin{};
	glm::vec3 direction{};
};

glm::vec2 cursorPositionInFramebuffer(GLFWwindow* window, uint32_t framebuffer_width,
                                      uint32_t framebuffer_height) {
	double cursor_x = 0.0;
	double cursor_y = 0.0;
	glfwGetCursorPos(window, &cursor_x, &cursor_y);

	int window_width  = 1;
	int window_height = 1;
	glfwGetWindowSize(window, &window_width, &window_height);

	const float scale_x = window_width > 0 ? static_cast<float>(framebuffer_width) /
	                                             static_cast<float>(window_width) :
	                                         1.0f;
	const float scale_y = window_height > 0 ? static_cast<float>(framebuffer_height) /
	                                              static_cast<float>(window_height) :
	                                          1.0f;

	return glm::vec2(static_cast<float>(cursor_x) * scale_x,
	                 static_cast<float>(cursor_y) * scale_y);
}

CpuRay buildPickRay(const GPUCamera& camera, uint32_t framebuffer_width,
                    uint32_t framebuffer_height, const glm::vec2& cursor_position) {
	const glm::vec3 origin = glm::vec3(camera.position);
	const glm::vec3 target = glm::vec3(camera.target);
	const glm::vec3 up     = glm::normalize(glm::vec3(camera.up));
	const float     fov    = camera.fov_near_far.x;
	const float     aspect = static_cast<float>(framebuffer_width) /
	                         static_cast<float>(std::max(framebuffer_height, 1u));

	const float half_height = std::tan(glm::radians(fov) * 0.5f);
	const float half_width  = aspect * half_height;

	const glm::vec3 forward = glm::normalize(target - origin);
	const glm::vec3 right   = glm::normalize(glm::cross(forward, up));
	const glm::vec3 up_axis = glm::cross(right, forward);

	const glm::vec2 clamped_cursor = glm::clamp(
	    cursor_position, glm::vec2(0.0f),
	    glm::vec2(static_cast<float>(framebuffer_width), static_cast<float>(framebuffer_height)));
	const glm::vec2 uv =
	    glm::vec2(clamped_cursor.x / static_cast<float>(std::max(framebuffer_width, 1u)),
	              clamped_cursor.y / static_cast<float>(std::max(framebuffer_height, 1u)));

	const float u = (2.0f * uv.x - 1.0f) * half_width;
	const float v = (1.0f - 2.0f * uv.y) * half_height;

	return CpuRay{origin, glm::normalize(forward + u * right + v * up_axis)};
}

bool intersectRayAabb(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& min,
                      const glm::vec3& max, float& out_t_min, float& out_t_max) {
	float t_min = 0.0f;
	float t_max = std::numeric_limits<float>::infinity();

	for (int axis = 0; axis < 3; ++axis) {
		const float dir = direction[axis];
		const float ori = origin[axis];

		if (std::abs(dir) < 1.0e-8f) {
			if (ori < min[axis] || ori > max[axis]) {
				return false;
			}
			continue;
		}

		float t0 = (min[axis] - ori) / dir;
		float t1 = (max[axis] - ori) / dir;
		if (t0 > t1) {
			std::swap(t0, t1);
		}

		t_min = std::max(t_min, t0);
		t_max = std::min(t_max, t1);
		if (t_min > t_max) {
			return false;
		}
	}

	out_t_min = t_min;
	out_t_max = t_max;
	return t_max >= 0.0f;
}

bool intersectRayTriangle(const CpuRay& ray, const glm::vec3& v0, const glm::vec3& v1,
                          const glm::vec3& v2, float& out_t) {
	constexpr float kEpsilon = 1.0e-7f;

	const glm::vec3 edge1 = v1 - v0;
	const glm::vec3 edge2 = v2 - v0;
	const glm::vec3 pvec  = glm::cross(ray.direction, edge2);
	const float     det   = glm::dot(edge1, pvec);

	if (std::abs(det) < kEpsilon) {
		return false;
	}

	const float     inv_det = 1.0f / det;
	const glm::vec3 tvec    = ray.origin - v0;
	const float     u       = glm::dot(tvec, pvec) * inv_det;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	const glm::vec3 qvec = glm::cross(tvec, edge1);
	const float     v    = glm::dot(ray.direction, qvec) * inv_det;
	if (v < 0.0f || (u + v) > 1.0f) {
		return false;
	}

	const float t = glm::dot(edge2, qvec) * inv_det;
	if (t <= kEpsilon) {
		return false;
	}

	out_t = t;
	return true;
}

bool intersectRayWaterSelectionSurface(const CpuRay& ray, float& out_t) {
	if (!water_surface_render_ctx.enabled) {
		return false;
	}

	const float denom = glm::dot(ray.direction, water_surface_render_ctx.normal);
	if (std::abs(denom) < 1.0e-6f) {
		return false;
	}

	const float t =
	    glm::dot(water_surface_render_ctx.center - ray.origin, water_surface_render_ctx.normal) /
	    denom;
	if (t <= 1.0e-4f) {
		return false;
	}

	const glm::vec3 hit_point = ray.origin + ray.direction * t;
	const glm::vec3 delta     = hit_point - water_surface_render_ctx.center;
	const float     u         = glm::dot(delta, water_surface_render_ctx.axis_u);
	const float     v         = glm::dot(delta, water_surface_render_ctx.axis_v);
	if (std::abs(u) > water_surface_render_ctx.half_extent_u ||
	    std::abs(v) > water_surface_render_ctx.half_extent_v) {
		return false;
	}

	out_t = t;
	return true;
}

int pickMeshAtCursor(const Scene* scene, GLFWwindow* window, const GPUCamera& camera,
                     uint32_t framebuffer_width, uint32_t framebuffer_height) {
	if (scene == nullptr || window == nullptr || framebuffer_width == 0 ||
	    framebuffer_height == 0) {
		return -1;
	}

	const glm::vec2 cursor =
	    cursorPositionInFramebuffer(window, framebuffer_width, framebuffer_height);
	if (cursor.x < 0.0f || cursor.y < 0.0f || cursor.x >= static_cast<float>(framebuffer_width) ||
	    cursor.y >= static_cast<float>(framebuffer_height)) {
		return -1;
	}

	const CpuRay world_ray    = buildPickRay(camera, framebuffer_width, framebuffer_height, cursor);
	float        best_t       = std::numeric_limits<float>::infinity();
	int          best_mesh_id = -1;

	float water_hit_t = 0.0f;
	if (intersectRayWaterSelectionSurface(world_ray, water_hit_t)) {
		best_t       = water_hit_t;
		best_mesh_id = water_surface_render_ctx.mesh_index;
	}

	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto& mesh = scene->m_meshes[mesh_index];
		if (mesh == nullptr || mesh->gpuIndices.size() < 3 || mesh->gpuVertices.empty()) {
			continue;
		}

		const glm::mat4& inverse_transform = mesh->m_transform.m_inverseTransform;
		const CpuRay     local_ray{
		    glm::vec3(inverse_transform * glm::vec4(world_ray.origin, 1.0f)),
		    glm::vec3(inverse_transform * glm::vec4(world_ray.direction, 0.0f)),
		};

		if (glm::dot(local_ray.direction, local_ray.direction) < 1.0e-12f) {
			continue;
		}

		float aabb_t_min = 0.0f;
		float aabb_t_max = 0.0f;
		if (!intersectRayAabb(local_ray.origin, local_ray.direction, mesh->m_local_bounds_min,
		                      mesh->m_local_bounds_max, aabb_t_min, aabb_t_max)) {
			continue;
		}

		for (size_t index = 0; index + 2 < mesh->gpuIndices.size(); index += 3) {
			const glm::vec3& v0 = mesh->gpuVertices[mesh->gpuIndices[index + 0]].position;
			const glm::vec3& v1 = mesh->gpuVertices[mesh->gpuIndices[index + 1]].position;
			const glm::vec3& v2 = mesh->gpuVertices[mesh->gpuIndices[index + 2]].position;

			float hit_t = 0.0f;
			if (!intersectRayTriangle(local_ray, v0, v1, v2, hit_t)) {
				continue;
			}

			const glm::vec3 local_hit = local_ray.origin + local_ray.direction * hit_t;
			const glm::vec3 world_hit =
			    glm::vec3(mesh->m_transform.m_transform * glm::vec4(local_hit, 1.0f));
			const float world_t = glm::dot(world_hit - world_ray.origin, world_ray.direction);
			if (world_t > 1.0e-4f && world_t < best_t) {
				best_t       = world_t;
				best_mesh_id = mesh_index;
			}
		}
	}

	return best_mesh_id;
}

bool isSwapchainRecreationResult(VkResult result) {
	return result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR;
}

VkExtent2D computeRenderExtent(VkExtent2D swapchain_extent, float render_scale) {
	const float clamped_scale = std::clamp(render_scale, 0.50f, 1.0f);
	if (swapchain_extent.width == 0 || swapchain_extent.height == 0) {
		return swapchain_extent;
	}
	if (clamped_scale >= 0.999f) {
		return swapchain_extent;
	}

	return VkExtent2D{
	    std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(swapchain_extent.width) *
	                                                   static_cast<double>(clamped_scale)))),
	    std::max(1u,
	             static_cast<uint32_t>(std::lround(static_cast<double>(swapchain_extent.height) *
	                                               static_cast<double>(clamped_scale)))),
	};
}

void updateRenderDiagnostics(float delta_time_seconds) {
	frame_timing_history.pushFrame(delta_time_seconds);

	ui::AudienceRenderDiagnostics& render = overlay_ctx.diagnostics.render;
	render.current_fps                    = frame_timing_history.current_fps;
	render.average_fps                    = frame_timing_history.average_fps;
	render.min_fps                        = frame_timing_history.min_fps;
	render.max_fps                        = frame_timing_history.max_fps;
	render.current_frame_time_ms          = frame_timing_history.current_frame_ms;
	render.average_frame_time_ms          = frame_timing_history.average_frame_ms;
	render.min_frame_time_ms              = frame_timing_history.min_frame_ms;
	render.max_frame_time_ms              = frame_timing_history.max_frame_ms;
	render.simulation_pass_time_ms        = gpu_timing_ctx.simulation_pass_time_ms;
	render.render_pass_time_ms            = gpu_timing_ctx.render_pass_time_ms;
	render.total_pass_time_ms             = frame_timing_history.current_frame_ms;
	render.simulation_pass_fps =
	    render.simulation_pass_time_ms > 1.0e-4f ? 1000.0f / render.simulation_pass_time_ms : 0.0f;
	render.render_pass_fps =
	    render.render_pass_time_ms > 1.0e-4f ? 1000.0f / render.render_pass_time_ms : 0.0f;
	render.total_pass_fps =
	    render.total_pass_time_ms > 1.0e-4f ? 1000.0f / render.total_pass_time_ms : 0.0f;
	render.frame_sample_count    = static_cast<uint32_t>(frame_timing_history.sample_count);
	render.render_width          = render_target_ctx.extent.width;
	render.render_height         = render_target_ctx.extent.height;
	render.swapchain_image_count = static_cast<uint32_t>(swapchain_ctx.images.size());

	if (ImGui::GetCurrentContext() == nullptr) {
		render.imgui_vertex_count = 0;
		render.imgui_index_count  = 0;
		render.imgui_window_count = 0;
		return;
	}

	const ImGuiIO& io         = ImGui::GetIO();
	render.imgui_vertex_count = io.MetricsRenderVertices;
	render.imgui_index_count  = io.MetricsRenderIndices;
	render.imgui_window_count = io.MetricsRenderWindows;
}

namespace fs = std::filesystem;

static std::string toLowerCopy(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

static glm::vec3 transformPoint(const glm::mat4& transform, const glm::vec3& point) {
	return glm::vec3(transform * glm::vec4(point, 1.0f));
}

static bool matricesNearlyEqual(const glm::mat4& a, const glm::mat4& b, float epsilon = 1.0e-4f) {
	for (int column = 0; column < 4; ++column) {
		if (glm::length(a[column] - b[column]) > epsilon) {
			return false;
		}
	}
	return true;
}

static void expandBounds(Bounds3& bounds, const glm::vec3& point) {
	if (!bounds.valid) {
		bounds.min   = point;
		bounds.max   = point;
		bounds.valid = true;
		return;
	}

	bounds.min = glm::min(bounds.min, point);
	bounds.max = glm::max(bounds.max, point);
}

static Bounds3 computeMeshWorldBounds(const Mesh& mesh) {
	Bounds3         bounds{};
	const glm::vec3 local_min = mesh.m_local_bounds_min;
	const glm::vec3 local_max = mesh.m_local_bounds_max;

	for (int mask = 0; mask < 8; ++mask) {
		const glm::vec3 local_corner((mask & 1) != 0 ? local_max.x : local_min.x,
		                             (mask & 2) != 0 ? local_max.y : local_min.y,
		                             (mask & 4) != 0 ? local_max.z : local_min.z);
		expandBounds(bounds, transformPoint(mesh.m_transform.m_transform, local_corner));
	}

	return bounds;
}

static Bounds3 computeSceneMeshRangeBounds(const Scene* scene, int start_mesh_index,
                                           int end_mesh_index) {
	Bounds3 bounds{};
	if (scene == nullptr) {
		return bounds;
	}

	for (int mesh_index = start_mesh_index; mesh_index < end_mesh_index; ++mesh_index) {
		if (mesh_index < 0 || mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
		    scene->m_meshes[mesh_index] == nullptr) {
			continue;
		}

		const Bounds3 mesh_bounds = computeMeshWorldBounds(*scene->m_meshes[mesh_index]);
		if (!mesh_bounds.valid) {
			continue;
		}
		expandBounds(bounds, mesh_bounds.min);
		expandBounds(bounds, mesh_bounds.max);
	}

	return bounds;
}

static void ensureMeshTransformMetadataSize(size_t mesh_count) {
	mesh_pose_transforms.resize(mesh_count, glm::mat4(1.0f));
	mesh_base_transforms.resize(mesh_count, glm::mat4(1.0f));
	mesh_user_scales.resize(mesh_count, 1.0f);
	mesh_user_translations.resize(mesh_count, glm::vec3(0.0f));
	mesh_user_rotations_deg.resize(mesh_count, glm::vec3(0.0f));
	mesh_user_rotation_pivots.resize(mesh_count, glm::vec3(0.0f));
	mesh_floating_group_index.resize(mesh_count, -1);
}

static float meshScaleForTransform(int mesh_index) {
	if (mesh_index < 0) {
		return 1.0f;
	}

	if (mesh_index < static_cast<int>(mesh_floating_group_index.size())) {
		const int floating_group_index = mesh_floating_group_index[mesh_index];
		if (floating_group_index >= 0 &&
		    floating_group_index < static_cast<int>(floating_group_user_scales.size())) {
			return std::max(floating_group_user_scales[floating_group_index], 0.05f);
		}
	}

	if (mesh_index < static_cast<int>(mesh_user_scales.size())) {
		return std::max(mesh_user_scales[mesh_index], 0.05f);
	}

	return 1.0f;
}

static glm::vec3 clampVec3(const glm::vec3& value, float min_value, float max_value) {
	return glm::vec3(std::clamp(value.x, min_value, max_value),
	                 std::clamp(value.y, min_value, max_value),
	                 std::clamp(value.z, min_value, max_value));
}

static glm::mat4 rotationDegreesMatrix(const glm::vec3& rotation_deg) {
	glm::mat4 rotation(1.0f);
	rotation = glm::rotate(rotation, glm::radians(rotation_deg.x), glm::vec3(1.0f, 0.0f, 0.0f));
	rotation = glm::rotate(rotation, glm::radians(rotation_deg.y), glm::vec3(0.0f, 1.0f, 0.0f));
	rotation = glm::rotate(rotation, glm::radians(rotation_deg.z), glm::vec3(0.0f, 0.0f, 1.0f));
	return rotation;
}

static int floatingGroupIndexForMesh(int mesh_index) {
	if (mesh_index < 0 || mesh_index >= static_cast<int>(mesh_floating_group_index.size())) {
		return -1;
	}

	const int floating_group_index = mesh_floating_group_index[mesh_index];
	return floating_group_index >= 0 &&
	               floating_group_index < static_cast<int>(floating_mesh_groups.size()) ?
	           floating_group_index :
	           -1;
}

static glm::vec3 meshTranslationForTransform(int mesh_index) {
	const int floating_group_index = floatingGroupIndexForMesh(mesh_index);
	if (floating_group_index >= 0 &&
	    floating_group_index < static_cast<int>(floating_group_user_translations.size())) {
		return floating_group_user_translations[floating_group_index];
	}

	if (mesh_index >= 0 && mesh_index < static_cast<int>(mesh_user_translations.size())) {
		return mesh_user_translations[mesh_index];
	}

	return glm::vec3(0.0f);
}

static glm::vec3 meshRotationForTransform(int mesh_index) {
	const int floating_group_index = floatingGroupIndexForMesh(mesh_index);
	if (floating_group_index >= 0 &&
	    floating_group_index < static_cast<int>(floating_group_user_rotations_deg.size())) {
		return floating_group_user_rotations_deg[floating_group_index];
	}

	if (mesh_index >= 0 && mesh_index < static_cast<int>(mesh_user_rotations_deg.size())) {
		return mesh_user_rotations_deg[mesh_index];
	}

	return glm::vec3(0.0f);
}

static glm::vec3 meshRotationPivotForTransform(int mesh_index) {
	if (floatingGroupIndexForMesh(mesh_index) >= 0) {
		return glm::vec3(0.0f);
	}

	if (mesh_index >= 0 && mesh_index < static_cast<int>(mesh_user_rotation_pivots.size())) {
		return mesh_user_rotation_pivots[mesh_index];
	}

	return glm::vec3(0.0f);
}

static glm::mat4 meshWorldTranslationTransform(int mesh_index) {
	const glm::vec3 translation = clampVec3(meshTranslationForTransform(mesh_index),
	                                        kEditorTranslationMin, kEditorTranslationMax);
	return glm::translate(glm::mat4(1.0f), translation);
}

static glm::mat4 meshLocalOffsetTransform(int mesh_index) {
	const glm::vec3 rotation_deg = clampVec3(meshRotationForTransform(mesh_index),
	                                         kEditorRotationMinDeg, kEditorRotationMaxDeg);
	const glm::vec3 pivot        = meshRotationPivotForTransform(mesh_index);

	return glm::translate(glm::mat4(1.0f), pivot) * rotationDegreesMatrix(rotation_deg) *
	       glm::translate(glm::mat4(1.0f), -pivot);
}

static glm::mat4 composeMeshTransform(int mesh_index) {
	if (mesh_index < 0 || mesh_index >= static_cast<int>(mesh_pose_transforms.size()) ||
	    mesh_index >= static_cast<int>(mesh_base_transforms.size())) {
		return glm::mat4(1.0f);
	}

	return meshWorldTranslationTransform(mesh_index) * mesh_pose_transforms[mesh_index] *
	       meshLocalOffsetTransform(mesh_index) *
	       glm::scale(glm::mat4(1.0f), glm::vec3(meshScaleForTransform(mesh_index))) *
	       mesh_base_transforms[mesh_index];
}

static void writeMeshTransformToSceneAndGpu(Scene* scene, int mesh_index) {
	if (scene == nullptr || mesh_index < 0 ||
	    mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[mesh_index] == nullptr) {
		return;
	}

	const glm::mat4 transform            = composeMeshTransform(mesh_index);
	auto&           mesh                 = scene->m_meshes[mesh_index];
	mesh->m_transform.m_transform        = transform;
	mesh->m_transform.m_inverseTransform = glm::inverse(transform);

	if (scene_ctx.mesh_mapped != nullptr && mesh_index < static_cast<int>(scene_ctx.mesh_count)) {
		auto* gpu_meshes                        = reinterpret_cast<GPUMesh*>(scene_ctx.mesh_mapped);
		gpu_meshes[mesh_index].transform        = transform;
		gpu_meshes[mesh_index].inverseTransform = mesh->m_transform.m_inverseTransform;
	}
}

static void flushMeshTransforms(VmaAllocator allocator) {
	if (allocator == nullptr || scene_ctx.mesh_mapped == nullptr ||
	    scene_ctx.mesh_alloc == VK_NULL_HANDLE || scene_ctx.mesh_count == 0) {
		return;
	}

	vmaFlushAllocation(allocator, scene_ctx.mesh_alloc, 0,
	                   sizeof(GPUMesh) * static_cast<VkDeviceSize>(scene_ctx.mesh_count));
}

static void initializeMeshTransformMetadata(const Scene* scene) {
	mesh_pose_transforms.clear();
	mesh_base_transforms.clear();
	mesh_user_scales.clear();
	mesh_user_translations.clear();
	mesh_user_rotations_deg.clear();
	mesh_user_rotation_pivots.clear();
	mesh_floating_group_index.clear();
	floating_mesh_groups.clear();
	floating_group_user_scales.clear();
	floating_group_user_translations.clear();
	floating_group_user_rotations_deg.clear();
	floating_group_base_settings.clear();
	floating_simulation_settings.clear();
	floating_settings_dirty = false;
	tlas_update_pending     = false;

	if (scene == nullptr) {
		return;
	}

	ensureMeshTransformMetadataSize(scene->m_meshes.size());
	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto& mesh = scene->m_meshes[mesh_index];
		mesh_pose_transforms[mesh_index] =
		    mesh != nullptr ? mesh->m_transform.m_transform : glm::mat4(1.0f);
		mesh_base_transforms[mesh_index]      = glm::mat4(1.0f);
		mesh_user_scales[mesh_index]          = 1.0f;
		mesh_floating_group_index[mesh_index] = -1;
	}
}

static void syncFloatingSimulationSettings(simulation::WaterSurfaceSimulation* water_surface) {
	if (water_surface == nullptr || floating_simulation_settings.empty()) {
		return;
	}
	water_surface->setFloatingObjects(floating_simulation_settings);
}

static void syncFloatingSimulationBoundary(simulation::WaterSurfaceSimulation* water_surface) {
	if (water_surface == nullptr || !water_surface_render_ctx.enabled) {
		return;
	}
	water_surface->setFloatingSurfaceBoundary(water_surface_render_ctx.floating_surface_bounds,
	                                          water_surface_render_ctx.floating_boundary_exponent);
}

static void addFloatingMeshesFromResources(Scene* scene) {
	if (scene == nullptr || !water_surface_render_ctx.enabled) {
		return;
	}

	std::vector<fs::path> asset_paths;
	for (const fs::directory_entry& entry : fs::directory_iterator("resources/meshes")) {
		if (!entry.is_regular_file()) {
			continue;
		}

		const std::string extension = toLowerCopy(entry.path().extension().string());
		if (extension == ".glb" || extension == ".gltf") {
			asset_paths.push_back(entry.path());
		}
	}

	std::sort(asset_paths.begin(), asset_paths.end());
	if (asset_paths.size() > simulation::kMaxFloatingObjects) {
		asset_paths.resize(simulation::kMaxFloatingObjects);
	}

	ensureMeshTransformMetadataSize(scene->m_meshes.size());
	for (const fs::path& asset_path : asset_paths) {
		const int mesh_start = static_cast<int>(scene->m_meshes.size());
		scene->append_gltf(asset_path.string());
		const int mesh_end = static_cast<int>(scene->m_meshes.size());
		if (mesh_end <= mesh_start) {
			continue;
		}

		ensureMeshTransformMetadataSize(scene->m_meshes.size());
		const Bounds3 bounds = computeSceneMeshRangeBounds(scene, mesh_start, mesh_end);
		if (!bounds.valid) {
			continue;
		}

		const glm::vec3   asset_center     = 0.5f * (bounds.min + bounds.max);
		const glm::vec3   asset_world_size = glm::max(bounds.max - bounds.min, glm::vec3(0.01f));
		const std::string asset_name       = asset_path.stem().string();
		const std::string asset_name_lower = toLowerCopy(asset_name);
		const float target_major_world = std::max(vulkan::floating::floatingTargetMajorWorldSize(
		                                              water_surface_render_ctx, asset_name_lower),
		                                          0.08f);
		const float source_major_world = std::max(asset_world_size.x, asset_world_size.z);
		const float default_scale      = target_major_world / std::max(source_major_world, 1.0e-4f);

		vulkan::floating::FloatingMeshGroup group{};
		group.display_name     = asset_name;
		group.simulation_index = static_cast<int>(floating_simulation_settings.size());
		const simulation::FloatingObjectSettings settings =
		    vulkan::floating::makeFloatingObjectSettings(
		        water_surface_render_ctx, asset_name, asset_world_size, default_scale,
		        static_cast<uint32_t>(group.simulation_index));
		floating_simulation_settings.push_back(settings);
		floating_group_base_settings.push_back(settings);
		floating_group_user_scales.push_back(1.0f);
		floating_group_user_translations.push_back(glm::vec3(0.0f));
		floating_group_user_rotations_deg.push_back(glm::vec3(0.0f));
		const glm::mat4 pose =
		    vulkan::floating::makeFloatingWorldPose(water_surface_render_ctx, settings);

		for (int mesh_index = mesh_start; mesh_index < mesh_end; ++mesh_index) {
			const glm::mat4 pose =
			    vulkan::floating::makeFloatingWorldPose(water_surface_render_ctx, settings);
			auto& mesh = scene->m_meshes[mesh_index];
			if (mesh == nullptr) {
				continue;
			}

			const glm::mat4 centered_transform =
			    glm::translate(glm::mat4(1.0f), -asset_center) * mesh->m_transform.m_transform;
			mesh_pose_transforms[mesh_index] = pose;
			mesh_base_transforms[mesh_index] =
			    glm::scale(glm::mat4(1.0f), glm::vec3(default_scale)) * centered_transform;
			mesh_floating_group_index[mesh_index] = static_cast<int>(floating_mesh_groups.size());
			mesh_user_rotation_pivots[mesh_index] = glm::vec3(0.0f);
			group.mesh_indices.push_back(mesh_index);
			writeMeshTransformToSceneAndGpu(scene, mesh_index);
		}

		floating_mesh_groups.push_back(std::move(group));
	}
}

static bool updateFloatingMeshTransformsFromSimulation(
    Scene* scene, const simulation::WaterSurfaceSimulation* water_surface, VmaAllocator allocator) {
	if (scene == nullptr || water_surface == nullptr || floating_mesh_groups.empty()) {
		return false;
	}

	const std::vector<simulation::FloatingObjectRenderData> render_data =
	    water_surface->floatingObjectRenderData();
	bool       any_pose_changed = false;
	const auto has_valid_axis   = [](const glm::vec4& axis) {
		const float length = glm::length(glm::vec3(axis));
		return std::isfinite(length) && length > 1.0e-4f;
	};

	for (const vulkan::floating::FloatingMeshGroup& group : floating_mesh_groups) {
		if (group.simulation_index < 0 ||
		    group.simulation_index >= static_cast<int>(render_data.size())) {
			continue;
		}

		const simulation::FloatingObjectRenderData& object_render_data =
		    render_data[group.simulation_index];
		if (!has_valid_axis(object_render_data.axis_x) ||
		    !has_valid_axis(object_render_data.axis_y) ||
		    !has_valid_axis(object_render_data.axis_z)) {
			continue;
		}

		const glm::mat4 pose =
		    vulkan::floating::makeFloatingWorldPose(water_surface_render_ctx, object_render_data);
		for (int mesh_index : group.mesh_indices) {
			if (mesh_index < 0 || mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
			    mesh_index >= static_cast<int>(mesh_pose_transforms.size()) ||
			    scene->m_meshes[mesh_index] == nullptr) {
				continue;
			}

			if (matricesNearlyEqual(mesh_pose_transforms[mesh_index], pose)) {
				continue;
			}

			mesh_pose_transforms[mesh_index] = pose;
			writeMeshTransformToSceneAndGpu(scene, mesh_index);
			any_pose_changed = true;
		}
	}

	if (any_pose_changed) {
		flushMeshTransforms(allocator);
		tlas_update_pending = true;
	}

	return any_pose_changed;
}

static float selectedMeshScale() {
	const int selected_mesh_index = ui::selection_ctx.selected_mesh_index;
	if (selected_mesh_index < 0) {
		return 1.0f;
	}

	const int floating_group_index = floatingGroupIndexForMesh(selected_mesh_index);
	if (floating_group_index >= 0 &&
	    floating_group_index < static_cast<int>(floating_group_user_scales.size())) {
		return floating_group_user_scales[floating_group_index];
	}

	if (selected_mesh_index < static_cast<int>(mesh_user_scales.size())) {
		return mesh_user_scales[selected_mesh_index];
	}

	return 1.0f;
}

static glm::vec3 selectedMeshTranslation() {
	const int selected_mesh_index = ui::selection_ctx.selected_mesh_index;
	if (selected_mesh_index < 0) {
		return glm::vec3(0.0f);
	}

	return meshTranslationForTransform(selected_mesh_index);
}

static glm::vec3 selectedMeshRotation() {
	const int selected_mesh_index = ui::selection_ctx.selected_mesh_index;
	if (selected_mesh_index < 0) {
		return glm::vec3(0.0f);
	}

	return meshRotationForTransform(selected_mesh_index);
}

static void refreshSelectedTransformEditor() {
	ui::selection_ctx.editor_scale        = selectedMeshScale();
	ui::selection_ctx.editor_translation  = selectedMeshTranslation();
	ui::selection_ctx.editor_rotation_deg = selectedMeshRotation();
}

static bool vec3NearlyEqual(const glm::vec3& a, const glm::vec3& b, float epsilon = 1.0e-4f) {
	return glm::length(a - b) <= epsilon;
}

static bool applySelectedTransformEditor(Scene* scene, VmaAllocator allocator) {
	if (scene == nullptr) {
		return false;
	}

	const int selected_mesh_index = ui::selection_ctx.selected_mesh_index;
	if (selected_mesh_index < 0 ||
	    selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		ui::selection_ctx.editor_scale        = 1.0f;
		ui::selection_ctx.editor_translation  = glm::vec3(0.0f);
		ui::selection_ctx.editor_rotation_deg = glm::vec3(0.0f);
		return false;
	}

	const float     clamped_scale        = std::clamp(ui::selection_ctx.editor_scale, 0.10f, 3.00f);
	const glm::vec3 clamped_translation  = clampVec3(ui::selection_ctx.editor_translation,
	                                                 kEditorTranslationMin, kEditorTranslationMax);
	const glm::vec3 clamped_rotation     = clampVec3(ui::selection_ctx.editor_rotation_deg,
	                                                 kEditorRotationMinDeg, kEditorRotationMaxDeg);
	ui::selection_ctx.editor_scale       = clamped_scale;
	ui::selection_ctx.editor_translation = clamped_translation;
	ui::selection_ctx.editor_rotation_deg = clamped_rotation;

	std::vector<int> mesh_indices_to_update;
	const int        floating_group_index = floatingGroupIndexForMesh(selected_mesh_index);
	if (floating_group_index >= 0 &&
	    floating_group_index < static_cast<int>(floating_mesh_groups.size()) &&
	    floating_group_index < static_cast<int>(floating_group_user_scales.size()) &&
	    floating_group_index < static_cast<int>(floating_group_user_translations.size()) &&
	    floating_group_index < static_cast<int>(floating_group_user_rotations_deg.size())) {
		const bool scale_changed =
		    std::abs(floating_group_user_scales[floating_group_index] - clamped_scale) > 1.0e-4f;
		const bool translation_changed = !vec3NearlyEqual(
		    floating_group_user_translations[floating_group_index], clamped_translation);
		const bool rotation_changed = !vec3NearlyEqual(
		    floating_group_user_rotations_deg[floating_group_index], clamped_rotation);
		if (!scale_changed && !translation_changed && !rotation_changed) {
			return false;
		}

		floating_group_user_scales[floating_group_index]        = clamped_scale;
		floating_group_user_translations[floating_group_index]  = clamped_translation;
		floating_group_user_rotations_deg[floating_group_index] = clamped_rotation;
		mesh_indices_to_update = floating_mesh_groups[floating_group_index].mesh_indices;

		const int simulation_index = floating_mesh_groups[floating_group_index].simulation_index;
		if (scale_changed && simulation_index >= 0 &&
		    simulation_index < static_cast<int>(floating_simulation_settings.size()) &&
		    floating_group_index < static_cast<int>(floating_group_base_settings.size())) {
			simulation::FloatingObjectSettings scaled_settings =
			    floating_group_base_settings[floating_group_index];
			scaled_settings.size *= clamped_scale;
			scaled_settings.mass *= clamped_scale * clamped_scale * clamped_scale;
			scaled_settings.buoyancy_strength *= clamped_scale * clamped_scale;
			scaled_settings.drift_radius *= std::clamp(std::sqrt(clamped_scale), 0.5f, 2.0f);
			scaled_settings.waterline_offset *= clamped_scale;
			floating_simulation_settings[simulation_index] = scaled_settings;
			floating_settings_dirty                        = true;
		}
	}

	if (mesh_indices_to_update.empty()) {
		if (selected_mesh_index >= static_cast<int>(mesh_user_scales.size()) ||
		    selected_mesh_index >= static_cast<int>(mesh_user_translations.size()) ||
		    selected_mesh_index >= static_cast<int>(mesh_user_rotations_deg.size())) {
			return false;
		}

		const bool scale_changed =
		    std::abs(mesh_user_scales[selected_mesh_index] - clamped_scale) > 1.0e-4f;
		const bool translation_changed =
		    !vec3NearlyEqual(mesh_user_translations[selected_mesh_index], clamped_translation);
		const bool rotation_changed =
		    !vec3NearlyEqual(mesh_user_rotations_deg[selected_mesh_index], clamped_rotation);
		if (!scale_changed && !translation_changed && !rotation_changed) {
			return false;
		}

		mesh_user_scales[selected_mesh_index]        = clamped_scale;
		mesh_user_translations[selected_mesh_index]  = clamped_translation;
		mesh_user_rotations_deg[selected_mesh_index] = clamped_rotation;
		mesh_indices_to_update.push_back(selected_mesh_index);
	}

	for (int mesh_index : mesh_indices_to_update) {
		writeMeshTransformToSceneAndGpu(scene, mesh_index);
	}
	flushMeshTransforms(allocator);
	tlas_update_pending      = true;
	water_surface_render_ctx = simulation::buildWaterSurfacePlacement(scene);
	return true;
}

void Runtime::createSwapchainResources() {
	vkb::SwapchainBuilder swapchain_builder{vulkan_ctx.log_device};
	auto                  swap_ret =
	    swapchain_builder
	        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
	        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	        .set_desired_extent(m_window->width(), m_window->height())
	        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
	        .build();
	if (!swap_ret) {
		throw std::runtime_error("failed to create swapchain");
	}
	swapchain_ctx.swapchain    = swap_ret.value();
	swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
	swapchain_ctx.extent       = swapchain_ctx.swapchain.extent;
	std::cout << "[INFO] Created swapchain (format: " << swapchain_ctx.image_format
	          << ", extent: " << swapchain_ctx.extent.width << "x" << swapchain_ctx.extent.height
	          << ")\n";

	auto images_ret = swapchain_ctx.swapchain.get_images();
	if (!images_ret) {
		throw std::runtime_error("failed to get swapchain images");
	}
	swapchain_ctx.images            = images_ret.value();
	swapchain_ctx.image_initialized = std::vector<bool>(swapchain_ctx.images.size(), false);
	std::cout << "[INFO] Acquired " << swapchain_ctx.images.size() << " swapchain images\n";

	auto image_views_ret = swapchain_ctx.swapchain.get_image_views();
	if (!image_views_ret) {
		throw std::runtime_error("failed to get swapchain image views");
	}
	swapchain_ctx.image_views = image_views_ret.value();
	std::cout << "[INFO] Created swapchain image views\n";

	ui::createOverlayRenderPassAndFramebuffers(vulkan_ctx.device, swapchain_ctx.image_format,
	                                           swapchain_ctx.extent, swapchain_ctx.image_views,
	                                           overlay_ctx.render_pass, overlay_ctx.framebuffers);

	render_target_ctx.extent =
	    computeRenderExtent(swapchain_ctx.extent, overlay_ctx.controls.render_scale);
	std::vector<float> water_domain_mask = simulation::buildWaterSurfaceDomainMask(
	    m_scene.get(), water_surface_render_ctx, render_target_ctx.extent);
	m_water_surface =
	    std::make_unique<simulation::WaterSurfaceSimulation>(simulation::WaterSurfaceCreateInfo{
	        .device        = vulkan_ctx.device,
	        .allocator     = render_target_ctx.allocator,
	        .output_extent = render_target_ctx.extent,
	        .domain_mask   = std::move(water_domain_mask),
	    });
	syncFloatingSimulationBoundary(m_water_surface.get());
	syncFloatingSimulationSettings(m_water_surface.get());
	m_water_surface->requestObjectReset();
	std::cout << "[INFO] Created water surface simulation resources at "
	          << render_target_ctx.extent.width << "x" << render_target_ctx.extent.height << "\n";
}

void Runtime::destroySwapchainResources() {
	m_water_surface.reset();
	ui::destroyOverlayRenderResources(vulkan_ctx.device, overlay_ctx.render_pass,
	                                  overlay_ctx.framebuffers);

	if (!swapchain_ctx.image_views.empty()) {
		swapchain_ctx.swapchain.destroy_image_views(swapchain_ctx.image_views);
		swapchain_ctx.image_views.clear();
	}

	swapchain_ctx.images.clear();
	swapchain_ctx.image_initialized.clear();

	if (swapchain_ctx.swapchain.swapchain != VK_NULL_HANDLE) {
		vkb::destroy_swapchain(swapchain_ctx.swapchain);
		swapchain_ctx.swapchain = {};
	}

	swapchain_ctx.image_format = VK_FORMAT_UNDEFINED;
	swapchain_ctx.extent       = {};
}

static VkImageView create_image_view(VkDevice, VkImage, VkFormat, VkImageViewType,
                                     VkImageAspectFlags);
static void transition_layout(VkCommandBuffer, VkImage, VkImageLayout, VkImageLayout, VkAccessFlags,
                              VkAccessFlags, VkPipelineStageFlags, VkPipelineStageFlags);
static VkCommandBuffer begin_one_time_cmd(VkDevice, VkCommandPool);
static void            end_one_time_cmd(VkDevice, VkCommandPool, VkQueue, VkCommandBuffer);
static void            copyNaiveHistoryImages(VkCommandBuffer cmd) {
	if (render_target_ctx.accum_image == VK_NULL_HANDLE ||
	    render_target_ctx.accum_history_image == VK_NULL_HANDLE ||
	    render_target_ctx.object_id_image == VK_NULL_HANDLE ||
	    render_target_ctx.object_id_history_image == VK_NULL_HANDLE) {
		return;
	}

	transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT);
	transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT);
	transition_layout(cmd, render_target_ctx.accum_history_image, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT);
	transition_layout(cmd, render_target_ctx.object_id_history_image, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkImageCopy copy_region{};
	copy_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy_region.extent = {render_target_ctx.extent.width, render_target_ctx.extent.height, 1};

	vkCmdCopyImage(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	               render_target_ctx.accum_history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
	               &copy_region);
	vkCmdCopyImage(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	               render_target_ctx.object_id_history_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	               1, &copy_region);

	transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                  VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                  VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	transition_layout(cmd, render_target_ctx.accum_history_image,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	transition_layout(cmd, render_target_ctx.object_id_history_image,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
	                  VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void Runtime::recreateSwapchainResources() {
	uint32_t framebuffer_width  = m_window->width();
	uint32_t framebuffer_height = m_window->height();
	while (framebuffer_width == 0 || framebuffer_height == 0) {
		m_window->waitEvents();
		framebuffer_width  = m_window->width();
		framebuffer_height = m_window->height();
	}

	check_vk_result(vkDeviceWaitIdle(vulkan_ctx.device));
	ui::shutdownImGuiRenderer();
	destroySwapchainResources();
	createSwapchainResources();
	ui::initializeImGuiRenderer(makeImGuiRendererInitInfo());

	{
		const uint32_t image_count = static_cast<uint32_t>(swapchain_ctx.images.size());
		if (image_count != static_cast<uint32_t>(sync_ctx.image_available.size())) {
			for (VkSemaphore semaphore : sync_ctx.image_available) {
				if (semaphore != VK_NULL_HANDLE) {
					vkDestroySemaphore(vulkan_ctx.device, semaphore, nullptr);
				}
			}
			for (VkSemaphore semaphore : sync_ctx.render_finished) {
				if (semaphore != VK_NULL_HANDLE) {
					vkDestroySemaphore(vulkan_ctx.device, semaphore, nullptr);
				}
			}

			sync_ctx.image_available.resize(image_count, VK_NULL_HANDLE);
			sync_ctx.render_finished.resize(image_count, VK_NULL_HANDLE);

			VkSemaphoreCreateInfo semaphore_info{};
			semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			for (uint32_t index = 0; index < image_count; ++index) {
				check_vk_result(vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr,
				                                  &sync_ctx.image_available[index]));
				check_vk_result(vkCreateSemaphore(vulkan_ctx.device, &semaphore_info, nullptr,
				                                  &sync_ctx.render_finished[index]));
			}
		}
	}

	// Recreate pathtracer storage/accum/object_id (+ history) images at the current render extent.
	VmaAllocator allocator = render_target_ctx.allocator;
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.storage_image,
	                render_target_ctx.storage_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.object_id_image,
	                render_target_ctx.object_id_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_history_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.object_id_history_image,
	                render_target_ctx.object_id_history_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_history_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_history_image,
	                render_target_ctx.accum_history_image_alloc);

	const VkExtent3D new_ext = {render_target_ctx.extent.width, render_target_ctx.extent.height, 1};

	auto make_storage = [&](VkImage& img, VmaAllocation& alloc, VkFormat fmt,
	                        VkImageUsageFlags extra) {
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = fmt;
		ii.extent        = new_ext;
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		vmaCreateImage(allocator, &ii, &ai, &img, &alloc, nullptr);
	};
	make_storage(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.object_id_image, render_target_ctx.object_id_image_alloc,
	             VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.object_id_history_image,
	             render_target_ctx.object_id_history_image_alloc, VK_FORMAT_R32_SINT,
	             VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.accum_history_image, render_target_ctx.accum_history_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT);

	render_target_ctx.storage_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.storage_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.object_id_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image, VK_FORMAT_R32_SINT,
	                      VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.object_id_history_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_history_image,
	                      VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.accum_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.accum_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.accum_history_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.accum_history_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

	// Transition object_id/accum (+ history) → GENERAL so the shader can read/write them.
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.object_id_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
	}

	// Update descriptor set bindings 0 (output), 1 (accum), 13 (object_id), 22/23 (history).
	{
		VkDescriptorImageInfo out_img{};
		out_img.imageView   = render_target_ctx.storage_image_view;
		out_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo accum_img{};
		accum_img.imageView   = render_target_ctx.accum_image_view;
		accum_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo object_id_img{};
		object_id_img.imageView   = render_target_ctx.object_id_image_view;
		object_id_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo accum_history_img{};
		accum_history_img.imageView   = render_target_ctx.accum_history_image_view;
		accum_history_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo object_id_history_img{};
		object_id_history_img.imageView   = render_target_ctx.object_id_history_image_view;
		object_id_history_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkWriteDescriptorSet writes[5]{};
		writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             0,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &out_img};
		writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             1,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &accum_img};
		writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             13,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &object_id_img};
		writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             22,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &accum_history_img};
		writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             23,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &object_id_history_img};
		vkUpdateDescriptorSets(vulkan_ctx.device, 5, writes, 0, nullptr);
	}
	vulkan::waterbridge::updateWaterSurfaceImageDescriptors(
	    vulkan_ctx.device, render_target_ctx.descriptor_set, m_water_surface.get());
	vulkan::waterbridge::updateWaterSurfaceParamsBuffer(
	    makeWaterSurfaceParamsBufferContext(), water_surface_render_ctx, m_scene.get(),
	    m_water_surface.get(), vulkan::floating::firstFloatingObjectId(floating_mesh_groups));

	render_target_ctx.storage_image_initialized = false;
}

struct BLAS {
	VkAccelerationStructureKHR handle         = VK_NULL_HANDLE;
	VkBuffer                   buffer         = VK_NULL_HANDLE;
	VmaAllocation              buffer_alloc   = VK_NULL_HANDLE;
	VkDeviceAddress            device_address = 0;
};
struct AccelerationStructureContext {
	std::vector<BLAS>          blases;
	VkAccelerationStructureKHR tlas                  = VK_NULL_HANDLE;
	VkBuffer                   tlas_buffer           = VK_NULL_HANDLE;
	VmaAllocation              tlas_buffer_alloc     = VK_NULL_HANDLE;
	VkBuffer                   instance_buffer       = VK_NULL_HANDLE;
	VmaAllocation              instance_buffer_alloc = VK_NULL_HANDLE;
	void*                      instance_mapped       = nullptr;
	VkBuffer                   scratch_buffer        = VK_NULL_HANDLE;
	VmaAllocation              scratch_buffer_alloc  = VK_NULL_HANDLE;
	VkDeviceSize               scratch_size          = 0;
	uint32_t                   instance_count        = 0;
	VkDeviceAddress            tlas_address          = 0;
} as_ctx;

// ============================================================
// === Buffer helpers
// ============================================================
static VkBuffer create_gpu_buffer(VmaAllocator alloc, VkDeviceSize sz, VkBufferUsageFlags usage,
                                  VmaAllocation& out, VkDeviceSize align = 0) {
	VkBufferCreateInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size  = sz;
	bi.usage = usage;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	VkBuffer buf;
	VkResult r = align > 0 ?
	                 vmaCreateBufferWithAlignment(alloc, &bi, &ai, align, &buf, &out, nullptr) :
	                 vmaCreateBuffer(alloc, &bi, &ai, &buf, &out, nullptr);
	if (r != VK_SUCCESS)
		throw std::runtime_error("failed to create gpu buffer");
	return buf;
}
static VkBuffer create_and_upload_buffer(VmaAllocator alloc, VkDeviceSize sz, const void* data,
                                         VkBufferUsageFlags usage, VmaAllocation& out) {
	VkBufferCreateInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size  = sz;
	bi.usage = usage;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBuffer          buf;
	VmaAllocationInfo info;
	if (vmaCreateBuffer(alloc, &bi, &ai, &buf, &out, &info) != VK_SUCCESS)
		throw std::runtime_error("failed to create+upload buffer");
	memcpy(info.pMappedData, data, sz);
	return buf;
}
static VkDeviceAddress get_bda(VkDevice dev, VkBuffer buf) {
	VkBufferDeviceAddressInfo i{};
	i.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	i.buffer = buf;
	return vkGetBufferDeviceAddress(dev, &i);
}

// ============================================================
// === Image helpers
// ============================================================
static VkImageView create_image_view(VkDevice dev, VkImage img, VkFormat fmt, VkImageViewType type,
                                     VkImageAspectFlags asp = VK_IMAGE_ASPECT_COLOR_BIT) {
	VkImageViewCreateInfo vi{};
	vi.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vi.image            = img;
	vi.viewType         = type;
	vi.format           = fmt;
	vi.subresourceRange = {asp, 0, 1, 0, 1};
	VkImageView v       = VK_NULL_HANDLE;
	if (vkCreateImageView(dev, &vi, nullptr, &v) != VK_SUCCESS)
		throw std::runtime_error("failed to create image view");
	return v;
}
static void transition_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout old, VkImageLayout n,
                              VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags sp,
                              VkPipelineStageFlags dp) {
	VkImageMemoryBarrier b{};
	b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	b.oldLayout           = old;
	b.newLayout           = n;
	b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image                                       = img;
	b.subresourceRange                            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	b.srcAccessMask                               = src;
	b.dstAccessMask                               = dst;
	vkCmdPipelineBarrier(cmd, sp, dp, 0, 0, nullptr, 0, nullptr, 1, &b);
}
// Keep old name as alias for callers
static void transition_image_layout(VkCommandBuffer cmd, VkImage img, VkImageLayout o,
                                    VkImageLayout n, VkAccessFlags sa, VkAccessFlags da,
                                    VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
	transition_layout(cmd, img, o, n, sa, da, ss, ds);
}

// ============================================================
// === Command helpers
// ============================================================
static VkCommandBuffer begin_one_time_cmd(VkDevice dev, VkCommandPool pool) {
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = pool;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(dev, &ai, &cmd);
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bi);
	return cmd;
}
static void end_one_time_cmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd) {
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(q);
	vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

// ============================================================
// === Generic RGBA32F image upload
// ============================================================
static void upload_rgba32f_2d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, LutTexture2D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_2D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, 1};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);
	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, 1};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);
	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
	out.width  = w;
	out.height = h;
	vmaDestroyBuffer(alloc, stg, sa);
}
static void upload_rgba32f_3d(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                              const float* rgba, uint32_t w, uint32_t h, uint32_t d,
                              LutTexture3D& out) {
	VkDeviceSize  sz = (VkDeviceSize) w * h * d * 4 * sizeof(float);
	VmaAllocation sa;
	VkBuffer stg = create_and_upload_buffer(alloc, sz, rgba, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
	VkImageCreateInfo ii{};
	ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType     = VK_IMAGE_TYPE_3D;
	ii.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent        = {w, h, d};
	ii.mipLevels     = 1;
	ii.arrayLayers   = 1;
	ii.samples       = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
	ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo ai{};
	ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(alloc, &ii, &ai, &out.image, &out.alloc, nullptr);
	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkBufferImageCopy r{};
	r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	r.imageExtent      = {w, h, d};
	vkCmdCopyBufferToImage(cmd, stg, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
	transition_layout(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	end_one_time_cmd(dev, pool, q, cmd);
	out.view =
	    create_image_view(dev, out.image, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_VIEW_TYPE_3D);
	out.width  = w;
	out.height = h;
	out.depth  = d;
	vmaDestroyBuffer(alloc, stg, sa);
}

// =======================
// === Window resizing ===
// =======================
static uint32_t g_frame_number_ref = 0;
static void     handle_resize(uint32_t& frame_number, uint32_t fb_w, uint32_t fb_h) {
	vkDeviceWaitIdle(vulkan_ctx.device);

	uint32_t new_w = fb_w;
	uint32_t new_h = fb_h;
	if (new_w == 0 || new_h == 0)
		return;        // minimised – skip

	VmaAllocator allocator = render_target_ctx.allocator;

	// -- Destroy old swapchain image views --
	for (auto v : swapchain_ctx.image_views)
		vkDestroyImageView(vulkan_ctx.device, v, nullptr);
	swapchain_ctx.image_views.clear();

	// -- Destroy old render-target images --
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.storage_image,
	                render_target_ctx.storage_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.object_id_image,
	                render_target_ctx.object_id_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_history_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.object_id_history_image,
	                render_target_ctx.object_id_history_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_history_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_history_image,
	                render_target_ctx.accum_history_image_alloc);

	// -- Rebuild swapchain --
	vkb::Swapchain old_swapchain = swapchain_ctx.swapchain;
	auto           swap_ret =
	    vkb::SwapchainBuilder{vulkan_ctx.log_device}
	        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
	        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	        .set_desired_extent(new_w, new_h)
	        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
	        .set_old_swapchain(old_swapchain.swapchain)
	        .build();
	vkb::destroy_swapchain(old_swapchain);

	if (!swap_ret) {
		std::cerr << "[RESIZE] Failed to rebuild swapchain\n";
		return;
	}
	swapchain_ctx.swapchain    = swap_ret.value();
	swapchain_ctx.image_format = swapchain_ctx.swapchain.image_format;
	swapchain_ctx.extent       = swapchain_ctx.swapchain.extent;
	auto images_ret            = swapchain_ctx.swapchain.get_images();
	if (!images_ret)
		return;
	swapchain_ctx.images = images_ret.value();
	auto views_ret       = swapchain_ctx.swapchain.get_image_views();
	if (!views_ret)
		return;
	swapchain_ctx.image_views = views_ret.value();
	swapchain_ctx.image_initialized.assign(swapchain_ctx.images.size(), false);

	// -- Recreate render-target images at new size --
	// Use the actual swapchain extent (Vulkan may adjust from the requested size)
	const VkExtent3D ext = {swapchain_ctx.extent.width, swapchain_ctx.extent.height, 1};

	auto make_storage = [&](VkImage& img, VmaAllocation& alloc, VkFormat format,
	                        VkImageUsageFlags extra) {
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = format;
		ii.extent        = ext;
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		vmaCreateImage(allocator, &ii, &ai, &img, &alloc, nullptr);
	};

	make_storage(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.object_id_image, render_target_ctx.object_id_image_alloc,
	             VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.object_id_history_image,
	             render_target_ctx.object_id_history_image_alloc, VK_FORMAT_R32_SINT,
	             VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	make_storage(render_target_ctx.accum_history_image, render_target_ctx.accum_history_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT);

	render_target_ctx.storage_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
	                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
	render_target_ctx.object_id_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image, VK_FORMAT_R32_SINT,
	                      VK_IMAGE_VIEW_TYPE_2D);
	render_target_ctx.object_id_history_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_history_image,
	                      VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D);
	render_target_ctx.accum_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
	                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
	render_target_ctx.accum_history_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.accum_history_image,
	                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);

	// Transition accum → GENERAL
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.object_id_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
	}

	// Update the two storage-image descriptors (bindings 0 and 1)
	{
		VkDescriptorImageInfo out_img{};
		out_img.imageView   = render_target_ctx.storage_image_view;
		out_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo object_id_img{};
		object_id_img.imageView   = render_target_ctx.object_id_image_view;
		object_id_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo accum_img{};
		accum_img.imageView   = render_target_ctx.accum_image_view;
		accum_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo accum_history_img{};
		accum_history_img.imageView   = render_target_ctx.accum_history_image_view;
		accum_history_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkDescriptorImageInfo object_id_history_img{};
		object_id_history_img.imageView   = render_target_ctx.object_id_history_image_view;
		object_id_history_img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet writes[5]{};
		writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             0,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &out_img};
		writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             13,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &object_id_img};
		writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             1,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &accum_img};
		writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             22,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &accum_history_img};
		writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		             nullptr,
		             render_target_ctx.descriptor_set,
		             23,
		             0,
		             1,
		             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		             &object_id_history_img};
		vkUpdateDescriptorSets(vulkan_ctx.device, 5, writes, 0, nullptr);
	}

	// Reset sync semaphores if the swapchain image count changed
	{
		const uint32_t n = (uint32_t) swapchain_ctx.images.size();
		if (n != (uint32_t) sync_ctx.image_available.size()) {
			for (auto s : sync_ctx.image_available)
				vkDestroySemaphore(vulkan_ctx.device, s, nullptr);
			for (auto s : sync_ctx.render_finished)
				vkDestroySemaphore(vulkan_ctx.device, s, nullptr);
			sync_ctx.image_available.resize(n);
			sync_ctx.render_finished.resize(n);
			VkSemaphoreCreateInfo si{};
			si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			for (uint32_t i = 0; i < n; ++i) {
				vkCreateSemaphore(vulkan_ctx.device, &si, nullptr, &sync_ctx.image_available[i]);
				vkCreateSemaphore(vulkan_ctx.device, &si, nullptr, &sync_ctx.render_finished[i]);
			}
		}
	}

	// Recreate overlay framebuffers for the new swapchain image views / extent
	ui::recreateOverlayFramebuffers(vulkan_ctx.device, overlay_ctx.render_pass,
	                                swapchain_ctx.extent, swapchain_ctx.image_views,
	                                overlay_ctx.framebuffers);

	render_target_ctx.storage_image_initialized = false;
	frame_number                                = 0;        // reset temporal accumulation

	std::cout << "[RESIZE] " << swapchain_ctx.extent.width << "x" << swapchain_ctx.extent.height
	          << "\n";
}

// ============================================================
// === LUT data conversion helpers
// ============================================================
// Energy table: uint32 [0..65535] → RGBA32F  (R = val/65535, GBA = 0)
static std::vector<float> uint_lut_to_rgba32f(const uint32_t* data, size_t count) {
	std::vector<float> out(count * 4, 0.f);
	for (size_t i = 0; i < count; ++i)
		out[i * 4] = float(data[i]) / 65535.f;
	return out;
}
// LTC table: float3 per texel → RGBA32F  (A = 0)
static std::vector<float> float3_lut_to_rgba32f(const float* data, size_t count_vec3) {
	std::vector<float> out(count_vec3 * 4, 0.f);
	for (size_t i = 0; i < count_vec3; ++i) {
		out[i * 4 + 0] = data[i * 3 + 0];
		out[i * 4 + 1] = data[i * 3 + 1];
		out[i * 4 + 2] = data[i * 3 + 2];
	}
	return out;
}

static void upload_openpbr_luts(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q) {
	// Requires:
	//   #include "vendors/openpbr-bsdf/openpbr_data_constants.h"
	//   #include "vendors/openpbr-bsdf/impl/data/openpbr_energy_arrays.h"
	//   #include "vendors/openpbr-bsdf/impl/data/openpbr_ltc_array.h"

	// ---------------------------------------------------------------------
	// 3D energy LUTs (32 x 32 x 32)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);

		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_IdealDielectricEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize *
		                                    OpenPBR_EnergyTableSize);

		upload_rgba32f_3d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_3d[OpenPBR_LutId_OpaqueDielectricEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// 2D energy LUTs (32 x 32)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_IdealDielectricAverageEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealDielectricReflectionRatio_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealDielectricReflectionRatio]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx
		        .lut_textures_2d[OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement]);
	}

	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize * OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, OpenPBR_EnergyTableSize,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// 1D energy LUT stored as thin 2D texture (32 x 1)
	// ---------------------------------------------------------------------
	{
		auto data = uint_lut_to_rgba32f(OpenPBR_IdealMetalAverageEnergyComplement_Array,
		                                OpenPBR_EnergyTableSize);

		upload_rgba32f_2d(
		    alloc, dev, pool, q, data.data(), OpenPBR_EnergyTableSize, 1,
		    render_target_ctx.lut_textures_2d[OpenPBR_LutId_IdealMetalAverageEnergyComplement]);
	}

	// ---------------------------------------------------------------------
	// LTC LUT (32 x 32), float3 -> rgba32f
	// ---------------------------------------------------------------------
	{
		auto data = float3_lut_to_rgba32f(reinterpret_cast<const float*>(OpenPBR_LTC_Array),
		                                  OpenPBR_LTCTableSize * OpenPBR_LTCTableSize);

		upload_rgba32f_2d(alloc, dev, pool, q, data.data(), OpenPBR_LTCTableSize,
		                  OpenPBR_LTCTableSize,
		                  render_target_ctx.lut_textures_2d[OpenPBR_LutId_LTC]);
	}

	std::cout << "[INFO] Uploaded OpenPBR LUTs\n";
}

// ============================================================
// === Material texture upload
// ============================================================
static void upload_material_textures(VmaAllocator alloc, VkDevice dev, VkCommandPool pool,
                                     VkQueue                                      q,
                                     const std::vector<std::shared_ptr<Texture>>& textures) {
	auto& images = render_target_ctx.mat_images;
	auto& allocs = render_target_ctx.mat_allocs;
	auto& views  = render_target_ctx.mat_views;
	images.reserve(textures.size());
	allocs.reserve(textures.size());
	views.reserve(textures.size());

	for (const auto& tex : textures) {
		if (!tex || !tex->valid()) {
			images.push_back(VK_NULL_HANDLE);
			allocs.push_back(VK_NULL_HANDLE);
			views.push_back(VK_NULL_HANDLE);
			continue;
		}
		VkDeviceSize      sz = (VkDeviceSize) tex->width * tex->height * 4;
		VmaAllocation     sa;
		VkBuffer          stg = create_and_upload_buffer(alloc, sz, tex->pixels.data(),
		                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sa);
		const VkFormat    fmt = tex->is_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = fmt;
		ii.extent        = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai2{};
		ai2.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage       img;
		VmaAllocation imgalloc;
		if (vmaCreateImage(alloc, &ii, &ai2, &img, &imgalloc, nullptr) != VK_SUCCESS)
			throw std::runtime_error("failed to create material texture");
		VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy r{};
		r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		r.imageExtent      = {(uint32_t) tex->width, (uint32_t) tex->height, 1};
		vkCmdCopyBufferToImage(cmd, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
		transition_layout(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(dev, pool, q, cmd);
		images.push_back(img);
		allocs.push_back(imgalloc);
		views.push_back(create_image_view(dev, img, fmt, VK_IMAGE_VIEW_TYPE_2D));
		vmaDestroyBuffer(alloc, stg, sa);
	}
	std::cout << "[INFO] Uploaded " << textures.size() << " material textures\n";
}

static size_t render_mode_index(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return 0;
		case ui::RenderDebugViewMode::Naive:
			return 1;
		case ui::RenderDebugViewMode::HiPRVis:
			return 2;
		case ui::RenderDebugViewMode::ObjectIds:
			return 3;
	}
	return 0;
}

static constexpr std::array<ui::RenderDebugViewMode, 4> kRenderModes = {
    ui::RenderDebugViewMode::HiPR,
    ui::RenderDebugViewMode::Naive,
    ui::RenderDebugViewMode::HiPRVis,
    ui::RenderDebugViewMode::ObjectIds,
};

static const char* render_mode_shader_filename(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return "hipr.slang";
		case ui::RenderDebugViewMode::Naive:
			return "naivept.slang";
		case ui::RenderDebugViewMode::HiPRVis:
			return "hiprvis.slang";
		case ui::RenderDebugViewMode::ObjectIds:
			return "objectid.slang";
	}
	return "hipr.slang";
}

static const char* render_mode_label(ui::RenderDebugViewMode mode) {
	switch (mode) {
		case ui::RenderDebugViewMode::HiPR:
			return "HiPR";
		case ui::RenderDebugViewMode::Naive:
			return "NaivePT";
		case ui::RenderDebugViewMode::HiPRVis:
			return "HiPRVis";
		case ui::RenderDebugViewMode::ObjectIds:
			return "ObjectID";
	}
	return "HiPR";
}

static bool create_pipeline_for_mode(ui::RenderDebugViewMode mode, VkPipeline& out_pipeline,
                                     size_t* out_shader_size_bytes = nullptr) {
	std::vector<uint32_t> spirv;
	const std::string     shader_path =
	    std::string(SHADERS_DIR) + "/" + render_mode_shader_filename(mode);
	try {
		spirv = shader::slang::compileSlangShaderOrThrow(shader_path, "main", {VENDORS_DIR},
		                                                 "spirv_1_4");
	} catch (const std::exception& e) {
		std::cerr << "[PIPELINE] " << render_mode_label(mode) << " compile failed: " << e.what()
		          << "\n";
		return false;
	}

	if (out_shader_size_bytes != nullptr) {
		*out_shader_size_bytes = spirv.size() * sizeof(uint32_t);
	}

	VkShaderModuleCreateInfo mci{};
	mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	mci.codeSize = spirv.size() * sizeof(uint32_t);
	mci.pCode    = spirv.data();

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &shader_module) != VK_SUCCESS) {
		std::cerr << "[PIPELINE] " << render_mode_label(mode) << " vkCreateShaderModule failed\n";
		return false;
	}

	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = shader_module;
	stage.pName  = "main";

	VkComputePipelineCreateInfo pci{};
	pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pci.stage  = stage;
	pci.layout = compute_ctx.pipeline_layout;

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
	                             &new_pipeline) != VK_SUCCESS) {
		vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);
		std::cerr << "[PIPELINE] " << render_mode_label(mode)
		          << " vkCreateComputePipelines failed\n";
		return false;
	}

	vkDestroyShaderModule(vulkan_ctx.device, shader_module, nullptr);
	out_pipeline = new_pipeline;
	return true;
}

static bool ensure_pipeline_for_mode(ui::RenderDebugViewMode mode) {
	const size_t index = render_mode_index(mode);
	if (compute_ctx.compiled[index] && compute_ctx.pipelines[index] != VK_NULL_HANDLE) {
		return true;
	}

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	size_t     shader_size  = 0;
	if (!create_pipeline_for_mode(mode, new_pipeline, &shader_size)) {
		return false;
	}

	compute_ctx.pipelines[index] = new_pipeline;
	compute_ctx.compiled[index]  = true;
	std::cout << "[PIPELINE] Built " << render_mode_label(mode) << " ("
	          << static_cast<unsigned long long>(shader_size) << " bytes SPIR-V)\n";
	return true;
}

static bool build_all_mode_pipelines() {
	for (ui::RenderDebugViewMode mode : kRenderModes) {
		if (!ensure_pipeline_for_mode(mode)) {
			return false;
		}
	}
	return true;
}

static bool rebuild_pipeline(ui::RenderDebugViewMode mode) {
	const size_t index = render_mode_index(mode);

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	size_t     shader_size  = 0;
	if (!create_pipeline_for_mode(mode, new_pipeline, &shader_size)) {
		return false;
	}

	vkDeviceWaitIdle(vulkan_ctx.device);
	if (compute_ctx.pipelines[index] != VK_NULL_HANDLE) {
		vkDestroyPipeline(vulkan_ctx.device, compute_ctx.pipelines[index], nullptr);
	}
	compute_ctx.pipelines[index] = new_pipeline;
	compute_ctx.compiled[index]  = true;

	std::cout << "[SHADER RELOAD] Rebuilt " << render_mode_label(mode) << " ("
	          << static_cast<unsigned long long>(shader_size) << " bytes SPIR-V)\n";
	return true;
}

// ============================================================
// === Descriptor layout helper
// ============================================================
static VkDescriptorSetLayoutBinding make_binding(uint32_t binding, VkDescriptorType type,
                                                 uint32_t count = 1) {
	VkDescriptorSetLayoutBinding b{};
	b.binding         = binding;
	b.descriptorType  = type;
	b.descriptorCount = count;
	b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	return b;
}

// ============================================================
// === Acceleration structure helpers
// ============================================================
static constexpr VkBuildAccelerationStructureFlagsKHR AS_BUILD_FLAGS =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
static constexpr VkBufferUsageFlags AS_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
static constexpr VkBufferUsageFlags AS_INPUT_BUFFER_USAGE =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
static constexpr VkBufferUsageFlags SCRATCH_BUFFER_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

static BLAS build_blas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                       const void* verts, uint32_t vc, uint32_t vstride, const void* idxs,
                       uint32_t ic) {
	VmaAllocation va, ia;
	VkBuffer      vb =
	    create_and_upload_buffer(alloc, (VkDeviceSize) vstride * vc, verts,
	                             AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, va);
	VkBuffer ib =
	    create_and_upload_buffer(alloc, (VkDeviceSize) sizeof(uint32_t) * ic, idxs,
	                             AS_INPUT_BUFFER_USAGE | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ia);
	VkAccelerationStructureGeometryTrianglesDataKHR tri{};
	tri.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	tri.vertexData   = {.deviceAddress = get_bda(dev, vb)};
	tri.vertexStride = vstride;
	tri.maxVertex    = vc - 1;
	tri.indexType    = VK_INDEX_TYPE_UINT32;
	tri.indexData    = {.deviceAddress = get_bda(dev, ib)};
	VkAccelerationStructureGeometryKHR geom{};
	geom.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geom.geometry.triangles = tri;
	geom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
	const uint32_t                              tc = ic / 3;
	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build.flags         = AS_BUILD_FLAGS;
	build.geometryCount = 1;
	build.pGeometries   = &geom;
	VkAccelerationStructureBuildSizesInfoKHR si{};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build, &tc, &si);
	BLAS blas;
	blas.buffer =
	    create_gpu_buffer(alloc, si.accelerationStructureSize, AS_BUFFER_USAGE, blas.buffer_alloc);
	VmaAllocation sca;
	VkBuffer      scb = create_gpu_buffer(alloc, si.buildScratchSize, SCRATCH_BUFFER_USAGE, sca,
	                                      vulkan_ctx.scratch_alignment);
	VkAccelerationStructureCreateInfoKHR aci{};
	aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	aci.buffer = blas.buffer;
	aci.size   = si.accelerationStructureSize;
	aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	if (vkCreateAccelerationStructureKHR(dev, &aci, nullptr, &blas.handle) != VK_SUCCESS)
		throw std::runtime_error("failed to create BLAS");
	build.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.dstAccelerationStructure  = blas.handle;
	build.scratchData.deviceAddress = get_bda(dev, scb);
	VkAccelerationStructureBuildRangeInfoKHR ri{};
	ri.primitiveCount                                   = tc;
	const VkAccelerationStructureBuildRangeInfoKHR* pri = &ri;
	VkCommandBuffer                                 cmd = begin_one_time_cmd(dev, pool);
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pri);
	end_one_time_cmd(dev, pool, q, cmd);
	VkAccelerationStructureDeviceAddressInfoKHR dai{};
	dai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	dai.accelerationStructure = blas.handle;
	blas.device_address       = vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);
	vmaDestroyBuffer(alloc, scb, sca);
	vmaDestroyBuffer(alloc, vb, va);
	vmaDestroyBuffer(alloc, ib, ia);
	return blas;
}

static VkTransformMatrixKHR glm_to_vk_transform(const glm::mat4& m) {
	// Vulkan expects a 3x4 row-major transform matrix.
	// GLM is column-major, so we write rows explicitly.
	VkTransformMatrixKHR out{};
	out.matrix[0][0] = m[0][0];
	out.matrix[0][1] = m[1][0];
	out.matrix[0][2] = m[2][0];
	out.matrix[0][3] = m[3][0];

	out.matrix[1][0] = m[0][1];
	out.matrix[1][1] = m[1][1];
	out.matrix[1][2] = m[2][1];
	out.matrix[1][3] = m[3][1];

	out.matrix[2][0] = m[0][2];
	out.matrix[2][1] = m[1][2];
	out.matrix[2][2] = m[2][2];
	out.matrix[2][3] = m[3][2];
	return out;
}

static void destroy_tlas_resources(VmaAllocator alloc, VkDevice dev) {
	if (as_ctx.tlas != VK_NULL_HANDLE) {
		vkDestroyAccelerationStructureKHR(dev, as_ctx.tlas, nullptr);
		as_ctx.tlas = VK_NULL_HANDLE;
	}
	if (as_ctx.tlas_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(alloc, as_ctx.tlas_buffer, as_ctx.tlas_buffer_alloc);
		as_ctx.tlas_buffer       = VK_NULL_HANDLE;
		as_ctx.tlas_buffer_alloc = VK_NULL_HANDLE;
	}
	if (as_ctx.instance_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(alloc, as_ctx.instance_buffer, as_ctx.instance_buffer_alloc);
		as_ctx.instance_buffer       = VK_NULL_HANDLE;
		as_ctx.instance_buffer_alloc = VK_NULL_HANDLE;
		as_ctx.instance_mapped       = nullptr;
	}
	if (as_ctx.scratch_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(alloc, as_ctx.scratch_buffer, as_ctx.scratch_buffer_alloc);
		as_ctx.scratch_buffer       = VK_NULL_HANDLE;
		as_ctx.scratch_buffer_alloc = VK_NULL_HANDLE;
		as_ctx.scratch_size         = 0;
	}
	as_ctx.instance_count = 0;
	as_ctx.tlas_address   = 0;
}

static void fill_tlas_instances(const std::vector<BLAS>&                  blases,
                                const std::vector<std::unique_ptr<Mesh>>& meshes,
                                VkAccelerationStructureInstanceKHR*       instances) {
	if (blases.size() != meshes.size()) {
		throw std::runtime_error("fill_tlas_instances: blases.size() != meshes.size()");
	}

	for (uint32_t i = 0; i < static_cast<uint32_t>(blases.size()); ++i) {
		VkAccelerationStructureInstanceKHR inst{};
		inst.transform           = glm_to_vk_transform(meshes[i]->m_transform.m_transform);
		inst.instanceCustomIndex = i;
		inst.mask                = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;
		inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		inst.accelerationStructureReference = blases[i].device_address;
		instances[i]                        = inst;
	}
}

static void upload_tlas_instances(VmaAllocator alloc, const std::vector<BLAS>& blases,
                                  const std::vector<std::unique_ptr<Mesh>>& meshes) {
	if (as_ctx.instance_mapped == nullptr || blases.empty()) {
		return;
	}

	fill_tlas_instances(
	    blases, meshes,
	    reinterpret_cast<VkAccelerationStructureInstanceKHR*>(as_ctx.instance_mapped));
	vmaFlushAllocation(alloc, as_ctx.instance_buffer_alloc, 0,
	                   sizeof(VkAccelerationStructureInstanceKHR) *
	                       static_cast<VkDeviceSize>(blases.size()));
}

static void record_tlas_build(VkDevice dev, VkCommandBuffer cmd, bool update_mode) {
	if (as_ctx.tlas == VK_NULL_HANDLE || as_ctx.instance_count == 0) {
		return;
	}

	VkAccelerationStructureGeometryInstancesDataKHR instances{};
	instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	instances.data.deviceAddress = get_bda(dev, as_ctx.instance_buffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.geometry.instances = instances;

	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags = AS_BUILD_FLAGS | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
	build.mode  = update_mode ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
	                            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.srcAccelerationStructure  = update_mode ? as_ctx.tlas : VK_NULL_HANDLE;
	build.dstAccelerationStructure  = as_ctx.tlas;
	build.geometryCount             = 1;
	build.pGeometries               = &geometry;
	build.scratchData.deviceAddress = get_bda(dev, as_ctx.scratch_buffer);

	VkAccelerationStructureBuildRangeInfoKHR range{};
	range.primitiveCount                                   = as_ctx.instance_count;
	const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &ranges);

	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
	                     nullptr);
}

static void build_tlas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                       const std::vector<BLAS>&                  blases,
                       const std::vector<std::unique_ptr<Mesh>>& meshes) {
	if (blases.size() != meshes.size()) {
		throw std::runtime_error("build_tlas: blases.size() != meshes.size()");
	}
	if (blases.empty()) {
		destroy_tlas_resources(alloc, dev);
		return;
	}

	const uint32_t instance_count = static_cast<uint32_t>(blases.size());
	if (as_ctx.tlas != VK_NULL_HANDLE) {
		destroy_tlas_resources(alloc, dev);
	}

	VkBufferCreateInfo instance_buffer_info{};
	instance_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	instance_buffer_info.size =
	    sizeof(VkAccelerationStructureInstanceKHR) * static_cast<VkDeviceSize>(instance_count);
	instance_buffer_info.usage = AS_INPUT_BUFFER_USAGE;

	VmaAllocationCreateInfo instance_alloc_info{};
	instance_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	instance_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo instance_info{};
	if (vmaCreateBuffer(alloc, &instance_buffer_info, &instance_alloc_info, &as_ctx.instance_buffer,
	                    &as_ctx.instance_buffer_alloc, &instance_info) != VK_SUCCESS) {
		throw std::runtime_error("failed to create TLAS instance buffer");
	}
	as_ctx.instance_mapped = instance_info.pMappedData;
	as_ctx.instance_count  = instance_count;
	upload_tlas_instances(alloc, blases, meshes);

	VkAccelerationStructureGeometryInstancesDataKHR instances{};
	instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	instances.data.deviceAddress = get_bda(dev, as_ctx.instance_buffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.geometry.instances = instances;

	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags         = AS_BUILD_FLAGS | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
	build.geometryCount = 1;
	build.pGeometries   = &geometry;

	VkAccelerationStructureBuildSizesInfoKHR sizes{};
	sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build, &instance_count, &sizes);

	as_ctx.tlas_buffer  = create_gpu_buffer(alloc, sizes.accelerationStructureSize, AS_BUFFER_USAGE,
	                                        as_ctx.tlas_buffer_alloc);
	as_ctx.scratch_size = std::max(sizes.buildScratchSize, sizes.updateScratchSize);
	as_ctx.scratch_buffer =
	    create_gpu_buffer(alloc, as_ctx.scratch_size, SCRATCH_BUFFER_USAGE,
	                      as_ctx.scratch_buffer_alloc, vulkan_ctx.scratch_alignment);

	VkAccelerationStructureCreateInfoKHR create_info{};
	create_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	create_info.buffer = as_ctx.tlas_buffer;
	create_info.size   = sizes.accelerationStructureSize;
	create_info.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	if (vkCreateAccelerationStructureKHR(dev, &create_info, nullptr, &as_ctx.tlas) != VK_SUCCESS) {
		throw std::runtime_error("failed to create TLAS");
	}

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	record_tlas_build(dev, cmd, false);
	end_one_time_cmd(dev, pool, q, cmd);

	VkAccelerationStructureDeviceAddressInfoKHR address_info{};
	address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	address_info.accelerationStructure = as_ctx.tlas;
	as_ctx.tlas_address = vkGetAccelerationStructureDeviceAddressKHR(dev, &address_info);
}

Runtime::Runtime(const std::string& scene_argument) {
	// ==============================
	// === 0. Scene setup
	// ==============================
	m_scene                      = std::make_unique<Scene>();
	m_scene->m_camera            = Camera(glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.f, 0.f, 0.f),
	                                      glm::vec3(0.f, 1.f, 0.f), 60.f, 0.1f, 10000.f);
	const std::string scene_path = app::scene::resolveScenePathOrThrow(scene_argument);
	std::cout << "[INFO] Loading scene: " << scene_path << "\n";
	m_scene->load_gltf(scene_path);
	if (m_scene->m_meshes.empty()) {
		throw std::runtime_error("failed to load scene: " + scene_path);
	}
	water_surface_render_ctx = simulation::buildWaterSurfacePlacement(m_scene.get());
	simulation::applyDedicatedWaterMaterial(m_scene.get(), water_surface_render_ctx);
	initializeMeshTransformMetadata(m_scene.get());
	addFloatingMeshesFromResources(m_scene.get());
	ui::rebuildObjectIdMap(m_scene.get());

	// ========================================
	// === I. Vulkan function pointers
	// ========================================
	if (volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("failed to initialize volk");
	std::cout << "[INFO] Volk initialized\n";

	// =========================
	// === II. Window
	// =========================
	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});
	std::cout << "[INFO] Window created\n";
	glfwPollEvents();

	// ==================================
	// === III. Vulkan instance/device
	// ==================================
	{
		vkb::InstanceBuilder b;
		auto                 r = b.set_app_name("tsunami")
		                             .request_validation_layers()
		                             .use_default_debug_messenger()
		                             .require_api_version(1, 3, 0)
		                             .build();
		if (!r)
			throw std::runtime_error("failed to create Vulkan instance");
		vulkan_ctx.instance = r.value();
		volkLoadInstance(vulkan_ctx.instance.instance);
		std::cout << "[INFO] Vulkan instance created\n";
	}
	if (glfwCreateWindowSurface(vulkan_ctx.instance, m_window->handle(), nullptr,
	                            &vulkan_ctx.surface) != VK_SUCCESS)
		throw std::runtime_error("failed to create window surface");
	{
		std::vector<const char*> exts = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		                                 VK_KHR_RAY_QUERY_EXTENSION_NAME,
		                                 VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
		auto                     r    = vkb::PhysicalDeviceSelector(vulkan_ctx.instance)
		                                    .set_surface(vulkan_ctx.surface)
		                                    .set_minimum_version(1, 3)
		                                    .add_required_extensions(exts)
		                                    .select();
		if (!r)
			throw std::runtime_error("failed to select physical device");
		vulkan_ctx.phys_device = r.value();
		VkPhysicalDeviceProperties p{};
		vkGetPhysicalDeviceProperties(vulkan_ctx.phys_device.physical_device, &p);
		vulkan_ctx.timestamp_period_ns = p.limits.timestampPeriod;
		std::cout << "[INFO] GPU: " << p.deviceName << "\n";
	}
	{
		VkPhysicalDeviceAccelerationStructureFeaturesKHR af{};
		af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		af.accelerationStructure = VK_TRUE;
		VkPhysicalDeviceRayQueryFeaturesKHR rf{};
		rf.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
		rf.rayQuery = VK_TRUE;
		VkPhysicalDeviceBufferDeviceAddressFeatures bf{};
		bf.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		bf.bufferDeviceAddress = VK_TRUE;
		auto r                 = vkb::DeviceBuilder{vulkan_ctx.phys_device}
		                             .add_pNext(&af)
		                             .add_pNext(&rf)
		                             .add_pNext(&bf)
		                             .build();
		if (!r)
			throw std::runtime_error("failed to create logical device");
		vulkan_ctx.log_device = r.value();
		vulkan_ctx.device     = vulkan_ctx.log_device.device;
		volkLoadDevice(vulkan_ctx.device);
		std::cout << "[INFO] Logical device created\n";
	}
	{
		VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{};
		asp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
		VkPhysicalDeviceProperties2 p2{};
		p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		p2.pNext = &asp;
		vkGetPhysicalDeviceProperties2(vulkan_ctx.phys_device.physical_device, &p2);
		vulkan_ctx.scratch_alignment = asp.minAccelerationStructureScratchOffsetAlignment;
	}
	{
		auto qr = vulkan_ctx.log_device.get_queue(vkb::QueueType::graphics);
		if (!qr)
			throw std::runtime_error("no graphics queue");
		vulkan_ctx.graphics_queue = qr.value();
		auto fr                   = vulkan_ctx.log_device.get_queue_index(vkb::QueueType::graphics);
		if (!fr)
			throw std::runtime_error("no graphics queue family");
		vulkan_ctx.graphics_queue_family = fr.value();
		uint32_t queue_family_count      = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(vulkan_ctx.phys_device.physical_device,
		                                         &queue_family_count, nullptr);
		std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(vulkan_ctx.phys_device.physical_device,
		                                         &queue_family_count,
		                                         queue_family_properties.data());
		vulkan_ctx.supports_timestamps =
		    vulkan_ctx.graphics_queue_family < queue_family_properties.size() &&
		    queue_family_properties[vulkan_ctx.graphics_queue_family].timestampValidBits > 0 &&
		    vulkan_ctx.timestamp_period_ns > 0.0f;
	}

	// ==========================
	// === IV. VMA
	// ==========================
	{
		VmaVulkanFunctions vf{};
		vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
		vf.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
		VmaAllocatorCreateInfo ai{};
		ai.instance         = vulkan_ctx.instance.instance;
		ai.physicalDevice   = vulkan_ctx.phys_device.physical_device;
		ai.device           = vulkan_ctx.device;
		ai.vulkanApiVersion = VK_API_VERSION_1_3;
		ai.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
		ai.pVulkanFunctions = &vf;
		if (vmaCreateAllocator(&ai, &render_target_ctx.allocator) != VK_SUCCESS)
			throw std::runtime_error("failed to create VMA allocator");
		std::cout << "[INFO] VMA allocator created\n";
	}
	VmaAllocator allocator = render_target_ctx.allocator;

	// ===================================
	// === V. Swapchain
	// ===================================
	createSwapchainResources();

	// ========================================
	// === VI. Render target images
	// ========================================
	{
		if (render_target_ctx.extent.width == 0 || render_target_ctx.extent.height == 0) {
			render_target_ctx.extent = swapchain_ctx.extent;
		}
		VkExtent3D ext = {render_target_ctx.extent.width, render_target_ctx.extent.height, 1};
		auto       make_storage_image = [&](VkImage& img, VmaAllocation& a, VkFormat format,
		                                    VkImageUsageFlags extra) {
			VkImageCreateInfo ii{};
			ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ii.imageType     = VK_IMAGE_TYPE_2D;
			ii.format        = format;
			ii.extent        = ext;
			ii.mipLevels     = 1;
			ii.arrayLayers   = 1;
			ii.samples       = VK_SAMPLE_COUNT_1_BIT;
			ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | extra;
			ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VmaAllocationCreateInfo ai{};
			ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			if (vmaCreateImage(allocator, &ii, &ai, &img, &a, nullptr) != VK_SUCCESS)
				throw std::runtime_error("failed to create storage image");
		};
		auto make_dummy = [&](VkImageType itype, VkExtent3D e, VkImage& img, VmaAllocation& a) {
			VkImageCreateInfo ii{};
			ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			ii.imageType     = itype;
			ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
			ii.extent        = e;
			ii.mipLevels     = 1;
			ii.arrayLayers   = 1;
			ii.samples       = VK_SAMPLE_COUNT_1_BIT;
			ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
			ii.usage         = VK_IMAGE_USAGE_SAMPLED_BIT;
			ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VmaAllocationCreateInfo ai{};
			ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
			if (vmaCreateImage(allocator, &ii, &ai, &img, &a, nullptr) != VK_SUCCESS)
				throw std::runtime_error("failed to create dummy image");
		};
		make_storage_image(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
		                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		make_storage_image(render_target_ctx.object_id_image,
		                   render_target_ctx.object_id_image_alloc, VK_FORMAT_R32_SINT,
		                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		make_storage_image(render_target_ctx.object_id_history_image,
		                   render_target_ctx.object_id_history_image_alloc, VK_FORMAT_R32_SINT,
		                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		make_storage_image(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
		                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
		make_storage_image(render_target_ctx.accum_history_image,
		                   render_target_ctx.accum_history_image_alloc, VK_FORMAT_R8G8B8A8_UNORM,
		                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		render_target_ctx.storage_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.object_id_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image,
		                      VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.object_id_history_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_history_image,
		                      VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.accum_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.accum_history_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.accum_history_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		make_dummy(VK_IMAGE_TYPE_2D, {1, 1, 1}, render_target_ctx.dummy_image_2d,
		           render_target_ctx.dummy_image_2d_alloc);
		render_target_ctx.dummy_image_2d_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.dummy_image_2d,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		make_dummy(VK_IMAGE_TYPE_3D, {1, 1, 1}, render_target_ctx.dummy_image_3d,
		           render_target_ctx.dummy_image_3d_alloc);
		render_target_ctx.dummy_image_3d_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.dummy_image_3d,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_3D);
		std::cout << "[INFO] Render targets " << ext.width << "x" << ext.height << "\n";
	}

	// =============================================
	// === VI.5  Command pool (needed early)
	// =============================================
	{
		VkCommandPoolCreateInfo pi{};
		pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pi.queueFamilyIndex = vulkan_ctx.graphics_queue_family;
		pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(vulkan_ctx.device, &pi, nullptr, &command_ctx.command_pool) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create command pool");
		std::cout << "[INFO] Command pool created\n";
	}
	// Transition dummy images → SHADER_READ_ONLY once
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		for (VkImage img : {render_target_ctx.dummy_image_2d, render_target_ctx.dummy_image_3d})
			transition_layout(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED,
			                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
			                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
	}

	// =================================
	// === VII. Upload scene buffers
	// =================================
	{
		constexpr VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		VmaAllocationCreateInfo cam_alloc_info{};
		cam_alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		cam_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBufferCreateInfo cam_buf_info{};
		cam_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		cam_buf_info.size  = sizeof(GPUCamera) * 2u;
		cam_buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationInfo cam_info;
		vmaCreateBuffer(allocator, &cam_buf_info, &cam_alloc_info, &scene_ctx.camera_buffer,
		                &scene_ctx.camera_alloc, &cam_info);
		scene_ctx.camera_mapped      = cam_info.pMappedData;
		GPUCamera initial_cam        = m_scene->m_camera.pack();
		GPUCamera initial_cameras[2] = {initial_cam, initial_cam};
		memcpy(scene_ctx.camera_mapped, initial_cameras, sizeof(initial_cameras));
		std::vector<GPUMaterial> gms;

		for (auto& mesh : m_scene->m_meshes)
			gms.push_back(mesh->m_material->pack());
		scene_ctx.material_count  = (uint32_t) gms.size();
		scene_ctx.material_buffer = create_and_upload_buffer(
		    allocator, sizeof(GPUMaterial) * gms.size(), gms.data(), SB, scene_ctx.material_alloc);
		VmaAllocationInfo material_info{};
		vmaGetAllocationInfo(allocator, scene_ctx.material_alloc, &material_info);
		scene_ctx.material_mapped = material_info.pMappedData;
		if (scene_ctx.material_mapped == nullptr &&
		    vmaMapMemory(allocator, scene_ctx.material_alloc, &scene_ctx.material_mapped) !=
		        VK_SUCCESS) {
			throw std::runtime_error("failed to map material buffer");
		}
		std::cout << "[INFO] Uploaded " << scene_ctx.material_count << " materials\n";
	}

	// ======================================================
	// === VIII. BLAS per mesh + TLAS
	// ======================================================
	std::vector<GPUVertex> all_verts;
	std::vector<uint32_t>  all_idxs;
	std::vector<GPUMesh>   gpu_meshes;

	for (int mi = 0; mi < (int) m_scene->m_meshes.size(); ++mi) {
		auto& mesh = m_scene->m_meshes[mi];
		if (!mesh->m_material) {
			std::cerr << "[WARN] mesh " << mi << " no material\n";
			continue;
		}
		int voff = (int) all_verts.size(), ioff = (int) all_idxs.size();
		all_verts.insert(all_verts.end(), mesh->gpuVertices.begin(), mesh->gpuVertices.end());
		all_idxs.insert(all_idxs.end(), mesh->gpuIndices.begin(), mesh->gpuIndices.end());
		BLAS    blas     = build_blas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		                              vulkan_ctx.graphics_queue, mesh->gpuVertices.data(),
		                              (uint32_t) mesh->gpuVertices.size(), sizeof(GPUVertex),
		                              mesh->gpuIndices.data(), (uint32_t) mesh->gpuIndices.size());
		GPUMesh gm       = mesh->pack(mi, voff, ioff);
		gm.blasHandle_lo = (uint32_t) (blas.device_address & 0xFFFFFFFF);
		gm.blasHandle_hi = (uint32_t) (blas.device_address >> 32);
		as_ctx.blases.push_back(std::move(blas));
		gpu_meshes.push_back(gm);
		std::cout << "[INFO] BLAS " << mi << ": " << mesh->gpuIndices.size() / 3 << " tris\n";
	}
	if (!as_ctx.blases.empty()) {
		build_tlas(allocator, vulkan_ctx.device, command_ctx.command_pool,
		           vulkan_ctx.graphics_queue, as_ctx.blases, m_scene->m_meshes);
		std::cout << "[INFO] TLAS: " << as_ctx.blases.size() << " instances\n";
	}
	if (!gpu_meshes.empty()) {
		constexpr VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		scene_ctx.mesh_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUMesh) * gpu_meshes.size(),
		                             gpu_meshes.data(), SB, scene_ctx.mesh_alloc);
		scene_ctx.vertex_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUVertex) * all_verts.size(),
		                             all_verts.data(), SB, scene_ctx.vertex_alloc);
		scene_ctx.index_buffer =
		    create_and_upload_buffer(allocator, sizeof(uint32_t) * all_idxs.size(), all_idxs.data(),
		                             SB, scene_ctx.index_alloc);
		VmaAllocationInfo mesh_info{};
		vmaGetAllocationInfo(allocator, scene_ctx.mesh_alloc, &mesh_info);
		scene_ctx.mesh_mapped = mesh_info.pMappedData;
		if (scene_ctx.mesh_mapped == nullptr &&
		    vmaMapMemory(allocator, scene_ctx.mesh_alloc, &scene_ctx.mesh_mapped) != VK_SUCCESS) {
			throw std::runtime_error("failed to map mesh buffer");
		}
		scene_ctx.mesh_count = (uint32_t) gpu_meshes.size();
		std::cout << "[INFO] Mesh buffers: " << gpu_meshes.size() << " meshes, " << all_verts.size()
		          << " verts, " << all_idxs.size() << " idxs\n";
	}

	// ======================================================
	// === VIII.6  HiPR buffers (per-object influence stats)
	// ======================================================
	{
		constexpr VkBufferUsageFlags SB           = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		const uint32_t               object_count = std::max(scene_ctx.mesh_count, 1u);
		scene_ctx.hipr_top_k                      = std::min(HIPR_TOP_K, object_count);

		std::vector<uint32_t> zero_counts(object_count, 0u);
		std::vector<float>    zero_scores(object_count, 0.0f);
		std::vector<int32_t>  empty_order(scene_ctx.hipr_top_k, -1);

		scene_ctx.hipr_visible_count_buffer =
		    create_and_upload_buffer(allocator, sizeof(uint32_t) * object_count, zero_counts.data(),
		                             SB, scene_ctx.hipr_visible_count_alloc);
		scene_ctx.hipr_secondary_count_buffer =
		    create_and_upload_buffer(allocator, sizeof(uint32_t) * object_count, zero_counts.data(),
		                             SB, scene_ctx.hipr_secondary_count_alloc);
		scene_ctx.hipr_shadow_count_buffer =
		    create_and_upload_buffer(allocator, sizeof(uint32_t) * object_count, zero_counts.data(),
		                             SB, scene_ctx.hipr_shadow_count_alloc);
		scene_ctx.hipr_score_buffer =
		    create_and_upload_buffer(allocator, sizeof(float) * object_count, zero_scores.data(),
		                             SB, scene_ctx.hipr_score_alloc);
		scene_ctx.hipr_order_buffer =
		    create_and_upload_buffer(allocator, sizeof(int32_t) * scene_ctx.hipr_top_k,
		                             empty_order.data(), SB, scene_ctx.hipr_order_alloc);
	}

	{
		vulkan::waterbridge::WaterSurfaceParamsGpu initial_water_params{};
		render_target_ctx.water_surface_params_buffer = create_and_upload_buffer(
		    allocator, sizeof(initial_water_params), &initial_water_params,
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, render_target_ctx.water_surface_params_alloc);
		VmaAllocationInfo water_params_info{};
		vmaGetAllocationInfo(allocator, render_target_ctx.water_surface_params_alloc,
		                     &water_params_info);
		render_target_ctx.water_surface_params_mapped = water_params_info.pMappedData;
		if (render_target_ctx.water_surface_params_mapped == nullptr &&
		    vmaMapMemory(allocator, render_target_ctx.water_surface_params_alloc,
		                 &render_target_ctx.water_surface_params_mapped) != VK_SUCCESS) {
			throw std::runtime_error("failed to map water surface params buffer");
		}
		vulkan::waterbridge::updateWaterSurfaceParamsBuffer(
		    makeWaterSurfaceParamsBufferContext(), water_surface_render_ctx, m_scene.get(),
		    m_water_surface.get(), vulkan::floating::firstFloatingObjectId(floating_mesh_groups));
	}

	// ===================================================
	// === VIII.5  Upload OpenPBR LUTs + scene textures
	// ===================================================
	upload_openpbr_luts(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                    vulkan_ctx.graphics_queue);
	upload_material_textures(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                         vulkan_ctx.graphics_queue, m_scene->m_textures);

	// ==============================================
	// === IX. Descriptor layout, pool, sets
	// ==============================================
	// Binding map:
	//   0  = output image            STORAGE_IMAGE
	//   1  = accum image             STORAGE_IMAGE
	//   2  = camera buffer           STORAGE_BUFFER
	//   3  = lut sampler             SAMPLER
	//   4  = materials buffer        STORAGE_BUFFER
	//   5  = TLAS                    ACCELERATION_STRUCTURE
	//   6  = meshes buffer           STORAGE_BUFFER
	//   7  = vertices buffer         STORAGE_BUFFER
	//   8  = indices buffer          STORAGE_BUFFER
	//   9  = lut_textures_2d[8]      SAMPLED_IMAGE × NUM_LUTS
	//  10  = lut_textures_3d[8]      SAMPLED_IMAGE × NUM_LUTS
	//  11  = material_textures[256]  SAMPLED_IMAGE × MAX_MATERIAL_TEXTURES
	//  14  = hipr visible counts     STORAGE_BUFFER
	//  15  = hipr secondary counts   STORAGE_BUFFER
	//  16  = hipr shadow counts      STORAGE_BUFFER
	//  17  = hipr scores             STORAGE_BUFFER
	//  18  = hipr order              STORAGE_BUFFER
	//  19  = water params            STORAGE_BUFFER
	//  20  = water height            STORAGE_IMAGE
	//  21  = water previous height   STORAGE_IMAGE
	//  22  = accum history           STORAGE_IMAGE
	//  23  = object-id history       STORAGE_IMAGE
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
		    make_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(3, VK_DESCRIPTOR_TYPE_SAMPLER),
		    make_binding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NUM_LUTS),
		    make_binding(10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NUM_LUTS),
		    make_binding(11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_MATERIAL_TEXTURES),
		    make_binding(12, VK_DESCRIPTOR_TYPE_SAMPLER),
		    make_binding(13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
		    make_binding(20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(21, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		    make_binding(23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			bindings.push_back(make_binding(5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR));
		VkDescriptorSetLayoutCreateInfo dsl{};
		dsl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dsl.bindingCount = (uint32_t) bindings.size();
		dsl.pBindings    = bindings.data();
		if (vkCreateDescriptorSetLayout(vulkan_ctx.device, &dsl, nullptr,
		                                &render_target_ctx.descriptor_set_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout");
		std::cout << "[INFO] Descriptor set layout created\n";
	}
	{
		std::vector<VkDescriptorPoolSize> ps = {
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7},
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11},
		    {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
		    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * NUM_LUTS + MAX_MATERIAL_TEXTURES},
		};
		if (as_ctx.tlas != VK_NULL_HANDLE)
			ps.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
		VkDescriptorPoolCreateInfo pi{};
		pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pi.maxSets       = 1;
		pi.poolSizeCount = (uint32_t) ps.size();
		pi.pPoolSizes    = ps.data();
		if (vkCreateDescriptorPool(vulkan_ctx.device, &pi, nullptr,
		                           &render_target_ctx.descriptor_pool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool");
		VkDescriptorSetAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool     = render_target_ctx.descriptor_pool;
		ai.descriptorSetCount = 1;
		ai.pSetLayouts        = &render_target_ctx.descriptor_set_layout;
		if (vkAllocateDescriptorSets(vulkan_ctx.device, &ai, &render_target_ctx.descriptor_set) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to allocate descriptor set");
		std::cout << "[INFO] Descriptor pool+set allocated\n";
	}
	// Clamp sampler for LUTs.
	{
		VkSamplerCreateInfo si{};
		si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		si.magFilter    = VK_FILTER_LINEAR;
		si.minFilter    = VK_FILTER_LINEAR;
		si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		si.maxLod                                           = VK_LOD_CLAMP_NONE;
		if (vkCreateSampler(vulkan_ctx.device, &si, nullptr, &render_target_ctx.lut_sampler) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create LUT sampler");
	}
	// Repeat sampler for material textures. The glTF asset relies on tiled UVs
	// outside [0, 1], and glTF defaults wrap those textures with REPEAT.
	{
		VkSamplerCreateInfo si{};
		si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		si.magFilter    = VK_FILTER_LINEAR;
		si.minFilter    = VK_FILTER_LINEAR;
		si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		si.maxLod                                           = VK_LOD_CLAMP_NONE;
		if (vkCreateSampler(vulkan_ctx.device, &si, nullptr, &render_target_ctx.material_sampler) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to create material sampler");
	}
	// Write descriptors
	{
		std::vector<VkWriteDescriptorSet> writes;
		// 0 — output image
		VkDescriptorImageInfo oi{};
		oi.imageView   = render_target_ctx.storage_image_view;
		oi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 0, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi});
		// 1 — accum image
		VkDescriptorImageInfo ai2{};
		ai2.imageView   = render_target_ctx.accum_image_view;
		ai2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 1, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ai2});
		// 2 — camera
		VkDescriptorImageInfo object_id_info{};
		object_id_info.imageView   = render_target_ctx.object_id_image_view;
		object_id_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 13, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &object_id_info});
		VkDescriptorImageInfo accum_history_info{};
		accum_history_info.imageView   = render_target_ctx.accum_history_image_view;
		accum_history_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 22, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accum_history_info});
		VkDescriptorImageInfo object_id_history_info{};
		object_id_history_info.imageView   = render_target_ctx.object_id_history_image_view;
		object_id_history_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 23, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &object_id_history_info});
		VkDescriptorBufferInfo cb{scene_ctx.camera_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 2, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &cb});
		// 3 — sampler
		VkDescriptorImageInfo si2{};
		si2.sampler = render_target_ctx.lut_sampler;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
		                  &si2});
		// 4 — materials
		VkDescriptorBufferInfo mb{scene_ctx.material_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 4, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mb});
		// 5 — TLAS
		VkWriteDescriptorSetAccelerationStructureKHR as_ext{};
		VkWriteDescriptorSet                         as_write{};
		if (as_ctx.tlas != VK_NULL_HANDLE) {
			as_ext.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			as_ext.accelerationStructureCount = 1;
			as_ext.pAccelerationStructures    = &as_ctx.tlas;
			as_write                          = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			                                     &as_ext,
			                                     render_target_ctx.descriptor_set,
			                                     5,
			                                     0,
			                                     1,
			                                     VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR};
			writes.push_back(as_write);
		}
		// 6 — meshes
		VkDescriptorBufferInfo mshb{scene_ctx.mesh_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 6, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mshb});
		// 7 — vertices
		VkDescriptorBufferInfo vb{scene_ctx.vertex_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 7, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vb});
		// 8 — indices
		VkDescriptorBufferInfo ib{scene_ctx.index_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 8, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ib});
		// 14-18 — HiPR buffers
		VkDescriptorBufferInfo hipr_visible{scene_ctx.hipr_visible_count_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 14, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hipr_visible});
		VkDescriptorBufferInfo hipr_secondary{scene_ctx.hipr_secondary_count_buffer, 0,
		                                      VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 15, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hipr_secondary});
		VkDescriptorBufferInfo hipr_shadow{scene_ctx.hipr_shadow_count_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 16, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hipr_shadow});
		VkDescriptorBufferInfo hipr_scores{scene_ctx.hipr_score_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 17, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hipr_scores});
		VkDescriptorBufferInfo hipr_order{scene_ctx.hipr_order_buffer, 0, VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 18, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hipr_order});
		VkDescriptorBufferInfo water_params{render_target_ctx.water_surface_params_buffer, 0,
		                                    VK_WHOLE_SIZE};
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 19, 0, 1,
		                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &water_params});
		// 9 — LUT 2D (real or dummy fallback per slot)
		std::vector<VkDescriptorImageInfo> lut2(NUM_LUTS);
		for (uint32_t i = 0; i < NUM_LUTS; ++i) {
			lut2[i].imageView   = (render_target_ctx.lut_textures_2d[i].view != VK_NULL_HANDLE) ?
			                          render_target_ctx.lut_textures_2d[i].view :
			                          render_target_ctx.dummy_image_2d_view;
			lut2[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 9;
			w.dstArrayElement = 0;
			w.descriptorCount = NUM_LUTS;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = lut2.data();
			writes.push_back(w);
		}
		// 10 — LUT 3D
		std::vector<VkDescriptorImageInfo> lut3(NUM_LUTS);
		for (uint32_t i = 0; i < NUM_LUTS; ++i) {
			lut3[i].imageView   = (render_target_ctx.lut_textures_3d[i].view != VK_NULL_HANDLE) ?
			                          render_target_ctx.lut_textures_3d[i].view :
			                          render_target_ctx.dummy_image_3d_view;
			lut3[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 10;
			w.dstArrayElement = 0;
			w.descriptorCount = NUM_LUTS;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = lut3.data();
			writes.push_back(w);
		}
		// 11 — Material textures (pad unused slots with dummy)
		std::vector<VkDescriptorImageInfo> mt(MAX_MATERIAL_TEXTURES);
		for (uint32_t i = 0; i < MAX_MATERIAL_TEXTURES; ++i) {
			bool ok = i < render_target_ctx.mat_views.size() &&
			          render_target_ctx.mat_views[i] != VK_NULL_HANDLE;
			mt[i].imageView =
			    ok ? render_target_ctx.mat_views[i] : render_target_ctx.dummy_image_2d_view;
			mt[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		{
			VkWriteDescriptorSet w{};
			w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.dstSet          = render_target_ctx.descriptor_set;
			w.dstBinding      = 11;
			w.dstArrayElement = 0;
			w.descriptorCount = MAX_MATERIAL_TEXTURES;
			w.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			w.pImageInfo      = mt.data();
			writes.push_back(w);
		}
		VkDescriptorImageInfo material_sampler_info{};
		material_sampler_info.sampler = render_target_ctx.material_sampler;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 12, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
		                  &material_sampler_info});
		vkUpdateDescriptorSets(vulkan_ctx.device, (uint32_t) writes.size(), writes.data(), 0,
		                       nullptr);
		vulkan::waterbridge::updateWaterSurfaceImageDescriptors(
		    vulkan_ctx.device, render_target_ctx.descriptor_set, m_water_surface.get());
		std::cout << "[INFO] Descriptor sets updated\n";
	}

	// ============================================
	// === X. Compute Pipeline
	// ============================================
	{
		VkPushConstantRange pr{};
		pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pr.size       = sizeof(PathTracerPushConstants);
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &render_target_ctx.descriptor_set_layout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pr;
		if (vkCreatePipelineLayout(vulkan_ctx.device, &pli, nullptr,
		                           &compute_ctx.pipeline_layout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout");
		std::cout << "[INFO] Compute pipeline layout created\n";

		if (!build_all_mode_pipelines()) {
			throw std::runtime_error("failed to build one or more render-mode pipelines");
		}
		std::cout << "[INFO] Precompiled all render-mode pipelines\n";
	}

	// ================================================
	// === XI. Main command buffer
	// ================================================
	{
		VkCommandBufferAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ai.commandPool        = command_ctx.command_pool;
		ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(vulkan_ctx.device, &ai, &command_ctx.command_buffer) !=
		    VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffer");
	}
	std::cout << "[INFO] Allocated command buffer\n";

	ui::initializeImGuiContext(m_window->handle());
	ui::initializeImGuiRenderer(makeImGuiRendererInitInfo());
	std::cout << "[INFO] Initialized ImGui overlay\n";

	m_audio_controller = std::make_unique<audio::ReactiveAudioController>();
	m_microphone       = std::make_unique<audio::MicrophoneInput>();
	if (m_microphone->isAvailable()) {
		std::cout << "[INFO] Live microphone capture ready: " << m_microphone->deviceName() << "\n";
	} else {
		std::cout << "[WARN] " << m_microphone->statusMessage()
		          << " Falling back to the demo audience signal.\n";
	}

	// Accum image → GENERAL
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.object_id_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_history_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
		std::cout << "[INFO] Accum image → GENERAL\n";
	}

	// ==========================
	// === XII. Sync objects
	// ==========================
	{
		VkSemaphoreCreateInfo sei{};
		sei.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fi{};
		fi.sType         = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fi.flags         = VK_FENCE_CREATE_SIGNALED_BIT;
		const uint32_t n = (uint32_t) swapchain_ctx.images.size();
		sync_ctx.image_available.resize(n);
		sync_ctx.render_finished.resize(n);
		for (uint32_t i = 0; i < n; ++i) {
			if (vkCreateSemaphore(vulkan_ctx.device, &sei, nullptr, &sync_ctx.image_available[i]) !=
			    VK_SUCCESS)
				throw std::runtime_error("failed to create semaphore");
			if (vkCreateSemaphore(vulkan_ctx.device, &sei, nullptr, &sync_ctx.render_finished[i]) !=
			    VK_SUCCESS)
				throw std::runtime_error("failed to create semaphore");
		}
		if (vkCreateFence(vulkan_ctx.device, &fi, nullptr, &sync_ctx.in_flight) != VK_SUCCESS)
			throw std::runtime_error("failed to create fence");
		std::cout << "[INFO] Sync objects created\n";
	}

	if (vulkan_ctx.supports_timestamps) {
		VkQueryPoolCreateInfo query_pool_info{};
		query_pool_info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		query_pool_info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
		query_pool_info.queryCount = 4;
		if (vkCreateQueryPool(vulkan_ctx.device, &query_pool_info, nullptr,
		                      &gpu_timing_ctx.query_pool) != VK_SUCCESS) {
			std::cout
			    << "[WARN] Failed to create GPU timestamp query pool. Pass timing disabled.\n";
		} else {
			std::cout << "[INFO] GPU timestamp pass timing enabled\n";
		}
	}
}

Runtime::~Runtime() {
	if (vulkan_ctx.device == VK_NULL_HANDLE) {
		m_water_surface.reset();
		m_audio_controller.reset();
		m_microphone.reset();
		return;
	}

	vkDeviceWaitIdle(vulkan_ctx.device);

	m_audio_controller.reset();
	m_microphone.reset();

	ui::shutdownImGui();
	destroySwapchainResources();

	for (size_t pipeline_index = 0; pipeline_index < compute_ctx.pipelines.size();
	     ++pipeline_index) {
		if (compute_ctx.pipelines[pipeline_index] != VK_NULL_HANDLE) {
			vkDestroyPipeline(vulkan_ctx.device, compute_ctx.pipelines[pipeline_index], nullptr);
			compute_ctx.pipelines[pipeline_index] = VK_NULL_HANDLE;
		}
		compute_ctx.compiled[pipeline_index] = false;
	}
	if (compute_ctx.pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(vulkan_ctx.device, compute_ctx.pipeline_layout, nullptr);
		compute_ctx.pipeline_layout = VK_NULL_HANDLE;
	}

	for (VkImageView& view : render_target_ctx.mat_views) {
		if (view != VK_NULL_HANDLE) {
			vkDestroyImageView(vulkan_ctx.device, view, nullptr);
			view = VK_NULL_HANDLE;
		}
	}
	for (size_t i = 0; i < render_target_ctx.mat_images.size(); ++i) {
		if (render_target_ctx.mat_images[i] != VK_NULL_HANDLE &&
		    i < render_target_ctx.mat_allocs.size() &&
		    render_target_ctx.mat_allocs[i] != VK_NULL_HANDLE) {
			vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.mat_images[i],
			                render_target_ctx.mat_allocs[i]);
			render_target_ctx.mat_images[i] = VK_NULL_HANDLE;
			render_target_ctx.mat_allocs[i] = VK_NULL_HANDLE;
		}
	}
	render_target_ctx.mat_views.clear();
	render_target_ctx.mat_images.clear();
	render_target_ctx.mat_allocs.clear();

	for (auto& lut : render_target_ctx.lut_textures_2d) {
		if (lut.view != VK_NULL_HANDLE) {
			vkDestroyImageView(vulkan_ctx.device, lut.view, nullptr);
			lut.view = VK_NULL_HANDLE;
		}
		if (lut.image != VK_NULL_HANDLE && lut.alloc != VK_NULL_HANDLE) {
			vmaDestroyImage(render_target_ctx.allocator, lut.image, lut.alloc);
			lut.image = VK_NULL_HANDLE;
			lut.alloc = VK_NULL_HANDLE;
		}
	}
	for (auto& lut : render_target_ctx.lut_textures_3d) {
		if (lut.view != VK_NULL_HANDLE) {
			vkDestroyImageView(vulkan_ctx.device, lut.view, nullptr);
			lut.view = VK_NULL_HANDLE;
		}
		if (lut.image != VK_NULL_HANDLE && lut.alloc != VK_NULL_HANDLE) {
			vmaDestroyImage(render_target_ctx.allocator, lut.image, lut.alloc);
			lut.image = VK_NULL_HANDLE;
			lut.alloc = VK_NULL_HANDLE;
		}
	}

	if (render_target_ctx.storage_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
		render_target_ctx.storage_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.storage_image != VK_NULL_HANDLE &&
	    render_target_ctx.storage_image_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.storage_image,
		                render_target_ctx.storage_image_alloc);
		render_target_ctx.storage_image       = VK_NULL_HANDLE;
		render_target_ctx.storage_image_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.object_id_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_image_view, nullptr);
		render_target_ctx.object_id_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.object_id_image != VK_NULL_HANDLE &&
	    render_target_ctx.object_id_image_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.object_id_image,
		                render_target_ctx.object_id_image_alloc);
		render_target_ctx.object_id_image       = VK_NULL_HANDLE;
		render_target_ctx.object_id_image_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.object_id_history_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_history_image_view,
		                   nullptr);
		render_target_ctx.object_id_history_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.object_id_history_image != VK_NULL_HANDLE &&
	    render_target_ctx.object_id_history_image_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.object_id_history_image,
		                render_target_ctx.object_id_history_image_alloc);
		render_target_ctx.object_id_history_image       = VK_NULL_HANDLE;
		render_target_ctx.object_id_history_image_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.accum_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
		render_target_ctx.accum_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.accum_image != VK_NULL_HANDLE &&
	    render_target_ctx.accum_image_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.accum_image,
		                render_target_ctx.accum_image_alloc);
		render_target_ctx.accum_image       = VK_NULL_HANDLE;
		render_target_ctx.accum_image_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.accum_history_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_history_image_view, nullptr);
		render_target_ctx.accum_history_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.accum_history_image != VK_NULL_HANDLE &&
	    render_target_ctx.accum_history_image_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.accum_history_image,
		                render_target_ctx.accum_history_image_alloc);
		render_target_ctx.accum_history_image       = VK_NULL_HANDLE;
		render_target_ctx.accum_history_image_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.water_surface_params_buffer != VK_NULL_HANDLE &&
	    render_target_ctx.water_surface_params_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, render_target_ctx.water_surface_params_buffer,
		                 render_target_ctx.water_surface_params_alloc);
		render_target_ctx.water_surface_params_buffer = VK_NULL_HANDLE;
		render_target_ctx.water_surface_params_alloc  = VK_NULL_HANDLE;
		render_target_ctx.water_surface_params_mapped = nullptr;
	}
	if (render_target_ctx.dummy_image_2d_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.dummy_image_2d_view, nullptr);
		render_target_ctx.dummy_image_2d_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.dummy_image_2d != VK_NULL_HANDLE &&
	    render_target_ctx.dummy_image_2d_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.dummy_image_2d,
		                render_target_ctx.dummy_image_2d_alloc);
		render_target_ctx.dummy_image_2d       = VK_NULL_HANDLE;
		render_target_ctx.dummy_image_2d_alloc = VK_NULL_HANDLE;
	}
	if (render_target_ctx.dummy_image_3d_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.dummy_image_3d_view, nullptr);
		render_target_ctx.dummy_image_3d_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.dummy_image_3d != VK_NULL_HANDLE &&
	    render_target_ctx.dummy_image_3d_alloc != VK_NULL_HANDLE) {
		vmaDestroyImage(render_target_ctx.allocator, render_target_ctx.dummy_image_3d,
		                render_target_ctx.dummy_image_3d_alloc);
		render_target_ctx.dummy_image_3d       = VK_NULL_HANDLE;
		render_target_ctx.dummy_image_3d_alloc = VK_NULL_HANDLE;
	}

	if (render_target_ctx.lut_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(vulkan_ctx.device, render_target_ctx.lut_sampler, nullptr);
		render_target_ctx.lut_sampler = VK_NULL_HANDLE;
	}
	if (render_target_ctx.material_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(vulkan_ctx.device, render_target_ctx.material_sampler, nullptr);
		render_target_ctx.material_sampler = VK_NULL_HANDLE;
	}
	if (render_target_ctx.descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(vulkan_ctx.device, render_target_ctx.descriptor_pool, nullptr);
		render_target_ctx.descriptor_pool = VK_NULL_HANDLE;
	}
	if (render_target_ctx.descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(vulkan_ctx.device, render_target_ctx.descriptor_set_layout,
		                             nullptr);
		render_target_ctx.descriptor_set_layout = VK_NULL_HANDLE;
	}

	if (scene_ctx.camera_buffer != VK_NULL_HANDLE && scene_ctx.camera_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.camera_buffer,
		                 scene_ctx.camera_alloc);
		scene_ctx.camera_buffer = VK_NULL_HANDLE;
		scene_ctx.camera_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.material_buffer != VK_NULL_HANDLE && scene_ctx.material_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.material_buffer,
		                 scene_ctx.material_alloc);
		scene_ctx.material_buffer = VK_NULL_HANDLE;
		scene_ctx.material_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.mesh_buffer != VK_NULL_HANDLE && scene_ctx.mesh_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.mesh_buffer, scene_ctx.mesh_alloc);
		scene_ctx.mesh_buffer = VK_NULL_HANDLE;
		scene_ctx.mesh_alloc  = VK_NULL_HANDLE;
		scene_ctx.mesh_mapped = nullptr;
	}
	if (scene_ctx.vertex_buffer != VK_NULL_HANDLE && scene_ctx.vertex_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.vertex_buffer,
		                 scene_ctx.vertex_alloc);
		scene_ctx.vertex_buffer = VK_NULL_HANDLE;
		scene_ctx.vertex_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.index_buffer != VK_NULL_HANDLE && scene_ctx.index_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.index_buffer,
		                 scene_ctx.index_alloc);
		scene_ctx.index_buffer = VK_NULL_HANDLE;
		scene_ctx.index_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.hipr_visible_count_buffer != VK_NULL_HANDLE &&
	    scene_ctx.hipr_visible_count_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.hipr_visible_count_buffer,
		                 scene_ctx.hipr_visible_count_alloc);
		scene_ctx.hipr_visible_count_buffer = VK_NULL_HANDLE;
		scene_ctx.hipr_visible_count_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.hipr_secondary_count_buffer != VK_NULL_HANDLE &&
	    scene_ctx.hipr_secondary_count_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.hipr_secondary_count_buffer,
		                 scene_ctx.hipr_secondary_count_alloc);
		scene_ctx.hipr_secondary_count_buffer = VK_NULL_HANDLE;
		scene_ctx.hipr_secondary_count_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.hipr_shadow_count_buffer != VK_NULL_HANDLE &&
	    scene_ctx.hipr_shadow_count_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.hipr_shadow_count_buffer,
		                 scene_ctx.hipr_shadow_count_alloc);
		scene_ctx.hipr_shadow_count_buffer = VK_NULL_HANDLE;
		scene_ctx.hipr_shadow_count_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.hipr_score_buffer != VK_NULL_HANDLE &&
	    scene_ctx.hipr_score_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.hipr_score_buffer,
		                 scene_ctx.hipr_score_alloc);
		scene_ctx.hipr_score_buffer = VK_NULL_HANDLE;
		scene_ctx.hipr_score_alloc  = VK_NULL_HANDLE;
	}
	if (scene_ctx.hipr_order_buffer != VK_NULL_HANDLE &&
	    scene_ctx.hipr_order_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, scene_ctx.hipr_order_buffer,
		                 scene_ctx.hipr_order_alloc);
		scene_ctx.hipr_order_buffer = VK_NULL_HANDLE;
		scene_ctx.hipr_order_alloc  = VK_NULL_HANDLE;
	}

	destroy_tlas_resources(render_target_ctx.allocator, vulkan_ctx.device);
	for (auto& blas : as_ctx.blases) {
		if (blas.handle != VK_NULL_HANDLE) {
			vkDestroyAccelerationStructureKHR(vulkan_ctx.device, blas.handle, nullptr);
			blas.handle = VK_NULL_HANDLE;
		}
		if (blas.buffer != VK_NULL_HANDLE && blas.buffer_alloc != VK_NULL_HANDLE) {
			vmaDestroyBuffer(render_target_ctx.allocator, blas.buffer, blas.buffer_alloc);
			blas.buffer       = VK_NULL_HANDLE;
			blas.buffer_alloc = VK_NULL_HANDLE;
		}
	}
	as_ctx.blases.clear();

	if (sync_ctx.in_flight != VK_NULL_HANDLE) {
		vkDestroyFence(vulkan_ctx.device, sync_ctx.in_flight, nullptr);
	}

	for (auto s : sync_ctx.render_finished)
		vkDestroySemaphore(vulkan_ctx.device, s, nullptr);
	for (auto s : sync_ctx.image_available)
		vkDestroySemaphore(vulkan_ctx.device, s, nullptr);

	if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
		vkDestroyQueryPool(vulkan_ctx.device, gpu_timing_ctx.query_pool, nullptr);
		gpu_timing_ctx.query_pool = VK_NULL_HANDLE;
	}

	if (command_ctx.command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(vulkan_ctx.device, command_ctx.command_pool, nullptr);
	}

	if (render_target_ctx.allocator != VK_NULL_HANDLE) {
		vmaDestroyAllocator(render_target_ctx.allocator);
	}

	if (vulkan_ctx.surface != VK_NULL_HANDLE) {
		vkb::destroy_surface(vulkan_ctx.instance, vulkan_ctx.surface);
	}

	if (vulkan_ctx.device != VK_NULL_HANDLE) {
		vkb::destroy_device(vulkan_ctx.log_device);
	}

	if (vulkan_ctx.instance.instance != VK_NULL_HANDLE) {
		vkb::destroy_instance(vulkan_ctx.instance);
	}
}

// ============================================================
// === Main loop
// ============================================================
void Runtime::runMainLoop() {
	std::cout << "[INFO] Entering main loop\n";
	runLoop();
	std::cout << "[INFO] Main loop exited\n";
}
void Runtime::runLoop() {
	const auto sanitize_hipr_ten_step_value = [](uint32_t value) -> uint32_t {
		// Allowed set: 1, 10, 20, ... 100
		if (value <= 1u) {
			return 1u;
		}
		if (value >= 100u) {
			return 100u;
		}

		uint32_t snapped = ((value + 5u) / 10u) * 10u;        // nearest 10
		snapped          = std::max(snapped, 10u);
		snapped          = std::min(snapped, 100u);
		return snapped;
	};

	std::cout << "[INFO] Main loop startup state: should_close="
	          << (m_window->shouldClose() ? 1 : 0) << " framebuffer=" << m_window->width() << "x"
	          << m_window->height() << "\n";

	double last_frame_time = glfwGetTime();

	// Initialise fly camera from the scene camera
	FlyCamera fly_cam(m_scene->m_camera.m_position, m_scene->m_camera.m_target,
	                  m_scene->m_camera.m_fov, 0.5f);
	ui::selection_ctx.camera.fov_deg = fly_cam.m_fov;
	float last_camera_fov            = ui::selection_ctx.camera.fov_deg;

	double    last_time           = glfwGetTime();
	uint32_t  frame_number        = 0;
	GPUCamera previous_gpu_camera = fly_cam.pack();

	// Visibility pass must run before the first path-trace dispatch and
	// whenever the camera moves, resizes, or a new object is selected.
	bool needs_visibility_pass = true;

	// Tracks active material drag/edit interactions from the selection panel.
	bool                    material_edit_mode       = false;
	bool                    camera_was_moving        = false;
	bool                    hipr_force_clear_order   = true;
	ui::RenderDebugViewMode last_render_mode         = ui::selection_ctx.debug_view_mode;
	bool                    show_all_gui             = true;
	bool                    show_selection_panel     = true;
	bool                    auto_water_paused        = false;
	bool                    last_water_paused        = overlay_ctx.controls.water_paused;
	float                   selection_voice_loudness = 0.0f;
	float applied_render_scale        = std::clamp(overlay_ctx.controls.render_scale, 0.50f, 1.0f);
	overlay_ctx.controls.render_scale = applied_render_scale;

	// One-shot key-press trackers
	int prev_f6  = GLFW_RELEASE;
	int prev_f11 = GLFW_RELEASE;
	int prev_lmb = GLFW_RELEASE;

	ui::LightingSettings last_lighting               = ui::selection_ctx.lighting;
	bool                 hipr_object_sampling_active = false;
	bool                 hipr_object_sampling_done   = false;
	uint32_t             hipr_object_sampling_rank   = 0;
	uint32_t             hipr_object_sampling_frame  = 0;
	uint32_t             hipr_full_scene_frame       = 0;

	const auto reset_hipr_object_sampling = [&]() {
		hipr_object_sampling_active = false;
		hipr_object_sampling_done   = false;
		hipr_object_sampling_rank   = 0;
		hipr_object_sampling_frame  = 0;
		hipr_full_scene_frame       = 0;
	};
	const auto restart_hipr_object_sampling = [&]() {
		hipr_object_sampling_active = true;
		hipr_object_sampling_done   = false;
		hipr_object_sampling_rank   = 0;
		hipr_object_sampling_frame  = 0;
		hipr_full_scene_frame       = 0;
	};

	while (!m_window->shouldClose()) {
		m_window->pollEvents();

		const uint32_t framebuffer_width  = m_window->width();
		const uint32_t framebuffer_height = m_window->height();
		if (framebuffer_width == 0 || framebuffer_height == 0) {
			m_window->waitEvents();
			last_frame_time = glfwGetTime();
			continue;
		}

		if (swapchain_ctx.extent.width != framebuffer_width ||
		    swapchain_ctx.extent.height != framebuffer_height) {
			recreateSwapchainResources();
			frame_number          = 0;
			needs_visibility_pass = true;
			reset_hipr_object_sampling();
		}

		const double time_seconds_raw = glfwGetTime();
		const float  time_seconds     = static_cast<float>(time_seconds_raw);
		const float  delta_time =
		    static_cast<float>(std::max(time_seconds_raw - last_frame_time, 1.0e-6));
		last_frame_time = time_seconds_raw;

		updateRenderDiagnostics(delta_time);

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		const audio::ReactiveAudioInputFrame audio_input =
		    audio::buildAudioInputFrame(m_microphone.get(), time_seconds);

		const auto handle_water_pause_transition = [&](bool water_paused_now) {
			if (water_paused_now != last_water_paused) {
				frame_number           = 0;
				needs_visibility_pass  = true;
				hipr_force_clear_order = true;
				last_water_paused      = water_paused_now;
				reset_hipr_object_sampling();
			}
		};

		const auto run_audio_water_policy = [&]() {
			syncFloatingSimulationBoundary(m_water_surface.get());
			const auto policy = vulkan::framepolicy::updateAudioWaterPolicy(
			    m_audio_controller.get(), m_water_surface.get(), overlay_ctx.controls,
			    overlay_ctx.diagnostics, audio_input, time_seconds, delta_time,
			    selection_voice_loudness, auto_water_paused);
			handle_water_pause_transition(policy.effective_water_paused);
			return policy;
		};

		bool effective_water_paused = run_audio_water_policy().effective_water_paused;

		vulkan::framepolicy::handleGuiVisibilityHotkey(ImGui::IsKeyPressed(ImGuiKey_F1, false),
		                                               show_all_gui, overlay_ctx.show_control_panel,
		                                               show_selection_panel);

		bool controls_changed = false;
		if (show_all_gui && overlay_ctx.show_control_panel) {
			controls_changed = ui::drawAudienceControlPanel(
			    &overlay_ctx.show_control_panel, overlay_ctx.controls, overlay_ctx.diagnostics);
		}
		const float clamped_render_scale =
		    std::clamp(overlay_ctx.controls.render_scale, 0.50f, 1.0f);
		if (std::abs(clamped_render_scale - overlay_ctx.controls.render_scale) > 1.0e-4f) {
			overlay_ctx.controls.render_scale = clamped_render_scale;
			controls_changed                  = true;
		}
		if (std::abs(clamped_render_scale - applied_render_scale) > 1.0e-4f) {
			applied_render_scale = clamped_render_scale;
			recreateSwapchainResources();
			frame_number           = 0;
			needs_visibility_pass  = true;
			hipr_force_clear_order = true;
			reset_hipr_object_sampling();
			continue;
		}
		ui::SelectionPanelResult selection_panel_result{};
		if (show_all_gui && show_selection_panel) {
			selection_panel_result = ui::drawSelectionPanel(m_scene.get(), selection_voice_loudness,
			                                                &show_selection_panel);
		}
		{
			const uint32_t sanitized_rank_count =
			    std::clamp(ui::selection_ctx.hipr_debug.rank_count, 1u, HIPR_TOP_K);
			const uint32_t sanitized_frames =
			    sanitize_hipr_ten_step_value(ui::selection_ctx.hipr_debug.frames_per_object);
			const uint32_t sanitized_spp =
			    std::clamp(ui::selection_ctx.path_tracing.spp, 1u, 1024u);
			const uint32_t sanitized_max_bounces =
			    std::clamp(ui::selection_ctx.path_tracing.max_bounces, 1u, 1024u);
			const uint32_t sanitized_water_spp_override =
			    std::clamp(ui::selection_ctx.path_tracing.hipr_water_spp_override, 0u, 1024u);

			if (sanitized_rank_count != ui::selection_ctx.hipr_debug.rank_count) {
				ui::selection_ctx.hipr_debug.rank_count      = sanitized_rank_count;
				selection_panel_result.hipr_settings_changed = true;
			}
			if (sanitized_frames != ui::selection_ctx.hipr_debug.frames_per_object) {
				ui::selection_ctx.hipr_debug.frames_per_object = sanitized_frames;
				selection_panel_result.hipr_settings_changed   = true;
			}
			if (sanitized_spp != ui::selection_ctx.path_tracing.spp) {
				ui::selection_ctx.path_tracing.spp                   = sanitized_spp;
				selection_panel_result.path_tracing_settings_changed = true;
			}
			if (sanitized_max_bounces != ui::selection_ctx.path_tracing.max_bounces) {
				ui::selection_ctx.path_tracing.max_bounces           = sanitized_max_bounces;
				selection_panel_result.path_tracing_settings_changed = true;
			}
			if (sanitized_water_spp_override !=
			    ui::selection_ctx.path_tracing.hipr_water_spp_override) {
				ui::selection_ctx.path_tracing.hipr_water_spp_override =
				    sanitized_water_spp_override;
				selection_panel_result.path_tracing_settings_changed = true;
			}
		}

		if (ui::selection_ctx.debug_view_mode != last_render_mode) {
			frame_number           = 0;
			needs_visibility_pass  = true;
			hipr_force_clear_order = true;
			last_render_mode       = ui::selection_ctx.debug_view_mode;
			reset_hipr_object_sampling();
		}

		if (controls_changed) {
			effective_water_paused = run_audio_water_policy().effective_water_paused;
		}

		if (selection_panel_result.material_edit_active && !material_edit_mode) {
			// Entering edit mode: restart accumulation at the edited value.
			material_edit_mode = true;
			frame_number       = 0;
			reset_hipr_object_sampling();
		}

		if (selection_panel_result.material_changed) {
			ui::applySelectedMaterialEditor(m_scene.get(), render_target_ctx.allocator,
			                                scene_ctx.material_mapped, scene_ctx.material_count,
			                                scene_ctx.material_alloc);
			// Reset only when a parameter value actually changes.
			frame_number = 0;
			restart_hipr_object_sampling();
		}
		if (selection_panel_result.transform_changed) {
			if (applySelectedTransformEditor(m_scene.get(), render_target_ctx.allocator)) {
				frame_number           = 0;
				needs_visibility_pass  = true;
				hipr_force_clear_order = true;
				reset_hipr_object_sampling();
			}
		}

		if (selection_panel_result.material_edit_just_finished) {
			// Leaving edit mode: restart full-screen accumulation.
			material_edit_mode = false;
			frame_number       = 0;
		}

		if (selection_panel_result.selection_changed) {
			frame_number           = 0;
			needs_visibility_pass  = true;
			material_edit_mode     = false;
			hipr_force_clear_order = true;
			refreshSelectedTransformEditor();
			reset_hipr_object_sampling();
		}
		if (selection_panel_result.hipr_settings_changed) {
			// Safety: changing HiPR schedule parameters should restart ranked scheduling.
			frame_number           = 0;
			needs_visibility_pass  = true;
			hipr_force_clear_order = true;
			if (ui::selection_ctx.debug_view_mode == ui::RenderDebugViewMode::HiPR &&
			    ui::selection_ctx.selected_mesh_index >= 0) {
				restart_hipr_object_sampling();
			} else {
				reset_hipr_object_sampling();
			}
		}
		if (selection_panel_result.path_tracing_settings_changed) {
			frame_number = 0;
			if (ui::selection_ctx.debug_view_mode == ui::RenderDebugViewMode::HiPR &&
			    ui::selection_ctx.selected_mesh_index >= 0) {
				restart_hipr_object_sampling();
			} else {
				reset_hipr_object_sampling();
			}
		}
		if (floating_settings_dirty && m_water_surface != nullptr) {
			syncFloatingSimulationSettings(m_water_surface.get());
			floating_settings_dirty = false;
		}

		{
			const ui::LightingSettings& cur = ui::selection_ctx.lighting;
			if (cur.skybox_enabled != last_lighting.skybox_enabled ||
			    cur.directional_light_enabled != last_lighting.directional_light_enabled ||
			    cur.sun_elevation_deg != last_lighting.sun_elevation_deg ||
			    cur.sun_azimuth_deg != last_lighting.sun_azimuth_deg ||
			    cur.sun_intensity != last_lighting.sun_intensity) {
				frame_number  = 0;
				last_lighting = cur;
				reset_hipr_object_sampling();
			}
		}

		{
			float cur_camera_fov = std::clamp(ui::selection_ctx.camera.fov_deg, 20.0f, 120.0f);
			ui::selection_ctx.camera.fov_deg = cur_camera_fov;
			if (std::abs(cur_camera_fov - last_camera_fov) > 1.0e-4f) {
				last_camera_fov         = cur_camera_fov;
				fly_cam.m_fov           = cur_camera_fov;
				m_scene->m_camera.m_fov = cur_camera_fov;
				frame_number            = 0;
				needs_visibility_pass   = true;
				hipr_force_clear_order  = true;
				reset_hipr_object_sampling();
			}
		}

		if (overlay_ctx.controls.reset_water_requested) {
			if (m_water_surface != nullptr) {
				m_water_surface->requestReset();
			}
			overlay_ctx.controls.reset_water_requested = false;
		}

		if (overlay_ctx.controls.reset_objects_requested) {
			if (m_water_surface != nullptr) {
				m_water_surface->requestObjectReset();
			}
			overlay_ctx.controls.reset_objects_requested = false;
		}
		vulkan::waterbridge::updateWaterSurfaceParamsBuffer(
		    makeWaterSurfaceParamsBufferContext(), water_surface_render_ctx, m_scene.get(),
		    m_water_surface.get(), vulkan::floating::firstFloatingObjectId(floating_mesh_groups));

		if (show_all_gui && overlay_ctx.controls.show_overlay) {
			ui::drawAudienceOverlay(ImGui::GetIO().DisplaySize, overlay_ctx.controls.overlay,
			                        overlay_ctx.controls.style);
		}
		ImGui::Render();

		glfwPollEvents();

		double now = glfwGetTime();
		float  dt  = static_cast<float>(now - last_time);
		last_time  = now;
		dt         = std::clamp(dt, 0.0001f, 0.1f);

		// ---- Check if the window was resized --------------------------------
		uint32_t fb_w = m_window->width();
		uint32_t fb_h = m_window->height();
		if (fb_w != swapchain_ctx.swapchain.extent.width ||
		    fb_h != swapchain_ctx.swapchain.extent.height) {
			recreateSwapchainResources();
			frame_number          = 0;
			needs_visibility_pass = true;
			reset_hipr_object_sampling();
		}

		// ---- F11: fullscreen toggle -----------------------------------------
		int f11 = glfwGetKey(m_window->handle(), GLFW_KEY_F11);
		if (f11 == GLFW_PRESS && prev_f11 == GLFW_RELEASE) {
			m_window->toggle_fullscreen();
		}
		prev_f11 = f11;

		// ---- F6: shader hot-reload ------------------------------------------
		int f6 = glfwGetKey(m_window->handle(), GLFW_KEY_F6);
		if (f6 == GLFW_PRESS && prev_f6 == GLFW_RELEASE) {
			if (rebuild_pipeline(ui::selection_ctx.debug_view_mode)) {
				frame_number          = 0;
				needs_visibility_pass = true;
				reset_hipr_object_sampling();
			}
		}
		prev_f6 = f6;

		// ---- Fly-camera update ----------------------------------------------
		const bool camera_moving_this_frame = fly_cam.update(m_window->handle(), dt);
		if (camera_moving_this_frame) {
			frame_number           = 0;
			needs_visibility_pass  = true;
			hipr_force_clear_order = true;
			reset_hipr_object_sampling();
		}
		if (!camera_moving_this_frame && camera_was_moving) {
			// Start a fresh accumulation the first frame after camera motion stops.
			frame_number = 0;
			reset_hipr_object_sampling();
		}
		camera_was_moving = camera_moving_this_frame;

		// Upload camera to GPU (persistent mapping – no staging needed)
		const GPUCamera gpu_camera          = fly_cam.pack();
		GPUCamera       previous_for_upload = previous_gpu_camera;
		if (frame_number == 0u) {
			previous_for_upload = gpu_camera;
		}

		const int current_lmb = glfwGetMouseButton(m_window->handle(), GLFW_MOUSE_BUTTON_LEFT);
		if (current_lmb == GLFW_PRESS && prev_lmb == GLFW_RELEASE && !fly_cam.isMouseCaptured() &&
		    !ImGui::GetIO().WantCaptureMouse) {
			if (ui::selectMesh(m_scene.get(),
			                   pickMeshAtCursor(m_scene.get(), m_window->handle(), gpu_camera,
			                                    framebuffer_width, framebuffer_height))) {
				frame_number          = 0;
				needs_visibility_pass = true;
				material_edit_mode    = false;
				refreshSelectedTransformEditor();
				reset_hipr_object_sampling();
			}
		}
		prev_lmb = current_lmb;

		GPUCamera gpu_cameras[2] = {gpu_camera, previous_for_upload};
		memcpy(scene_ctx.camera_mapped, gpu_cameras, sizeof(gpu_cameras));
		previous_gpu_camera = gpu_camera;

		const auto       active_render_mode = ui::selection_ctx.debug_view_mode;
		const VkPipeline active_pipeline =
		    compute_ctx.pipelines[render_mode_index(active_render_mode)];
		if (active_pipeline == VK_NULL_HANDLE) {
			continue;
		}

		// ---- Frame rendering ------------------------------------------------
		const uint32_t frame_idx =
		    frame_number % static_cast<uint32_t>(swapchain_ctx.images.size());

		VkResult wait_result =
		    vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
		if (wait_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkWaitForFences failed: " << static_cast<int>(wait_result) << "\n";
			if (wait_result == VK_ERROR_DEVICE_LOST) {
				break;
			}
			continue;
		}
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE && gpu_timing_ctx.has_submission &&
		    vulkan_ctx.timestamp_period_ns > 0.0f) {
			uint64_t       timestamp_results[4] = {0, 0, 0, 0};
			const VkResult query_result         = vkGetQueryPoolResults(
			    vulkan_ctx.device, gpu_timing_ctx.query_pool, 0, 4, sizeof(timestamp_results),
			    timestamp_results, sizeof(uint64_t),
			    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
			if (query_result == VK_SUCCESS) {
				const float    ns_to_ms     = 1.0e-6f;
				const float    tick_ms      = vulkan_ctx.timestamp_period_ns * ns_to_ms;
				const uint64_t sim_ticks    = timestamp_results[1] >= timestamp_results[0] ?
				                                  (timestamp_results[1] - timestamp_results[0]) :
				                                  0ull;
				const uint64_t render_ticks = timestamp_results[3] >= timestamp_results[2] ?
				                                  (timestamp_results[3] - timestamp_results[2]) :
				                                  0ull;
				gpu_timing_ctx.simulation_pass_time_ms = static_cast<float>(sim_ticks) * tick_ms;
				gpu_timing_ctx.render_pass_time_ms     = static_cast<float>(render_ticks) * tick_ms;
			}
		}

		auto drain_acquire_semaphore = [&](uint32_t sem_index) {
			VkPipelineStageFlags drain_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			VkSubmitInfo         drain_info{};
			drain_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			drain_info.waitSemaphoreCount = 1;
			drain_info.pWaitSemaphores    = &sync_ctx.image_available[sem_index];
			drain_info.pWaitDstStageMask  = &drain_stage;
			VkResult drain_submit_result =
			    vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &drain_info, VK_NULL_HANDLE);
			if (drain_submit_result != VK_SUCCESS) {
				std::cerr << "[SYNC] failed to drain acquire semaphore: "
				          << static_cast<int>(drain_submit_result) << "\n";
				return;
			}
			VkResult drain_wait_result = vkQueueWaitIdle(vulkan_ctx.graphics_queue);
			if (drain_wait_result != VK_SUCCESS) {
				std::cerr << "[SYNC] vkQueueWaitIdle after drain failed: "
				          << static_cast<int>(drain_wait_result) << "\n";
			}
		};

		uint32_t image_index    = 0;
		VkResult acquire_result = vkAcquireNextImageKHR(
		    vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
		    sync_ctx.image_available[frame_idx], VK_NULL_HANDLE, &image_index);
		const bool acquire_suboptimal = (acquire_result == VK_SUBOPTIMAL_KHR);
		if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapchainResources();
			frame_number          = 0;
			needs_visibility_pass = true;
			reset_hipr_object_sampling();
			continue;
		}
		if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
			std::cerr << "[SYNC] vkAcquireNextImageKHR failed: " << static_cast<int>(acquire_result)
			          << "\n";
			continue;
		}

		VkResult reset_cmd_result = vkResetCommandBuffer(command_ctx.command_buffer, 0);
		if (reset_cmd_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkResetCommandBuffer failed: "
			          << static_cast<int>(reset_cmd_result) << "\n";
			drain_acquire_semaphore(frame_idx);
			continue;
		}
		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VkResult begin_result = vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);
		if (begin_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkBeginCommandBuffer failed: " << static_cast<int>(begin_result)
			          << "\n";
			drain_acquire_semaphore(frame_idx);
			continue;
		}

		VkCommandBuffer cmd = command_ctx.command_buffer;
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
			vkCmdResetQueryPool(cmd, gpu_timing_ctx.query_pool, 0, 4);
		}

		const bool floating_meshes_moved = updateFloatingMeshTransformsFromSimulation(
		    m_scene.get(), m_water_surface.get(), render_target_ctx.allocator);
		if (floating_meshes_moved) {
			needs_visibility_pass = true;
			// Floating animation is expected in water scenes. Keep HiPR/HiPRVis schedules alive
			// instead of restarting them every frame while transforms keep updating.
			if (ui::selection_ctx.debug_view_mode != ui::RenderDebugViewMode::HiPR &&
			    ui::selection_ctx.debug_view_mode != ui::RenderDebugViewMode::HiPRVis) {
				frame_number = 0;
			}
		}
		if (tlas_update_pending) {
			if (as_ctx.tlas != VK_NULL_HANDLE) {
				upload_tlas_instances(render_target_ctx.allocator, as_ctx.blases,
				                      m_scene->m_meshes);
				record_tlas_build(vulkan_ctx.device, cmd, true);
			}
			tlas_update_pending = false;
		}
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                    gpu_timing_ctx.query_pool, 0);
		}
		if (m_water_surface != nullptr &&
		    (!effective_water_paused || !m_water_surface->hasRecordedFrame())) {
			m_water_surface->record(cmd);
			vulkan::waterbridge::updateWaterSurfaceImageDescriptors(
			    vulkan_ctx.device, render_target_ctx.descriptor_set, m_water_surface.get());
		}
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                    gpu_timing_ctx.query_pool, 1);
		}

		VkImageLayout storage_old = render_target_ctx.storage_image_initialized ?
		                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
		                                VK_IMAGE_LAYOUT_UNDEFINED;
		transition_layout(cmd, render_target_ctx.storage_image, storage_old,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		VkImageLayout swap_old = swapchain_ctx.image_initialized[image_index] ?
		                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
		                             VK_IMAGE_LAYOUT_UNDEFINED;
		transition_layout(cmd, swapchain_ctx.images[image_index], swap_old,
		                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		const auto current_render_mode = ui::selection_ctx.debug_view_mode;
		const bool mode_uses_temporal_history =
		    current_render_mode == ui::RenderDebugViewMode::Naive ||
		    current_render_mode == ui::RenderDebugViewMode::HiPR ||
		    current_render_mode == ui::RenderDebugViewMode::HiPRVis;
		if (mode_uses_temporal_history && frame_number > 0u) {
			copyNaiveHistoryImages(cmd);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, active_pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline_layout, 0,
		                        1, &render_target_ctx.descriptor_set, 0, nullptr);
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                    gpu_timing_ctx.query_pool, 2);
		}

		const bool hold_naive_pause_accumulation =
		    effective_water_paused &&
		    ui::selection_ctx.debug_view_mode == ui::RenderDebugViewMode::Naive;
		const bool hold_hipr_animation_accumulation =
		    floating_meshes_moved && (current_render_mode == ui::RenderDebugViewMode::HiPR ||
		                              current_render_mode == ui::RenderDebugViewMode::HiPRVis);
		if (needs_visibility_pass && !hold_naive_pause_accumulation &&
		    !hold_hipr_animation_accumulation) {
			frame_number = 0;
		}

		const uint32_t hipr_frames_per_object =
		    sanitize_hipr_ten_step_value(ui::selection_ctx.hipr_debug.frames_per_object);

		PathTracerPushConstants pc{};
		pc.frame               = frame_number;
		pc.material_count      = scene_ctx.material_count;
		pc.selected_mesh_index = ui::selection_ctx.selected_mesh_index;
		pc.outline_width       = ui::selection_ctx.outline_width;
		pc.debug_view_mode     = static_cast<int32_t>(ui::selection_ctx.debug_view_mode);
		pc.outline_color       = ui::selection_ctx.outline_color;
		pc.spp                 = std::clamp(ui::selection_ctx.path_tracing.spp, 1u, 1024u);
		pc.max_bounces         = std::clamp(ui::selection_ctx.path_tracing.max_bounces, 1u, 1024u);
		pc.enable_tonemapping  = overlay_ctx.controls.render_post.enable_tonemapping ? 1u : 0u;
		pc.exposure_bias       = overlay_ctx.controls.render_post.exposure_bias;
		pc.hipr_object_count   = scene_ctx.mesh_count;
		pc.hipr_top_k =
		    std::min(scene_ctx.hipr_top_k, std::max(ui::selection_ctx.hipr_debug.rank_count, 1u));
		pc.hipr_render_rank      = -1;
		pc.hipr_incremental_sort = ui::selection_ctx.hipr_debug.incremental_sorting ? 1u : 0u;
		pc.hipr_clear_order      = hipr_force_clear_order ? 1u : 0u;
		pc.hipr_vis_enable_tint  = ui::selection_ctx.hipr_debug.vis_enable_influence_tint ? 1u : 0u;
		pc.hipr_vis_rainbow_tint = ui::selection_ctx.hipr_debug.vis_rainbow_tint ? 1u : 0u;
		const uint32_t water_spp_override =
		    std::clamp(ui::selection_ctx.path_tracing.hipr_water_spp_override, 0u, 1024u) &
		    kWaterSppMask;
		pc.hipr_reserved0 = water_spp_override | (effective_water_paused ? kWaterPauseBit : 0u) |
		                    (floating_meshes_moved ? kSceneDynamicBit : 0u);
		pc.hipr_frames_per_object = hipr_frames_per_object;
		pc.hipr_score_blend = std::clamp(ui::selection_ctx.hipr_debug.score_blend, 0.05f, 1.0f);
		pc.hipr_vis_tint_strength =
		    std::clamp(ui::selection_ctx.hipr_debug.vis_tint_strength, 0.0f, 1.0f);
		pc.skybox_enabled = ui::selection_ctx.lighting.skybox_enabled ? 1u : 0u;
		pc.directional_light_enabled =
		    ui::selection_ctx.lighting.directional_light_enabled ? 1u : 0u;
		{
			const float elev =
			    ui::selection_ctx.lighting.sun_elevation_deg * (3.14159265f / 180.0f);
			const float azim = ui::selection_ctx.lighting.sun_azimuth_deg * (3.14159265f / 180.0f);
			pc.sun_dir_x     = std::cos(elev) * std::sin(azim);
			pc.sun_dir_y     = std::sin(elev);
			pc.sun_dir_z     = std::cos(elev) * std::cos(azim);
		}
		pc.sun_intensity = ui::selection_ctx.lighting.sun_intensity;

		const uint32_t dispatch_w = (render_target_ctx.extent.width + 15) / 16;
		const uint32_t dispatch_h = (render_target_ctx.extent.height + 15) / 16;

		const auto render_mode   = ui::selection_ctx.debug_view_mode;
		const bool hipr_mode     = render_mode == ui::RenderDebugViewMode::HiPR;
		const bool obj_id_mode   = render_mode == ui::RenderDebugViewMode::ObjectIds;
		const bool hipr_vis_mode = render_mode == ui::RenderDebugViewMode::HiPRVis;
		const bool naive_mode    = render_mode == ui::RenderDebugViewMode::Naive;

		// Mode-specific push-constant sanitization: values that are not consumed by
		// the active shader are zeroed to make unused data explicit and predictable.
		switch (render_mode) {
			case ui::RenderDebugViewMode::HiPR:
				pc.hipr_vis_enable_tint   = 0u;
				pc.hipr_vis_rainbow_tint  = 0u;
				pc.hipr_vis_tint_strength = 0.0f;
				break;
			case ui::RenderDebugViewMode::HiPRVis:
				pc.enable_tonemapping = 0u;
				pc.exposure_bias      = 0.0f;
				break;
			case ui::RenderDebugViewMode::Naive:
				pc.hipr_object_count      = 0u;
				pc.hipr_top_k             = 0u;
				pc.hipr_render_rank       = -1;
				pc.hipr_incremental_sort  = 0u;
				pc.hipr_clear_order       = 0u;
				pc.hipr_vis_enable_tint   = 0u;
				pc.hipr_vis_rainbow_tint  = 0u;
				pc.hipr_reserved0         = effective_water_paused ? kWaterPauseBit : 0u;
				pc.hipr_frames_per_object = 0u;
				pc.hipr_score_blend       = 0.0f;
				pc.hipr_vis_tint_strength = 0.0f;
				break;
			case ui::RenderDebugViewMode::ObjectIds:
				pc.spp                       = 0u;
				pc.max_bounces               = 0u;
				pc.enable_tonemapping        = 0u;
				pc.exposure_bias             = 0.0f;
				pc.material_count            = 0u;
				pc.hipr_object_count         = 0u;
				pc.hipr_top_k                = 0u;
				pc.hipr_render_rank          = -1;
				pc.hipr_incremental_sort     = 0u;
				pc.hipr_clear_order          = 0u;
				pc.hipr_vis_enable_tint      = 0u;
				pc.hipr_vis_rainbow_tint     = 0u;
				pc.hipr_reserved0            = 0u;
				pc.hipr_frames_per_object    = 0u;
				pc.hipr_score_blend          = 0.0f;
				pc.hipr_vis_tint_strength    = 0.0f;
				pc.skybox_enabled            = 0u;
				pc.directional_light_enabled = 0u;
				pc.sun_dir_x                 = 0.0f;
				pc.sun_dir_y                 = 0.0f;
				pc.sun_dir_z                 = 0.0f;
				pc.sun_intensity             = 0.0f;
				break;
		}

		const bool hipr_active = ui::selection_ctx.selected_mesh_index >= 0;
		const bool use_hipr_ranked =
		    (hipr_mode || hipr_vis_mode) && hipr_active && !camera_moving_this_frame;
		const bool hipr_object_sampling_enabled =
		    hipr_mode && hipr_active && hipr_object_sampling_active && !camera_moving_this_frame;
		const bool hipr_full_scene_sampling =
		    hipr_mode && hipr_active && hipr_object_sampling_done && !camera_moving_this_frame;

		bool       run_stage1 = true;
		bool       run_stage2 = camera_moving_this_frame || !material_edit_mode || hipr_active;
		bool       run_single_rank_stage4 = false;
		bool       run_ranked_stage4_loop = false;
		int32_t    single_rank_stage4_id  = -1;
		const bool hipr_vis_reveal_complete =
		    hipr_vis_mode && pc.hipr_top_k > 0u &&
		    (frame_number / hipr_frames_per_object) >= (pc.hipr_top_k - 1u);
		if (obj_id_mode) {
			run_stage1 = false;
			run_stage2 = true;
		} else if (naive_mode) {
			run_stage1 = false;
			run_stage2 = true;
		} else if (hipr_vis_mode) {
			run_stage1 = hipr_active;
			run_stage2 = true;        // Composite pass for reveal visualization.
		}

		if (hipr_object_sampling_enabled) {
			const uint32_t sample_rank =
			    (pc.hipr_top_k > 0u) ? std::min(hipr_object_sampling_rank, pc.hipr_top_k - 1u) : 0u;
			run_stage2             = false;
			run_ranked_stage4_loop = false;
			if (pc.hipr_top_k == 0u || sample_rank == 0u) {
				run_stage1 = true;
			} else {
				run_stage1             = false;
				run_single_rank_stage4 = true;
				single_rank_stage4_id  = static_cast<int32_t>(sample_rank);
			}
		} else if (hipr_full_scene_sampling) {
			// After the focused per-object sequence, switch to single-pass naive accumulation.
			run_stage1             = false;
			run_stage2             = true;
			run_ranked_stage4_loop = false;
		} else {
			if (hipr_vis_mode && hipr_vis_reveal_complete) {
				// In visualization mode, once the ranked reveal is complete, switch to
				// full-scene stage-2 tracing only.
				run_stage1 = false;
				run_stage2 = true;
			}
			run_ranked_stage4_loop =
			    use_hipr_ranked && run_stage2 && !(hipr_vis_mode && hipr_vis_reveal_complete);
		}

		if ((!hipr_mode || !hipr_active) &&
		    (hipr_object_sampling_active || hipr_object_sampling_done)) {
			reset_hipr_object_sampling();
		}

		// HiPR ranking refresh policy:
		// - refresh on visibility-pass rebuilds (camera/view/selection-driven)
		// - only refresh when stage 1 is running (stage 1 populates influence stats)
		// - do not periodically resort every N frames
		bool hipr_refresh = use_hipr_ranked && needs_visibility_pass && run_stage1;
		if (hipr_full_scene_sampling) {
			hipr_refresh = false;
		}

		// Stage 10: clear per-object HiPR stats and ordering slots.
		if (use_hipr_ranked && hipr_refresh) {
			pc.stage = 10;
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			const uint32_t clear_count =
			    std::max(pc.hipr_object_count, std::max(pc.hipr_top_k, 1u));
			vkCmdDispatch(cmd, (clear_count + 15) / 16, 1, 1);
			hipr_force_clear_order = false;
		}

		// Stage 0: visibility pass.
		if (needs_visibility_pass) {
			pc.stage = 0;
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);

			VkImageMemoryBarrier obj_id_barrier{};
			obj_id_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			obj_id_barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
			obj_id_barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
			obj_id_barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
			obj_id_barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
			obj_id_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			obj_id_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			obj_id_barrier.image               = render_target_ctx.object_id_image;
			obj_id_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
			                     &obj_id_barrier);

			needs_visibility_pass = false;
		}

		// Stage 1: trace selected object pixels and gather influence.
		if (run_stage1) {
			pc.stage            = 1;
			pc.hipr_render_rank = hipr_full_scene_sampling ? -2 : -1;
			pc.frame = hipr_object_sampling_enabled ?
			               hipr_object_sampling_frame :
			               (hipr_full_scene_sampling ? hipr_full_scene_frame : frame_number);
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
		}

		if (use_hipr_ranked && hipr_refresh) {
			VkMemoryBarrier stats_barrier{};
			stats_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			stats_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			stats_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &stats_barrier, 0,
			                     nullptr, 0, nullptr);

			// Stage 3: build top-K HiPR ordering (selected object is rank 0).
			pc.stage = 3;
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			vkCmdDispatch(cmd, 1, 1, 1);
		}

		if (run_stage1 && (run_stage2 || use_hipr_ranked)) {
			VkMemoryBarrier stage_barrier{};
			stage_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			stage_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			stage_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &stage_barrier, 0,
			                     nullptr, 0, nullptr);
		}

		if (run_single_rank_stage4) {
			pc.stage            = 4;
			pc.frame            = hipr_object_sampling_frame;
			pc.hipr_render_rank = single_rank_stage4_id;
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
		}

		if (run_ranked_stage4_loop) {
			// Stage 4: render by HiPR-ranked object order (skip rank 0; already handled by stage
			// 1).
			uint32_t stage4_end_rank = pc.hipr_top_k;
			if (hipr_vis_mode) {
				const uint32_t reveal_rank_count =
				    std::min(pc.hipr_top_k, 1u + (frame_number / hipr_frames_per_object));
				stage4_end_rank = reveal_rank_count;
			}

			for (uint32_t rank = 1; rank < stage4_end_rank; ++rank) {
				pc.stage            = 4;
				pc.frame            = frame_number;
				pc.hipr_render_rank = static_cast<int32_t>(rank);
				vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				                   sizeof(pc), &pc);
				vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
			}
		}

		if (run_stage2) {
			// Stage 2:
			// - Naive mode: full-frame un-ordered path tracing.
			// - Obj ID mode: flat object-id visualization.
			// - HiPR Vis mode: reveal/composite pass.
			// - Fallback: full-scene path tracing when no ranked HiPR pass is active.
			pc.stage            = 2;
			pc.hipr_render_rank = hipr_full_scene_sampling ? -2 : -1;
			pc.frame            = hipr_full_scene_sampling ? hipr_full_scene_frame : frame_number;
			if (hipr_full_scene_sampling) {
				// Force stage-2 to include selected-object pixels (true naive path).
				pc.debug_view_mode = static_cast<int32_t>(ui::RenderDebugViewMode::Naive);
			}
			vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
		}

		if (hipr_object_sampling_enabled && pc.hipr_top_k > 0u) {
			++hipr_object_sampling_frame;
			if (hipr_object_sampling_frame >= hipr_frames_per_object) {
				hipr_object_sampling_frame = 0;
				++hipr_object_sampling_rank;
				if (hipr_object_sampling_rank >= pc.hipr_top_k) {
					hipr_object_sampling_active = false;
					hipr_object_sampling_done   = true;
					hipr_object_sampling_rank   = 0;
					hipr_object_sampling_frame  = 0;
					hipr_full_scene_frame       = 0;
				}
			}
		}
		if (hipr_full_scene_sampling) {
			++hipr_full_scene_frame;
		}
		if (gpu_timing_ctx.query_pool != VK_NULL_HANDLE) {
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                    gpu_timing_ctx.query_pool, 3);
		}

		// Make compute writes visible to transfer
		VkMemoryBarrier compute_to_transfer{};
		compute_to_transfer.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		compute_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		compute_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &compute_to_transfer, 0, nullptr,
		                     0, nullptr);

		transition_layout(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);

		VkImageBlit blit_region{};
		blit_region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		blit_region.srcSubresource.mipLevel       = 0;
		blit_region.srcSubresource.baseArrayLayer = 0;
		blit_region.srcSubresource.layerCount     = 1;
		blit_region.srcOffsets[0]                 = {0, 0, 0};
		blit_region.srcOffsets[1] = {static_cast<int32_t>(render_target_ctx.extent.width),
		                             static_cast<int32_t>(render_target_ctx.extent.height), 1};

		blit_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		blit_region.dstSubresource.mipLevel       = 0;
		blit_region.dstSubresource.baseArrayLayer = 0;
		blit_region.dstSubresource.layerCount     = 1;
		blit_region.dstOffsets[0]                 = {0, 0, 0};
		blit_region.dstOffsets[1] = {static_cast<int32_t>(swapchain_ctx.extent.width),
		                             static_cast<int32_t>(swapchain_ctx.extent.height), 1};

		const VkFilter present_filter =
		    (render_target_ctx.extent.width == swapchain_ctx.extent.width &&
		     render_target_ctx.extent.height == swapchain_ctx.extent.height) ?
		        VK_FILTER_NEAREST :
		        VK_FILTER_LINEAR;
		vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		               &blit_region, present_filter);

		render_target_ctx.storage_image_initialized  = true;
		swapchain_ctx.image_initialized[image_index] = true;

		// ---- ImGui render pass ----------------------------------------------
		transition_layout(cmd, swapchain_ctx.images[image_index],
		                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

		VkRenderPassBeginInfo rp_begin{};
		rp_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rp_begin.renderPass        = overlay_ctx.render_pass;
		rp_begin.framebuffer       = overlay_ctx.framebuffers[image_index];
		rp_begin.renderArea.offset = {0, 0};
		rp_begin.renderArea.extent = swapchain_ctx.extent;
		rp_begin.clearValueCount   = 0;
		rp_begin.pClearValues      = nullptr;

		vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		vkCmdEndRenderPass(cmd);

		VkResult end_result = vkEndCommandBuffer(cmd);
		if (end_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkEndCommandBuffer failed: " << static_cast<int>(end_result)
			          << "\n";
			drain_acquire_semaphore(frame_idx);
			continue;
		}

		VkResult reset_fence_result = vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);
		if (reset_fence_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkResetFences failed: " << static_cast<int>(reset_fence_result)
			          << "\n";
			drain_acquire_semaphore(frame_idx);
			continue;
		}

		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

		VkSubmitInfo submit_info{};
		submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount   = 1;
		submit_info.pWaitSemaphores      = &sync_ctx.image_available[frame_idx];
		submit_info.pWaitDstStageMask    = &wait_stage;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &cmd;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &sync_ctx.render_finished[image_index];

		VkResult submit_result =
		    vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit_info, sync_ctx.in_flight);
		if (submit_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkQueueSubmit failed: " << static_cast<int>(submit_result) << "\n";
			drain_acquire_semaphore(frame_idx);

			// Restore to a signaled fence so the next frame does not block forever.
			vkDestroyFence(vulkan_ctx.device, sync_ctx.in_flight, nullptr);
			VkFenceCreateInfo signaled_fence_info{};
			signaled_fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			signaled_fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
			if (vkCreateFence(vulkan_ctx.device, &signaled_fence_info, nullptr,
			                  &sync_ctx.in_flight) != VK_SUCCESS) {
				std::cerr << "[SYNC] failed to recreate in-flight fence\n";
				break;
			}
			continue;
		}
		gpu_timing_ctx.has_submission = (gpu_timing_ctx.query_pool != VK_NULL_HANDLE);

		VkPresentInfoKHR present_info{};
		present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores    = &sync_ctx.render_finished[image_index];
		present_info.swapchainCount     = 1;
		present_info.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present_info.pImageIndices      = &image_index;

		VkResult present_result = vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present_info);
		if (acquire_suboptimal || present_result == VK_ERROR_OUT_OF_DATE_KHR ||
		    present_result == VK_SUBOPTIMAL_KHR) {
			recreateSwapchainResources();
			frame_number          = 0;
			needs_visibility_pass = true;
			reset_hipr_object_sampling();
			continue;
		}
		if (present_result != VK_SUCCESS) {
			std::cerr << "[SYNC] vkQueuePresentKHR failed: " << static_cast<int>(present_result)
			          << "\n";
			continue;
		}

		++frame_number;
	}
}
