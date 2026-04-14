#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
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

#include "slang.h"
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
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"
#include "tsunami/ui/audience_overlay.h"
#include "tsunami/ui/selection_panel.h"

struct VulkanContext {
	vkb::Instance       instance{};
	vkb::PhysicalDevice phys_device{};
	vkb::Device         log_device{};
	uint32_t            scratch_alignment;
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
	VkImage                            storage_image;
	VmaAllocation                      storage_image_alloc;
	VkImageView                        storage_image_view;
	bool                               storage_image_initialized = false;
	VkImage                            object_id_image           = VK_NULL_HANDLE;
	VmaAllocation                      object_id_image_alloc     = VK_NULL_HANDLE;
	VkImageView                        object_id_image_view      = VK_NULL_HANDLE;
	VkImage                            accum_image;
	VmaAllocation                      accum_image_alloc;
	VkImageView                        accum_image_view;
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

struct WaterSurfaceParamsGpu {
	glm::vec4 center_trace_half_height = glm::vec4(0.0f);
	glm::vec4 axis_u_half_extent       = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 axis_v_half_extent       = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
	glm::vec4 normal                   = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	int32_t   water_object_id          = -1;
	uint32_t  water_enabled            = 0;
	int32_t   water_material_index     = -1;
	float     water_height_to_world_scale = 1.0f;
	int32_t   first_floating_object_id = -1;
	uint32_t  _pad0                    = 0;
	uint32_t  _pad1                    = 0;
	uint32_t  _pad2                    = 0;
};

static_assert(sizeof(WaterSurfaceParamsGpu) == 96);

struct WaterSurfaceRenderPlacement {
	bool      enabled           = false;
	int32_t   mesh_index        = -1;
	glm::vec3 center            = glm::vec3(0.0f);
	float     trace_half_height = 0.45f;
	glm::vec3 axis_u            = glm::vec3(1.0f, 0.0f, 0.0f);
	float     half_extent_u     = 1.0f;
	glm::vec3 axis_v            = glm::vec3(0.0f, 0.0f, 1.0f);
	float     half_extent_v     = 1.0f;
	glm::vec3 normal            = glm::vec3(0.0f, 1.0f, 0.0f);
};

struct FloatingMeshGroup {
	std::string      display_name;
	std::vector<int> mesh_indices;
	int              simulation_index = -1;
};

struct Bounds3 {
	glm::vec3 min   = glm::vec3(0.0f);
	glm::vec3 max   = glm::vec3(0.0f);
	bool      valid = false;
};

WaterSurfaceRenderPlacement                 water_surface_render_ctx{};
std::vector<glm::mat4>                      mesh_base_transforms;
std::vector<int>                            mesh_floating_group_index;
std::vector<FloatingMeshGroup>              floating_mesh_groups;
std::vector<simulation::FloatingObjectSettings> floating_simulation_settings;
bool                                        tlas_update_pending = false;

constexpr int              kRequestedPoolWaterMeshIndex     = 98;
constexpr std::string_view kRequestedPoolWaterMeshName      = "Pool F.017 / Pool F";
constexpr float            kPoolWaterPlanarInsetWorld       = 0.08f;
constexpr float            kPoolWaterSurfaceDepthInsetWorld = 0.02f;
constexpr float            kPoolWaterTraceHalfHeightWorld   = 0.45f;

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

void create_overlay_render_pass() {
	VkAttachmentDescription color_attachment{};
	color_attachment.format         = swapchain_ctx.image_format;
	color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
	color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref{};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &color_attachment_ref;

	VkSubpassDependency dependency{};
	dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass    = 0;
	dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstAccessMask =
	    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo render_pass_info{};
	render_pass_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments    = &color_attachment;
	render_pass_info.subpassCount    = 1;
	render_pass_info.pSubpasses      = &subpass;
	render_pass_info.dependencyCount = 1;
	render_pass_info.pDependencies   = &dependency;

	if (vkCreateRenderPass(vulkan_ctx.device, &render_pass_info, nullptr,
	                       &overlay_ctx.render_pass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create overlay render pass");
	}

	overlay_ctx.framebuffers.resize(swapchain_ctx.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain_ctx.image_views.size(); ++i) {
		VkImageView attachments[] = {swapchain_ctx.image_views[i]};

		VkFramebufferCreateInfo framebuffer_info{};
		framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_info.renderPass      = overlay_ctx.render_pass;
		framebuffer_info.attachmentCount = 1;
		framebuffer_info.pAttachments    = attachments;
		framebuffer_info.width           = swapchain_ctx.extent.width;
		framebuffer_info.height          = swapchain_ctx.extent.height;
		framebuffer_info.layers          = 1;

		if (vkCreateFramebuffer(vulkan_ctx.device, &framebuffer_info, nullptr,
		                        &overlay_ctx.framebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create overlay framebuffer");
		}
	}
}

void destroy_overlay_render_resources() {
	for (VkFramebuffer framebuffer : overlay_ctx.framebuffers) {
		if (framebuffer != VK_NULL_HANDLE) {
			vkDestroyFramebuffer(vulkan_ctx.device, framebuffer, nullptr);
		}
	}
	overlay_ctx.framebuffers.clear();

	if (overlay_ctx.render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(vulkan_ctx.device, overlay_ctx.render_pass, nullptr);
		overlay_ctx.render_pass = VK_NULL_HANDLE;
	}
}

void initialize_imgui_context(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io    = ImGui::GetIO();
	io.IniFilename = nullptr;

	ImGui::StyleColorsDark();

	if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
		throw std::runtime_error("failed to initialize ImGui for Vulkan");
	}
}

void initialize_imgui_renderer() {
	ImGui_ImplVulkan_InitInfo init_info{};
	init_info.ApiVersion                   = VK_API_VERSION_1_3;
	init_info.Instance                     = vulkan_ctx.instance.instance;
	init_info.PhysicalDevice               = vulkan_ctx.phys_device.physical_device;
	init_info.Device                       = vulkan_ctx.device;
	init_info.QueueFamily                  = vulkan_ctx.graphics_queue_family;
	init_info.Queue                        = vulkan_ctx.graphics_queue;
	init_info.DescriptorPoolSize           = 16;
	init_info.MinImageCount                = static_cast<uint32_t>(swapchain_ctx.images.size());
	init_info.ImageCount                   = static_cast<uint32_t>(swapchain_ctx.images.size());
	init_info.CheckVkResultFn              = check_vk_result;
	init_info.MinAllocationSize            = 1024 * 1024;
	init_info.PipelineInfoMain.RenderPass  = overlay_ctx.render_pass;
	init_info.PipelineInfoMain.Subpass     = 0;
	init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	if (!ImGui_ImplVulkan_Init(&init_info)) {
		throw std::runtime_error("failed to initialize ImGui Vulkan backend");
	}
}

void shutdown_imgui_renderer() {
	if (ImGui::GetCurrentContext() == nullptr) {
		return;
	}

	ImGui_ImplVulkan_Shutdown();
}

void shutdown_imgui() {
	if (ImGui::GetCurrentContext() == nullptr) {
		return;
	}

	shutdown_imgui_renderer();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

audio::ReactiveAudioInputFrame buildAudioInputFrame(const audio::MicrophoneInput* microphone,
                                                    float                         time_seconds) {
	audio::ReactiveAudioInputFrame input_frame{};
	input_frame.source_available = microphone != nullptr && microphone->isAvailable();
	input_frame.raw_level        = input_frame.source_available ? microphone->latestLevel() : 0.0f;
	input_frame.time_seconds     = time_seconds;
	input_frame.source_name =
	    microphone != nullptr ? microphone->deviceName() : std::string("Unavailable");
	input_frame.source_status = microphone != nullptr ?
	                                microphone->statusMessage() :
	                                std::string("Microphone capture is unavailable.");
	return input_frame;
}

void applyOverlayLevel(float value) {
	overlay_ctx.controls.overlay.volume_level   = std::clamp(value, 0.0f, 1.0f);
	overlay_ctx.controls.overlay.selected_index = ui::quantizeSelection(
	    overlay_ctx.controls.overlay.volume_level, overlay_ctx.controls.overlay.selection_count);
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
		                      mesh->m_local_bounds_max, aabb_t_min, aabb_t_max) ||
		    aabb_t_min > best_t) {
			continue;
		}

		for (size_t index = 0; index + 2 < mesh->gpuIndices.size(); index += 3) {
			const glm::vec3& v0 = mesh->gpuVertices[mesh->gpuIndices[index + 0]].position;
			const glm::vec3& v1 = mesh->gpuVertices[mesh->gpuIndices[index + 1]].position;
			const glm::vec3& v2 = mesh->gpuVertices[mesh->gpuIndices[index + 2]].position;

			float hit_t = 0.0f;
			if (intersectRayTriangle(local_ray, v0, v1, v2, hit_t) && hit_t < best_t) {
				best_t       = hit_t;
				best_mesh_id = mesh_index;
			}
		}
	}

	return best_mesh_id;
}

bool isSwapchainRecreationResult(VkResult result) {
	return result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR;
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
	render.frame_sample_count    = static_cast<uint32_t>(frame_timing_history.sample_count);
	render.render_width          = swapchain_ctx.extent.width;
	render.render_height         = swapchain_ctx.extent.height;
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

static bool matricesNearlyEqual(const glm::mat4& a, const glm::mat4& b,
                                float epsilon = 1.0e-4f) {
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
	mesh_base_transforms.resize(mesh_count, glm::mat4(1.0f));
	mesh_floating_group_index.resize(mesh_count, -1);
}

static void writeMeshTransformToSceneAndGpu(Scene* scene, int mesh_index,
                                            const glm::mat4& transform) {
	if (scene == nullptr || mesh_index < 0 || mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[mesh_index] == nullptr) {
		return;
	}

	auto& mesh                           = scene->m_meshes[mesh_index];
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

static int waterSurfaceObjectId(const Scene* scene) {
	return (scene != nullptr && water_surface_render_ctx.enabled) ?
	           static_cast<int>(scene->m_meshes.size()) :
	           -1;
}

static int firstFloatingObjectId() {
	int first_object_id = -1;
	for (const FloatingMeshGroup& group : floating_mesh_groups) {
		for (int mesh_index : group.mesh_indices) {
			if (mesh_index < 0) {
				continue;
			}
			first_object_id =
			    first_object_id < 0 ? mesh_index : std::min(first_object_id, mesh_index);
		}
	}
	return first_object_id;
}

static int resolveWaterSurfaceMeshIndex(const Scene* scene) {
	if (scene == nullptr) {
		return -1;
	}

	const auto name_matches = [&](int mesh_index) {
		return mesh_index >= 0 && mesh_index < static_cast<int>(scene->m_meshes.size()) &&
		       scene->m_meshes[mesh_index] != nullptr &&
		       scene->m_meshes[mesh_index]->m_name == kRequestedPoolWaterMeshName;
	};

	if (name_matches(kRequestedPoolWaterMeshIndex)) {
		return kRequestedPoolWaterMeshIndex;
	}

	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		if (name_matches(mesh_index)) {
			if (mesh_index != kRequestedPoolWaterMeshIndex) {
				std::cout << "[WARN] Pool water mesh name matched object " << mesh_index
				          << " instead of requested object " << kRequestedPoolWaterMeshIndex
				          << "\n";
			}
			return mesh_index;
		}
	}

	if (kRequestedPoolWaterMeshIndex >= 0 &&
	    kRequestedPoolWaterMeshIndex < static_cast<int>(scene->m_meshes.size()) &&
	    scene->m_meshes[kRequestedPoolWaterMeshIndex] != nullptr) {
		std::cout << "[WARN] Falling back to requested pool water mesh index "
		          << kRequestedPoolWaterMeshIndex << " with unexpected name \""
		          << scene->m_meshes[kRequestedPoolWaterMeshIndex]->m_name << "\"\n";
		return kRequestedPoolWaterMeshIndex;
	}

	return -1;
}

static WaterSurfaceRenderPlacement buildWaterSurfacePlacement(const Scene* scene) {
	WaterSurfaceRenderPlacement placement{};

	const int mesh_index = resolveWaterSurfaceMeshIndex(scene);
	if (mesh_index < 0 || scene == nullptr ||
	    mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[mesh_index] == nullptr) {
		std::cout << "[WARN] Unable to resolve pool water mesh placement\n";
		return placement;
	}

	const Mesh&     mesh       = *scene->m_meshes[mesh_index];
	const glm::mat4 transform  = mesh.m_transform.m_transform;
	const glm::vec3 local_min  = mesh.m_local_bounds_min;
	const glm::vec3 local_max  = mesh.m_local_bounds_max;
	const glm::vec3 local_size = glm::max(local_max - local_min, glm::vec3(0.0f));

	int normal_axis = 0;
	if (local_size.y < local_size.x) {
		normal_axis = 1;
	}
	if (local_size.z < local_size[normal_axis]) {
		normal_axis = 2;
	}

	std::array<int, 2> surface_axes{};
	for (int axis = 0, surface_axis_count = 0; axis < 3; ++axis) {
		if (axis == normal_axis) {
			continue;
		}
		surface_axes[surface_axis_count++] = axis;
	}

	std::array<glm::vec3, 3> world_axes = {
	    glm::vec3(transform[0]),
	    glm::vec3(transform[1]),
	    glm::vec3(transform[2]),
	};
	std::array<float, 3> world_axis_scales = {
	    glm::length(world_axes[0]),
	    glm::length(world_axes[1]),
	    glm::length(world_axes[2]),
	};

	for (int axis = 0; axis < 3; ++axis) {
		if (world_axis_scales[axis] <= 1.0e-5f) {
			std::cout << "[WARN] Pool water mesh has a degenerate transform axis at object "
			          << mesh_index << "\n";
			return placement;
		}
		world_axes[axis] /= world_axis_scales[axis];
	}

	glm::vec3  normal                = world_axes[normal_axis];
	const bool normal_axis_points_up = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f)) >= 0.0f;
	if (!normal_axis_points_up) {
		normal = -normal;
	}

	glm::vec3 axis_u = world_axes[surface_axes[0]];
	glm::vec3 axis_v = world_axes[surface_axes[1]];
	if (glm::dot(glm::cross(axis_u, axis_v), normal) < 0.0f) {
		axis_v = -axis_v;
	}

	const float half_extent_u =
	    0.5f * local_size[surface_axes[0]] * world_axis_scales[surface_axes[0]] -
	    kPoolWaterPlanarInsetWorld;
	const float half_extent_v =
	    0.5f * local_size[surface_axes[1]] * world_axis_scales[surface_axes[1]] -
	    kPoolWaterPlanarInsetWorld;
	if (half_extent_u <= 1.0e-4f || half_extent_v <= 1.0e-4f) {
		std::cout << "[WARN] Pool water mesh produced invalid half-extents for object "
		          << mesh_index << "\n";
		return placement;
	}

	glm::vec3 surface_local = 0.5f * (local_min + local_max);
	surface_local[normal_axis] =
	    normal_axis_points_up ? local_max[normal_axis] : local_min[normal_axis];
	glm::vec3 center_world = transformPoint(transform, surface_local);
	center_world -= normal * kPoolWaterSurfaceDepthInsetWorld;

	placement.enabled           = true;
	placement.mesh_index        = mesh_index;
	placement.center            = center_world;
	placement.trace_half_height = kPoolWaterTraceHalfHeightWorld;
	placement.axis_u            = axis_u;
	placement.half_extent_u     = half_extent_u;
	placement.axis_v            = axis_v;
	placement.half_extent_v     = half_extent_v;
	placement.normal            = normal;
	return placement;
}

static glm::vec2 floatingAnchorForIndex(uint32_t index) {
	static constexpr std::array<glm::vec2, simulation::kMaxFloatingObjects> kAnchors = {
	    glm::vec2(-0.52f, -0.28f), glm::vec2(0.46f, -0.30f), glm::vec2(-0.18f, 0.10f),
	    glm::vec2(0.28f, 0.16f),   glm::vec2(-0.46f, 0.32f), glm::vec2(0.06f, -0.46f),
	    glm::vec2(0.54f, 0.34f),   glm::vec2(-0.08f, 0.46f),
	};
	return kAnchors[std::min<size_t>(index, kAnchors.size() - 1)];
}

static float floatingYawForIndex(uint32_t index) {
	static constexpr std::array<float, simulation::kMaxFloatingObjects> kYaws = {
	    15.0f, -24.0f, 36.0f, -48.0f, 62.0f, -80.0f, 102.0f, -128.0f,
	};
	return glm::radians(kYaws[std::min<size_t>(index, kYaws.size() - 1)]);
}

static float floatingTargetMajorWorldSize(const std::string& asset_name_lower) {
	const float pool_span_world = 2.0f * std::min(water_surface_render_ctx.half_extent_u,
	                                              water_surface_render_ctx.half_extent_v);
	if (asset_name_lower.find("duck") != std::string::npos) {
		return pool_span_world * 0.12f;
	}
	if (asset_name_lower.find("teapot") != std::string::npos) {
		return pool_span_world * 0.10f;
	}
	if (asset_name_lower.find("ring") != std::string::npos) {
		return pool_span_world * 0.18f;
	}
	return pool_span_world * 0.14f;
}

static float floatingDesiredDraftFraction(const std::string& asset_name_lower) {
	if (asset_name_lower.find("ring") != std::string::npos) {
		return 0.18f;
	}
	if (asset_name_lower.find("duck") != std::string::npos) {
		return 0.30f;
	}
	if (asset_name_lower.find("teapot") != std::string::npos) {
		return 0.56f;
	}
	return 0.28f;
}

static simulation::FloatingObjectSettings makeFloatingObjectSettings(
    const std::string& asset_name, const glm::vec3& asset_world_size, float default_scale,
    uint32_t simulation_index) {
	const std::string asset_name_lower = toLowerCopy(asset_name);
	const bool        is_ring          = asset_name_lower.find("ring") != std::string::npos;
	const bool        is_duck          = asset_name_lower.find("duck") != std::string::npos;
	const bool        is_teapot        = asset_name_lower.find("teapot") != std::string::npos;
	const glm::vec3   scaled_world_size =
	    glm::max(asset_world_size * default_scale, glm::vec3(0.03f, 0.03f, 0.03f));

	simulation::FloatingObjectSettings settings{};
	settings.anchor           = floatingAnchorForIndex(simulation_index);
	settings.base_height      = 0.02f + scaled_world_size.y * 0.10f;
	settings.base_yaw_radians = floatingYawForIndex(simulation_index);
	settings.size.x =
	    scaled_world_size.x / std::max(water_surface_render_ctx.half_extent_u, 1.0e-4f);
	settings.size.z =
	    scaled_world_size.z / std::max(water_surface_render_ctx.half_extent_v, 1.0e-4f);
	settings.size.y    = scaled_world_size.y;
	const float volume = scaled_world_size.x * scaled_world_size.y * scaled_world_size.z;
	settings.mass      = std::clamp(volume * 30.0f, 0.35f, 1.80f);
	if (is_teapot) {
		settings.mass = std::clamp(volume * 44.0f, 0.70f, 1.80f);
	}
	settings.color            = glm::vec3(0.86f, 0.58f, 0.28f);
	const float desired_draft = std::max(
	    settings.size.y * floatingDesiredDraftFraction(asset_name_lower), settings.size.y * 0.12f);
	settings.buoyancy_strength =
	    std::clamp((4.5f * settings.mass) / std::max(desired_draft * 5.0f, 1.0e-4f), 16.0f, 46.0f);
	settings.buoyancy_damping      = is_teapot ? 14.0f : 7.5f;
	settings.linear_damping        = is_teapot ? 3.4f : 1.8f;
	settings.angular_strength      = is_ring ? 9.5f : (is_teapot ? 8.0f : 12.0f);
	settings.angular_damping       = is_teapot ? 7.5f : 5.5f;
	settings.self_righting         = is_ring ? 4.5f : (is_teapot ? 4.0f : 6.5f);
	settings.max_tilt_radians      = is_ring ? 0.42f : (is_teapot ? 0.24f : 0.34f);
	settings.planar_drift_strength = is_teapot ? 1.6f : 2.4f;
	settings.planar_damping        = is_teapot ? 1.9f : 1.4f;
	settings.anchor_pull_strength  = 0.45f;
	settings.drift_radius          = is_ring ? 0.56f : (is_teapot ? 0.28f : 0.42f);
	settings.waterline_offset =
	    is_ring ? -settings.size.y * 0.08f :
	              (is_teapot ? -settings.size.y * 0.18f : settings.size.y * 0.02f);
	settings.yaw_follow_strength = is_teapot ? 1.4f : 2.2f;
	return settings;
}

static glm::mat4 makeFloatingWorldPose(const simulation::FloatingObjectSettings& settings) {
	const glm::vec3 axis_y = water_surface_render_ctx.normal;
	glm::vec3       axis_x = water_surface_render_ctx.axis_u;
	glm::vec3       axis_z = water_surface_render_ctx.axis_v;
	const float     c      = std::cos(settings.base_yaw_radians);
	const float     s      = std::sin(settings.base_yaw_radians);
	const glm::vec3 rot_x  = glm::normalize(axis_x * c + axis_z * s);
	const glm::vec3 rot_z  = glm::normalize(-axis_x * s + axis_z * c);
	const glm::vec3 center = water_surface_render_ctx.center +
	                         water_surface_render_ctx.axis_u *
	                             (settings.anchor.x * water_surface_render_ctx.half_extent_u) +
	                         water_surface_render_ctx.axis_v *
	                             (settings.anchor.y * water_surface_render_ctx.half_extent_v) +
	                         water_surface_render_ctx.normal * settings.base_height;

	glm::mat4 pose(1.0f);
	pose[0] = glm::vec4(rot_x, 0.0f);
	pose[1] = glm::vec4(axis_y, 0.0f);
	pose[2] = glm::vec4(rot_z, 0.0f);
	pose[3] = glm::vec4(center, 1.0f);
	return pose;
}

static glm::mat4 makeFloatingWorldPose(const simulation::FloatingObjectRenderData& render_data) {
	const auto to_world_axis = [](const glm::vec3& axis) {
		return glm::normalize(water_surface_render_ctx.axis_u * axis.x +
		                      water_surface_render_ctx.normal * axis.y +
		                      water_surface_render_ctx.axis_v * axis.z);
	};

	const glm::vec3 center = water_surface_render_ctx.center +
	                         water_surface_render_ctx.axis_u *
	                             (render_data.center.x * water_surface_render_ctx.half_extent_u) +
	                         water_surface_render_ctx.axis_v *
	                             (render_data.center.z * water_surface_render_ctx.half_extent_v) +
	                         water_surface_render_ctx.normal * render_data.center.y;

	glm::mat4 pose(1.0f);
	pose[0] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_x)), 0.0f);
	pose[1] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_y)), 0.0f);
	pose[2] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_z)), 0.0f);
	pose[3] = glm::vec4(center, 1.0f);
	return pose;
}

static void syncFloatingSimulationSettings(simulation::WaterSurfaceSimulation* water_surface) {
	if (water_surface == nullptr || floating_simulation_settings.empty()) {
		return;
	}
	water_surface->setFloatingObjects(floating_simulation_settings);
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
		const float       target_major_world =
		    std::max(floatingTargetMajorWorldSize(asset_name_lower), 0.08f);
		const float source_major_world = std::max(asset_world_size.x, asset_world_size.z);
		const float default_scale      = target_major_world / std::max(source_major_world, 1.0e-4f);

		FloatingMeshGroup group{};
		group.display_name     = asset_name;
		group.simulation_index = static_cast<int>(floating_simulation_settings.size());
		const simulation::FloatingObjectSettings settings =
		    makeFloatingObjectSettings(asset_name, asset_world_size, default_scale,
		                               static_cast<uint32_t>(group.simulation_index));
		floating_simulation_settings.push_back(settings);
		const glm::mat4 pose = makeFloatingWorldPose(settings);

		for (int mesh_index = mesh_start; mesh_index < mesh_end; ++mesh_index) {
			auto& mesh = scene->m_meshes[mesh_index];
			if (mesh == nullptr) {
				continue;
			}

			const glm::mat4 centered_transform =
			    glm::translate(glm::mat4(1.0f), -asset_center) * mesh->m_transform.m_transform;
			mesh_base_transforms[mesh_index] =
			    glm::scale(glm::mat4(1.0f), glm::vec3(default_scale)) * centered_transform;
			mesh_floating_group_index[mesh_index] = static_cast<int>(floating_mesh_groups.size());
			group.mesh_indices.push_back(mesh_index);
			writeMeshTransformToSceneAndGpu(scene, mesh_index, pose * mesh_base_transforms[mesh_index]);
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
	bool any_pose_changed = false;
	const auto has_valid_axis = [](const glm::vec4& axis) {
		const float length = glm::length(glm::vec3(axis));
		return std::isfinite(length) && length > 1.0e-4f;
	};

	for (const FloatingMeshGroup& group : floating_mesh_groups) {
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

		const glm::mat4 pose = makeFloatingWorldPose(object_render_data);
		for (int mesh_index : group.mesh_indices) {
			if (mesh_index < 0 || mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
			    scene->m_meshes[mesh_index] == nullptr) {
				continue;
			}

			const glm::mat4 transform = pose * mesh_base_transforms[mesh_index];
			if (matricesNearlyEqual(scene->m_meshes[mesh_index]->m_transform.m_transform, transform)) {
				continue;
			}

			writeMeshTransformToSceneAndGpu(scene, mesh_index, transform);
			any_pose_changed = true;
		}
	}

	if (any_pose_changed) {
		flushMeshTransforms(allocator);
		tlas_update_pending = true;
	}

	return any_pose_changed;
}

static void updateWaterSurfaceParamsBuffer(Scene* scene,
                                           const simulation::WaterSurfaceSimulation* water_surface) {
	if (render_target_ctx.water_surface_params_mapped == nullptr) {
		return;
	}

	WaterSurfaceParamsGpu params{};
	params.center_trace_half_height =
	    glm::vec4(water_surface_render_ctx.center, water_surface_render_ctx.trace_half_height);
	params.axis_u_half_extent =
	    glm::vec4(water_surface_render_ctx.axis_u, water_surface_render_ctx.half_extent_u);
	params.axis_v_half_extent =
	    glm::vec4(water_surface_render_ctx.axis_v, water_surface_render_ctx.half_extent_v);
	params.normal = glm::vec4(water_surface_render_ctx.normal, 0.0f);
	params.water_object_id          = waterSurfaceObjectId(scene);
	params.water_enabled            =
	    (water_surface != nullptr && water_surface_render_ctx.enabled) ? 1u : 0u;
	params.water_material_index     = -1;
	params.water_height_to_world_scale =
	    water_surface != nullptr ? water_surface->heightToWorldScale() : 1.0f;
	params.first_floating_object_id = firstFloatingObjectId();

	std::memcpy(render_target_ctx.water_surface_params_mapped, &params, sizeof(params));
	vmaFlushAllocation(render_target_ctx.allocator, render_target_ctx.water_surface_params_alloc, 0,
	                   sizeof(params));
}

static void updateWaterSurfaceImageDescriptors(
    const simulation::WaterSurfaceSimulation* water_surface) {
	if (render_target_ctx.descriptor_set == VK_NULL_HANDLE || water_surface == nullptr) {
		return;
	}

	VkDescriptorImageInfo current_height_info{};
	current_height_info.imageView   = water_surface->currentHeightImageView();
	current_height_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo previous_height_info{};
	previous_height_info.imageView   = water_surface->previousHeightImageView();
	previous_height_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet writes[2]{};
	writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	             nullptr,
	             render_target_ctx.descriptor_set,
	             20,
	             0,
	             1,
	             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	             &current_height_info};
	writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	             nullptr,
	             render_target_ctx.descriptor_set,
	             21,
	             0,
	             1,
	             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	             &previous_height_info};
	vkUpdateDescriptorSets(vulkan_ctx.device, 2, writes, 0, nullptr);
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

	create_overlay_render_pass();

	m_water_surface =
	    std::make_unique<simulation::WaterSurfaceSimulation>(simulation::WaterSurfaceCreateInfo{
	        .device        = vulkan_ctx.device,
	        .allocator     = render_target_ctx.allocator,
	        .output_extent = swapchain_ctx.extent,
	    });
	syncFloatingSimulationSettings(m_water_surface.get());
	m_water_surface->requestObjectReset();
	std::cout << "[INFO] Created water surface simulation resources\n";
}

void Runtime::destroySwapchainResources() {
	m_water_surface.reset();
	destroy_overlay_render_resources();

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

void Runtime::recreateSwapchainResources() {
	uint32_t framebuffer_width  = m_window->width();
	uint32_t framebuffer_height = m_window->height();
	while (framebuffer_width == 0 || framebuffer_height == 0) {
		m_window->waitEvents();
		framebuffer_width  = m_window->width();
		framebuffer_height = m_window->height();
	}

	check_vk_result(vkDeviceWaitIdle(vulkan_ctx.device));
	shutdown_imgui_renderer();
	destroySwapchainResources();
	createSwapchainResources();
	initialize_imgui_renderer();

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

	// Recreate pathtracer storage/accum/object_id images at the new swapchain extent
	VmaAllocator allocator = render_target_ctx.allocator;
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.storage_image,
	                render_target_ctx.storage_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.object_id_image,
	                render_target_ctx.object_id_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);

	const VkExtent3D new_ext = {swapchain_ctx.extent.width, swapchain_ctx.extent.height, 1};

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
	             VK_FORMAT_R32_SINT, 0);
	make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
	             VK_FORMAT_R8G8B8A8_UNORM, 0);

	render_target_ctx.storage_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.storage_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.object_id_image_view =
	    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image, VK_FORMAT_R32_SINT,
	                      VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.accum_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.accum_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

	// Transition object_id and accum → GENERAL so the shader can read/write them
	{
		VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
		transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
		end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool, vulkan_ctx.graphics_queue,
		                 cmd);
	}

	// Update descriptor set bindings 0 (output), 1 (accum), 13 (object_id)
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
		VkWriteDescriptorSet writes[3]{};
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
		vkUpdateDescriptorSets(vulkan_ctx.device, 3, writes, 0, nullptr);
	}
	updateWaterSurfaceImageDescriptors(m_water_surface.get());
	updateWaterSurfaceParamsBuffer(m_scene.get(), m_water_surface.get());

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
	VkAccelerationStructureKHR tlas              = VK_NULL_HANDLE;
	VkBuffer                   tlas_buffer       = VK_NULL_HANDLE;
	VmaAllocation              tlas_buffer_alloc = VK_NULL_HANDLE;
	VkBuffer                   instance_buffer       = VK_NULL_HANDLE;
	VmaAllocation              instance_buffer_alloc = VK_NULL_HANDLE;
	void*                      instance_mapped       = nullptr;
	VkBuffer                   scratch_buffer       = VK_NULL_HANDLE;
	VmaAllocation              scratch_buffer_alloc = VK_NULL_HANDLE;
	VkDeviceSize               scratch_size         = 0;
	uint32_t                   instance_count       = 0;
	VkDeviceAddress            tlas_address      = 0;
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
    vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
    vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);

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
    make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
	                 VK_FORMAT_R8G8B8A8_UNORM, 0);

    render_target_ctx.storage_image_view =
        create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
	                          VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
    render_target_ctx.object_id_image_view =
        create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image, VK_FORMAT_R32_SINT,
	                          VK_IMAGE_VIEW_TYPE_2D);
    render_target_ctx.accum_image_view =
        create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
	                          VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);

    // Transition accum → GENERAL
    {
        VkCommandBuffer cmd = begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
        transition_layout(cmd, render_target_ctx.object_id_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
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

        VkWriteDescriptorSet writes[3]{};
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
        vkUpdateDescriptorSets(vulkan_ctx.device, 3, writes, 0, nullptr);
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
    for (VkFramebuffer fb : overlay_ctx.framebuffers)
        vkDestroyFramebuffer(vulkan_ctx.device, fb, nullptr);
    overlay_ctx.framebuffers.clear();
    overlay_ctx.framebuffers.resize(swapchain_ctx.image_views.size(), VK_NULL_HANDLE);
    for (size_t i = 0; i < swapchain_ctx.image_views.size(); ++i) {
        VkImageView             attachments[] = {swapchain_ctx.image_views[i]};
        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass      = overlay_ctx.render_pass;
        framebuffer_info.attachmentCount = 1;
        framebuffer_info.pAttachments    = attachments;
        framebuffer_info.width           = swapchain_ctx.extent.width;
        framebuffer_info.height          = swapchain_ctx.extent.height;
        framebuffer_info.layers          = 1;
        vkCreateFramebuffer(vulkan_ctx.device, &framebuffer_info, nullptr,
		                        &overlay_ctx.framebuffers[i]);
    }

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

// ============================================================
// === Shader compilation
// ============================================================
static std::vector<uint32_t>
    compile_slang_shader(const std::string& path, const std::string& entry_point,
                         const std::vector<std::string>& search_paths = {}) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* req     = spCreateCompileRequest(session);
	for (const auto& sp : search_paths)
		spAddSearchPath(req, sp.c_str());
	int ti = spAddCodeGenTarget(req, SLANG_SPIRV);
	spSetTargetProfile(req, ti, spFindProfile(session, "spirv_1_4"));
	int ui = spAddTranslationUnit(req, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(req, ui, path.c_str());
	spAddEntryPoint(req, ui, entry_point.c_str(), SLANG_STAGE_COMPUTE);
	SlangResult res  = spCompile(req);
	const char* diag = spGetDiagnosticOutput(req);
	if (diag && diag[0] != '\0')
		std::cerr << "[SLANG] " << path << ":\n" << diag << "\n";
	if (res != SLANG_OK) {
		spDestroyCompileRequest(req);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + path);
	}
	size_t                sz   = 0;
	const void*           data = spGetEntryPointCode(req, 0, &sz);
	std::vector<uint32_t> spirv(sz / sizeof(uint32_t));
	memcpy(spirv.data(), data, sz);
	spDestroyCompileRequest(req);
	spDestroySession(session);
	return spirv;
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
		spirv = compile_slang_shader(shader_path, "main", {VENDORS_DIR});
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

	as_ctx.tlas_buffer = create_gpu_buffer(alloc, sizes.accelerationStructureSize, AS_BUFFER_USAGE,
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

// =======================
// === App constructor ===
// =======================
static std::string toLowerAscii(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

static std::string resolveScenePathOrThrow(const std::string& scene_argument) {
	namespace fs = std::filesystem;

	if (scene_argument.empty()) {
		return "resources/scenes/cornell/cornell.glb";
	}

	const std::string scene_key = toLowerAscii(scene_argument);
	if (scene_key == "pool") {
		return "resources/scenes/poolHouse/poolHouse_optimized.glb";
	}
	if (scene_key == "chess") {
		return "resources/scenes/ABeautifulGame/glTF-Binary/ABeautifulGame.glb";
	}
	if (scene_key == "cornell") {
		return "resources/scenes/cornell/cornell.glb";
	}
	if (scene_key == "cornellsimple") {
		return "resources/scenes/cornell/cornell_simple.glb";
	}
	if (scene_key == "sponza") {
		return "resources/scenes/Sponza/glTF/Sponza.gltf";
	}

	const fs::path    user_path(scene_argument);
	const std::string ext = toLowerAscii(user_path.extension().string());
	if (ext != ".gltf" && ext != ".glb") {
		throw std::runtime_error(
		    "unknown scene alias '" + scene_argument +
		    "'. Use one of: pool, chess, cornell, cornellsimple, sponza, or provide a .gltf/.glb "
		    "file path.");
	}

	std::error_code ec;
	if (!fs::exists(user_path, ec) || ec) {
		throw std::runtime_error("scene file does not exist: " + user_path.string());
	}
	if (!fs::is_regular_file(user_path, ec) || ec) {
		throw std::runtime_error("scene path is not a regular file: " + user_path.string());
	}

	return user_path.lexically_normal().string();
}

Runtime::Runtime(const std::string& scene_argument) {
	// ==============================
	// === 0. Scene setup
	// ==============================
	m_scene                      = std::make_unique<Scene>();
	m_scene->m_camera            = Camera(glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.f, 0.f, 0.f),
	                                      glm::vec3(0.f, 1.f, 0.f), 60.f, 0.1f, 10000.f);
	const std::string scene_path = resolveScenePathOrThrow(scene_argument);
	std::cout << "[INFO] Loading scene: " << scene_path << "\n";
	m_scene->load_gltf(scene_path);
	if (m_scene->m_meshes.empty()) {
		throw std::runtime_error("failed to load scene: " + scene_path);
	}
	water_surface_render_ctx      = {};
	mesh_base_transforms.clear();
	mesh_floating_group_index.clear();
	floating_mesh_groups.clear();
	floating_simulation_settings.clear();
	tlas_update_pending = false;
	water_surface_render_ctx = buildWaterSurfacePlacement(m_scene.get());
	ensureMeshTransformMetadataSize(m_scene->m_meshes.size());
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
		VkExtent3D ext = {(uint32_t) m_window->width(), (uint32_t) m_window->height(), 1};
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
		make_storage_image(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
		                   VK_FORMAT_R8G8B8A8_UNORM, 0);
		render_target_ctx.storage_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
		                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.object_id_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image,
		                      VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.accum_image_view =
		    create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
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
		cam_buf_info.size  = sizeof(GPUCamera);
		cam_buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationInfo cam_info;
		vmaCreateBuffer(allocator, &cam_buf_info, &cam_alloc_info, &scene_ctx.camera_buffer,
		                &scene_ctx.camera_alloc, &cam_info);
		scene_ctx.camera_mapped = cam_info.pMappedData;
		GPUCamera initial_cam   = m_scene->m_camera.pack();
		memcpy(scene_ctx.camera_mapped, &initial_cam, sizeof(GPUCamera));
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
		WaterSurfaceParamsGpu initial_water_params{};
		render_target_ctx.water_surface_params_buffer =
		    create_and_upload_buffer(allocator, sizeof(initial_water_params),
		                             &initial_water_params, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		                             render_target_ctx.water_surface_params_alloc);
		VmaAllocationInfo water_params_info{};
		vmaGetAllocationInfo(allocator, render_target_ctx.water_surface_params_alloc,
		                     &water_params_info);
		render_target_ctx.water_surface_params_mapped = water_params_info.pMappedData;
		if (render_target_ctx.water_surface_params_mapped == nullptr &&
		    vmaMapMemory(allocator, render_target_ctx.water_surface_params_alloc,
		                 &render_target_ctx.water_surface_params_mapped) != VK_SUCCESS) {
			throw std::runtime_error("failed to map water surface params buffer");
		}
		updateWaterSurfaceParamsBuffer(m_scene.get(), m_water_surface.get());
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
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},
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
		updateWaterSurfaceImageDescriptors(m_water_surface.get());
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

	initialize_imgui_context(m_window->handle());
	initialize_imgui_renderer();
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
		transition_layout(cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_UNDEFINED,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
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

	shutdown_imgui();
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
	MainLoop();
}
void Runtime::MainLoop() {
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

	float last_frame_time = static_cast<float>(glfwGetTime());

	// Initialise fly camera from the scene camera
	FlyCamera fly_cam(m_scene->m_camera.m_position, m_scene->m_camera.m_target,
	                  m_scene->m_camera.m_fov, 0.5f);
	ui::selection_ctx.camera.fov_deg = fly_cam.m_fov;
	float last_camera_fov            = ui::selection_ctx.camera.fov_deg;

	double   last_time    = glfwGetTime();
	uint32_t frame_number = 0;

	// Visibility pass must run before the first path-trace dispatch and
	// whenever the camera moves, resizes, or a new object is selected.
	bool needs_visibility_pass = true;

	// Tracks active material drag/edit interactions from the selection panel.
	bool                    material_edit_mode     = false;
	bool                    camera_was_moving      = false;
	bool                    hipr_force_clear_order = true;
	ui::RenderDebugViewMode last_render_mode       = ui::selection_ctx.debug_view_mode;
	bool                    show_all_gui           = true;
	bool                    show_selection_panel   = true;

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
			last_frame_time = static_cast<float>(glfwGetTime());
			continue;
		}

		if (swapchain_ctx.extent.width != framebuffer_width ||
		    swapchain_ctx.extent.height != framebuffer_height) {
			recreateSwapchainResources();
			frame_number          = 0;
			needs_visibility_pass = true;
			reset_hipr_object_sampling();
		}

		const float time_seconds = static_cast<float>(glfwGetTime());
		const float delta_time   = std::max(time_seconds - last_frame_time, 1.0f / 240.0f);
		last_frame_time          = time_seconds;

		updateRenderDiagnostics(delta_time);

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		const audio::ReactiveAudioInputFrame audio_input =
		    buildAudioInputFrame(m_microphone.get(), time_seconds);

		const auto update_water_and_floaters = [&](float water_audio_level) {
			if (m_water_surface != nullptr) {
				overlay_ctx.diagnostics.water = m_water_surface->prepareFrame(
				    overlay_ctx.controls.water, water_audio_level, time_seconds, delta_time);
			}
		};

		float audio_level = 0.0f;
		if (m_audio_controller != nullptr) {
			audio_level = m_audio_controller->update(overlay_ctx.controls.audio, audio_input);
			overlay_ctx.diagnostics.audio = m_audio_controller->diagnostics();
		}
		const float water_audio_level = overlay_ctx.diagnostics.audio.normalized_level;
		applyOverlayLevel(audio_level);
		update_water_and_floaters(water_audio_level);

		if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) {
			show_all_gui = !show_all_gui;
			if (show_all_gui) {
				overlay_ctx.show_control_panel = true;
				show_selection_panel           = true;
			}
		}

		bool controls_changed = false;
		if (show_all_gui && overlay_ctx.show_control_panel) {
			controls_changed = ui::drawAudienceControlPanel(
			    &overlay_ctx.show_control_panel, overlay_ctx.controls, overlay_ctx.diagnostics);
		}

		ui::SelectionPanelResult selection_panel_result{};
		if (show_all_gui && show_selection_panel) {
			selection_panel_result = ui::drawSelectionPanel(m_scene.get(), &show_selection_panel);
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
		}

		if (ui::selection_ctx.debug_view_mode != last_render_mode) {
			frame_number           = 0;
			needs_visibility_pass  = true;
			hipr_force_clear_order = true;
			last_render_mode       = ui::selection_ctx.debug_view_mode;
			reset_hipr_object_sampling();
		}

		if (controls_changed) {
			if (m_audio_controller != nullptr) {
				audio_level = m_audio_controller->update(overlay_ctx.controls.audio, audio_input);
				overlay_ctx.diagnostics.audio = m_audio_controller->diagnostics();
			}
			const float updated_water_audio_level = overlay_ctx.diagnostics.audio.normalized_level;
			applyOverlayLevel(audio_level);
			update_water_and_floaters(updated_water_audio_level);
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
		updateWaterSurfaceParamsBuffer(m_scene.get(), m_water_surface.get());

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
			frame_number = 0;
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
		const GPUCamera gpu_camera = fly_cam.pack();

		const int current_lmb = glfwGetMouseButton(m_window->handle(), GLFW_MOUSE_BUTTON_LEFT);
		if (current_lmb == GLFW_PRESS && prev_lmb == GLFW_RELEASE && !fly_cam.isMouseCaptured() &&
		    !ImGui::GetIO().WantCaptureMouse) {
			if (ui::selectMesh(m_scene.get(),
			                   pickMeshAtCursor(m_scene.get(), m_window->handle(), gpu_camera,
			                                    framebuffer_width, framebuffer_height))) {
				frame_number          = 0;
				needs_visibility_pass = true;
				material_edit_mode    = false;
				reset_hipr_object_sampling();
			}
		}
		prev_lmb = current_lmb;

		memcpy(scene_ctx.camera_mapped, &gpu_camera, sizeof(GPUCamera));

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
			frame_number = 0;
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

		updateFloatingMeshTransformsFromSimulation(m_scene.get(), m_water_surface.get(),
		                                           render_target_ctx.allocator);
		if (tlas_update_pending) {
			if (as_ctx.tlas != VK_NULL_HANDLE) {
				upload_tlas_instances(render_target_ctx.allocator, as_ctx.blases,
				                      m_scene->m_meshes);
				record_tlas_build(vulkan_ctx.device, cmd, true);
			}
			tlas_update_pending = false;
		}
		if (m_water_surface != nullptr) {
			m_water_surface->record(cmd);
			updateWaterSurfaceImageDescriptors(m_water_surface.get());
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

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, active_pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline_layout, 0,
		                        1, &render_target_ctx.descriptor_set, 0, nullptr);

		if (needs_visibility_pass) {
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

		const uint32_t dispatch_w = (swapchain_ctx.extent.width + 15) / 16;
		const uint32_t dispatch_h = (swapchain_ctx.extent.height + 15) / 16;

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
				pc.hipr_reserved0         = 0u;
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
		// - do not periodically resort every N frames
		bool hipr_refresh = use_hipr_ranked && needs_visibility_pass;
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
			pc.frame            = hipr_object_sampling_enabled ?
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
		blit_region.srcOffsets[1] = {static_cast<int32_t>(swapchain_ctx.extent.width),
		                             static_cast<int32_t>(swapchain_ctx.extent.height), 1};

		blit_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		blit_region.dstSubresource.mipLevel       = 0;
		blit_region.dstSubresource.baseArrayLayer = 0;
		blit_region.dstSubresource.layerCount     = 1;
		blit_region.dstOffsets[0]                 = {0, 0, 0};
		blit_region.dstOffsets[1] = {static_cast<int32_t>(swapchain_ctx.extent.width),
		                             static_cast<int32_t>(swapchain_ctx.extent.height), 1};

		vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		               &blit_region, VK_FILTER_NEAREST);

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
			frame_number = 0;
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
