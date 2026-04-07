#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "VkBootstrap.h"
#include "tsunami/camera/fly_camera.h"
#include "vk_mem_alloc.h"

#include "slang.h"
#include "tsunami/app/app.h"

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
#include <glm/gtc/type_ptr.hpp>
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

// =======================
// === Context structs ===
// =======================
struct SceneContext {
	void*         camera_mapped   = nullptr;
	VkBuffer      camera_buffer   = VK_NULL_HANDLE;
	VmaAllocation camera_alloc    = VK_NULL_HANDLE;
	void*         material_mapped = nullptr;
	VkBuffer      material_buffer = VK_NULL_HANDLE;
	VmaAllocation material_alloc  = VK_NULL_HANDLE;
	uint32_t      material_count  = 0;
	VkBuffer      mesh_buffer     = VK_NULL_HANDLE;
	VmaAllocation mesh_alloc      = VK_NULL_HANDLE;
	VkBuffer      vertex_buffer   = VK_NULL_HANDLE;
	VmaAllocation vertex_alloc    = VK_NULL_HANDLE;
	VkBuffer      index_buffer    = VK_NULL_HANDLE;
	VmaAllocation index_alloc     = VK_NULL_HANDLE;
	uint32_t      mesh_count      = 0;
} scene_ctx;
#include "tsunami/audio/microphone_input.h"
#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"
#include "tsunami/ui/audience_overlay.h"

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
	VkDescriptorSetLayout              descriptor_set_layout;
	VkDescriptorPool                   descriptor_pool;
	VkDescriptorSet                    descriptor_set;
	// tile probe buffers
	VkBuffer      tile_buffer       = VK_NULL_HANDLE;
	VmaAllocation tile_buffer_alloc = VK_NULL_HANDLE;
	void*    tile_buffer_mapped     = nullptr;        // CPU pointer to read back tile data from GPU
	VkBuffer tile_render_flags_buffer            = VK_NULL_HANDLE;
	VmaAllocation tile_render_flags_buffer_alloc = VK_NULL_HANDLE;
	void*         tile_render_flags_mapped       = nullptr;
	uint32_t      tile_count                     = 0;
	// debug: store image extents used for created images
	uint32_t storage_width = 0, storage_height = 0;
	uint32_t object_id_width = 0, object_id_height = 0;
	uint32_t accum_width = 0, accum_height = 0;
} render_target_ctx;

struct ComputePipelineContext {
	VkPipelineLayout pipeline_layout;
	VkPipeline       pipeline;
} compute_ctx;

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

enum class MaterialEditMode : int {
	Gui   = 0,
	Voice = 1,
};

enum class RenderDebugViewMode : int {
	Beauty    = 0,
	ObjectIds = 1,
};

struct PathTracerPushConstants {
	uint32_t  frame               = 0;
	uint32_t  material_count      = 0;
	int32_t   selected_mesh_index = -1;
	uint32_t  outline_width       = 1;
	int32_t   debug_view_mode     = 0;
	uint32_t  stage               = 0;        // was _pad0
	uint32_t  _pad1               = 0;
	uint32_t  _pad2               = 0;
	glm::vec4 outline_color       = glm::vec4(1.f, 0.65f, 0.15f, 1.f);
};
static_assert(sizeof(PathTracerPushConstants) == 48);

struct ObjectIdEntry {
	int         object_id = -1;
	std::string display_name;
	int         mesh_index     = -1;
	int         material_index = -1;
};

struct SelectionContext {
	int                        selected_mesh_index = -1;
	MaterialEditMode           material_edit_mode  = MaterialEditMode::Gui;
	RenderDebugViewMode        debug_view_mode     = RenderDebugViewMode::Beauty;
	GPUMaterial                editor_material{};
	glm::vec4                  outline_color = glm::vec4(1.0f, 0.65f, 0.15f, 1.0f);
	uint32_t                   outline_width = 1;
	std::vector<ObjectIdEntry> object_id_map;

	SelectionContext() {
		editor_material = Material{}.pack();
	}
} selection_ctx{};

struct SelectionPanelResult {
	bool selection_changed = false;
	bool material_changed  = false;
};

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

// Tile structs for HiPR ===================
struct TileData {
	int32_t primary_object  = -1;
	int32_t next_objects[3] = {-1, -1, -1};
};

static constexpr int BFS_MAX_DEPTH              = 2;
static constexpr int BACKGROUND_TILES_PER_FRAME = 8;        // tune to taste

struct ProbeContext {
	std::vector<TileData> tiles;        // CPU readback of tile_buffer
	bool                  valid             = false;
	int                   last_selected     = -2;
	uint32_t              background_cursor = 0;        // round-robin background pass
	bool hipr_enabled = false;        // 	set to true to enable HiPR tile probing and selective
	                                  // rendering based on probe results
	bool pause_background = false;
	bool show_tile_debug  = false;
} probe_ctx;
// =======================================

void check_vk_result(VkResult result) {
	if (result == VK_SUCCESS) {
		return;
	}

	std::cerr << "[Vulkan] VkResult = " << result << "\n";
	if (result < 0) {
		throw std::runtime_error("Vulkan call failed");
	}
}

// Cleanup function to destroy tile buffers when the render target is resized or destroyed
static void destroy_tile_buffers(VmaAllocator alloc) {
    if (render_target_ctx.tile_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(alloc, render_target_ctx.tile_buffer,
                         render_target_ctx.tile_buffer_alloc);
        render_target_ctx.tile_buffer = VK_NULL_HANDLE;
        render_target_ctx.tile_buffer_alloc = VK_NULL_HANDLE;
        render_target_ctx.tile_buffer_mapped = nullptr;
    }
    if (render_target_ctx.tile_render_flags_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(alloc, render_target_ctx.tile_render_flags_buffer,
                         render_target_ctx.tile_render_flags_buffer_alloc);
        render_target_ctx.tile_render_flags_buffer = VK_NULL_HANDLE;
        render_target_ctx.tile_render_flags_buffer_alloc = VK_NULL_HANDLE;
        render_target_ctx.tile_render_flags_mapped = nullptr;
    }
    render_target_ctx.tile_count = 0;
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

std::string meshDisplayName(const Scene* scene, int mesh_index) {
	if (scene == nullptr || mesh_index < 0 ||
	    mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		return "None";
	}

	const Mesh* mesh = scene->m_meshes[mesh_index].get();
	if (mesh == nullptr || mesh->m_name.empty()) {
		return "Mesh " + std::to_string(mesh_index);
	}

	return mesh->m_name;
}

void rebuildObjectIdMap(const Scene* scene) {
	selection_ctx.object_id_map.clear();
	if (scene == nullptr) {
		return;
	}

	selection_ctx.object_id_map.reserve(scene->m_meshes.size());
	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto&   mesh = scene->m_meshes[mesh_index];
		ObjectIdEntry entry{};
		entry.object_id      = mesh_index;
		entry.mesh_index     = mesh_index;
		entry.material_index = mesh_index;
		entry.display_name   = meshDisplayName(scene, mesh_index);
		if (mesh == nullptr || mesh->m_material == nullptr) {
			entry.material_index = -1;
		}
		selection_ctx.object_id_map.push_back(std::move(entry));
	}
}

const ObjectIdEntry* objectIdEntryForId(int object_id) {
	if (object_id < 0 || object_id >= static_cast<int>(selection_ctx.object_id_map.size())) {
		return nullptr;
	}
	return &selection_ctx.object_id_map[object_id];
}

void refreshSelectedMaterialEditor(const Scene* scene) {
	if (scene == nullptr || selection_ctx.selected_mesh_index < 0 ||
	    selection_ctx.selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		selection_ctx.editor_material = Material{}.pack();
		return;
	}

	const auto& mesh              = scene->m_meshes[selection_ctx.selected_mesh_index];
	selection_ctx.editor_material = (mesh != nullptr && mesh->m_material != nullptr) ?
	                                    mesh->m_material->pack() :
	                                    Material{}.pack();
}

bool selectMesh(const Scene* scene, int mesh_index) {
	const int max_mesh_index =
	    (scene != nullptr) ? static_cast<int>(scene->m_meshes.size()) - 1 : -1;
	const int clamped_index = (mesh_index >= 0 && mesh_index <= max_mesh_index) ? mesh_index : -1;
	if (selection_ctx.selected_mesh_index == clamped_index) {
		return false;
	}

	selection_ctx.selected_mesh_index = clamped_index;
	std::cout << "[DEBUG] selectMesh -> selected_mesh_index = " << selection_ctx.selected_mesh_index << "\n";
	refreshSelectedMaterialEditor(scene);
	return true;
}

void updateMaterialBufferSlot(VmaAllocator allocator, int material_index,
                              const GPUMaterial& material) {
	if (scene_ctx.material_mapped == nullptr || material_index < 0 ||
	    material_index >= static_cast<int>(scene_ctx.material_count)) {
		return;
	}

	auto* gpu_materials           = reinterpret_cast<GPUMaterial*>(scene_ctx.material_mapped);
	gpu_materials[material_index] = material;
	vmaFlushAllocation(allocator, scene_ctx.material_alloc,
	                   static_cast<VkDeviceSize>(material_index) * sizeof(GPUMaterial),
	                   sizeof(GPUMaterial));
}

void applySelectedMaterialEditor(Scene* scene, VmaAllocator allocator) {
	if (scene == nullptr || selection_ctx.selected_mesh_index < 0 ||
	    selection_ctx.selected_mesh_index >= static_cast<int>(scene->m_meshes.size())) {
		return;
	}

	auto& mesh = scene->m_meshes[selection_ctx.selected_mesh_index];
	if (mesh == nullptr || mesh->m_material == nullptr) {
		return;
	}

	mesh->m_material->m_gpu = selection_ctx.editor_material;
	updateMaterialBufferSlot(allocator, selection_ctx.selected_mesh_index,
	                         selection_ctx.editor_material);
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

// Given the CPU tile map and a frontier set, return the set of tiles whose
// primary object is in the frontier, then extend the frontier to their neighbors.
static void bfs_compute_tile_flags(const std::vector<TileData>& tiles, uint32_t tile_count,
                                   int seed_object,        // -1 = render everything
                                   std::vector<uint32_t>& out_flags)        // sized to tile_count
{
	out_flags.assign(tile_count, 0u);
	if (seed_object < 0) {
		// No selection: render everything
		std::fill(out_flags.begin(), out_flags.end(), 1u);
		return;
	}

	std::unordered_set<int> frontier = {seed_object};
	std::unordered_set<int> visited;

	for (int depth = 0; depth < BFS_MAX_DEPTH && !frontier.empty(); ++depth) {
		std::unordered_set<int> next_frontier;

		for (uint32_t ti = 0; ti < tile_count; ++ti) {
			if (frontier.count(tiles[ti].primary_object)) {
				out_flags[ti] = 1u;
				for (int k = 0; k < 3; ++k) {
					int obj = tiles[ti].next_objects[k];
					if (obj >= 0 && !visited.count(obj))
						next_frontier.insert(obj);
				}
			}
		}

		visited.insert(frontier.begin(), frontier.end());
		frontier = std::move(next_frontier);
		// Remove already-visited from next
		for (auto it = frontier.begin(); it != frontier.end();) {
			if (visited.count(*it))
				it = frontier.erase(it);
			else
				++it;
		}
	}
}

SelectionPanelResult drawSelectionPanel(const Scene* scene) {
	SelectionPanelResult result{};

	ImGui::SetNextWindowPos(ImVec2(470.0f, 24.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 320.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Object Inspector")) {
		ImGui::End();
		return result;
	}

	ImGui::TextUnformatted("Click the render view to select a mesh.");

	int         edit_mode  = static_cast<int>(selection_ctx.material_edit_mode);
	const char* edit_items = "GUI\0Voice\0";
	if (ImGui::Combo("Material input", &edit_mode, edit_items)) {
		selection_ctx.material_edit_mode = static_cast<MaterialEditMode>(edit_mode);
	}

	int         debug_view_mode  = static_cast<int>(selection_ctx.debug_view_mode);
	const char* debug_view_items = "Beauty\0Object IDs\0";
	if (ImGui::Combo("Renderer view", &debug_view_mode, debug_view_items)) {
		selection_ctx.debug_view_mode = static_cast<RenderDebugViewMode>(debug_view_mode);
	}

	ImGui::Text("Scene objects: %d",
	            scene != nullptr ? static_cast<int>(scene->m_meshes.size()) : 0);
	ImGui::Text("Object IDs: %d", static_cast<int>(selection_ctx.object_id_map.size()));
	ImGui::Text("Selected mesh: %s",
	            meshDisplayName(scene, selection_ctx.selected_mesh_index).c_str());

	const bool has_selection =
	    scene != nullptr && selection_ctx.selected_mesh_index >= 0 &&
	    selection_ctx.selected_mesh_index < static_cast<int>(scene->m_meshes.size());

	if (has_selection) {
		const ObjectIdEntry* selected_entry = objectIdEntryForId(selection_ctx.selected_mesh_index);
		ImGui::Text("Object ID: %d", selected_entry != nullptr ? selected_entry->object_id : -1);
		ImGui::Text("Mesh index: %d", selection_ctx.selected_mesh_index);
		if (ImGui::Button("Clear selection")) {
			result.selection_changed = selectMesh(scene, -1);
		}
	} else {
		ImGui::TextUnformatted("No mesh selected.");
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Selection outline");
	int outline_width = static_cast<int>(selection_ctx.outline_width);
	if (ImGui::SliderInt("Outline width", &outline_width, 1, 4)) {
		selection_ctx.outline_width = static_cast<uint32_t>(outline_width);
	}
	ImGui::ColorEdit4("Outline color", glm::value_ptr(selection_ctx.outline_color),
	                  ImGuiColorEditFlags_AlphaBar);

	if (ImGui::CollapsingHeader("Object ID Map")) {
		for (const ObjectIdEntry& entry : selection_ctx.object_id_map) {
			ImGui::Text("ID %d -> %s", entry.object_id, entry.display_name.c_str());
		}
	}

	if (!has_selection) {
		if (selection_ctx.material_edit_mode == MaterialEditMode::Voice) {
			ImGui::Separator();
			ImGui::TextWrapped(
			    "Voice mode is selected, but there is not yet a speech-to-text command layer for "
			    "material edits in this project.");
		}
		ImGui::End();
		return result;
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Material");
	ImGui::TextWrapped(
	    "Texture-backed meshes use these controls as live multipliers and overrides.");

	const bool gui_mode_enabled = selection_ctx.material_edit_mode == MaterialEditMode::Gui;
	ImGui::BeginDisabled(!gui_mode_enabled);
	result.material_changed |=
	    ImGui::ColorEdit3("Base tint", glm::value_ptr(selection_ctx.editor_material.base_color));
	result.material_changed |= ImGui::SliderFloat(
	    "Opacity", &selection_ctx.editor_material.geometry_opacity, 0.0f, 1.0f, "%.2f");
	result.material_changed |= ImGui::SliderFloat(
	    "Metalness", &selection_ctx.editor_material.base_metalness, 0.0f, 1.0f, "%.2f");
	result.material_changed |= ImGui::SliderFloat(
	    "Roughness", &selection_ctx.editor_material.specular_roughness, 0.02f, 1.0f, "%.2f");
	result.material_changed |= ImGui::SliderFloat(
	    "Transmission", &selection_ctx.editor_material.transmission_weight, 0.0f, 1.0f, "%.2f");
	result.material_changed |=
	    ImGui::SliderFloat("IOR", &selection_ctx.editor_material.specular_ior, 1.0f, 2.5f, "%.2f");
	result.material_changed |= ImGui::ColorEdit3(
	    "Emission color", glm::value_ptr(selection_ctx.editor_material.emission_color));
	result.material_changed |=
	    ImGui::SliderFloat("Emission intensity", &selection_ctx.editor_material.emission_luminance,
	                       0.0f, 20.0f, "%.2f");
	bool thin_walled = selection_ctx.editor_material.geometry_thin_walled > 0.5f;
	if (ImGui::Checkbox("Thin walled", &thin_walled)) {
		selection_ctx.editor_material.geometry_thin_walled = thin_walled ? 1.0f : 0.0f;
		result.material_changed                            = true;
	}
	ImGui::EndDisabled();

	if (!gui_mode_enabled) {
		ImGui::TextWrapped(
		    "Voice mode is selected, but there is not yet a speech-to-text command layer for "
		    "material edits in this project.");
	}

	// ── HiPR probe debug ─────────────────────────────────────────────────────
	if (ImGui::CollapsingHeader("HiPR Probe")) {
		ImGui::Text("Probe valid: %s", probe_ctx.valid ? "yes" : "no");
		ImGui::Text("Last selected: %d", probe_ctx.last_selected);
		ImGui::Text("Tile count: %u", render_target_ctx.tile_count);
		ImGui::Text("BG cursor: %u / %u", probe_ctx.background_cursor,
		            render_target_ctx.tile_count);

		if (probe_ctx.valid && !probe_ctx.tiles.empty()) {
			// Count how many tiles are currently flagged
			uint32_t flagged = 0;
			if (render_target_ctx.tile_render_flags_mapped != nullptr) {
				const auto* f =
				    reinterpret_cast<const uint32_t*>(render_target_ctx.tile_render_flags_mapped);
				for (uint32_t i = 0; i < render_target_ctx.tile_count; ++i)
					if (f[i])
						++flagged;
			}
			ImGui::Text("Flagged tiles: %u / %u (%.1f%%)", flagged, render_target_ctx.tile_count,
			            100.f * flagged / std::max(render_target_ctx.tile_count, 1u));

			// Show the selected object's tile and its neighbors
			if (selection_ctx.selected_mesh_index >= 0) {
				const uint32_t tiles_x       = (swapchain_ctx.extent.width + 15u) / 16u;
				uint32_t       primary_tiles = 0, neighbor_tiles = 0;
				for (uint32_t ti = 0; ti < render_target_ctx.tile_count; ++ti) {
					const TileData& td = probe_ctx.tiles[ti];
					if (td.primary_object == selection_ctx.selected_mesh_index)
						++primary_tiles;
					for (int k = 0; k < 3; ++k)
						if (td.next_objects[k] == selection_ctx.selected_mesh_index)
							++neighbor_tiles;
				}
				ImGui::Text("Primary tiles for selection: %u", primary_tiles);
				ImGui::Text("Neighbor tiles for selection: %u", neighbor_tiles);

				// Show top nextObjects for the selected object's first primary tile
				for (uint32_t ti = 0; ti < render_target_ctx.tile_count; ++ti) {
					if (probe_ctx.tiles[ti].primary_object == selection_ctx.selected_mesh_index) {
						ImGui::Text("First primary tile %u neighbors: [%d, %d, %d]", ti,
						            probe_ctx.tiles[ti].next_objects[0],
						            probe_ctx.tiles[ti].next_objects[1],
						            probe_ctx.tiles[ti].next_objects[2]);
						break;
					}
				}
			}
		}
		if (ImGui::Checkbox("Enable HiPR", &probe_ctx.hipr_enabled)) {
			probe_ctx.valid = false;
		}
		ImGui::Checkbox("Pause background fill", &probe_ctx.pause_background);
		ImGui::Checkbox("Show tile flags", &probe_ctx.show_tile_debug);
	}
	ImGui::End();
	return result;
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

void App::createSwapchainResources() {
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
	std::cout << "[INFO] Created water surface simulation resources\n";
}

// Creates buffers for tile-based light culling and HiPR, sized according to the number of tiles
static void create_tile_buffers(VmaAllocator alloc, uint32_t tiles_x, uint32_t tiles_y) {
	const uint32_t tile_count    = tiles_x * tiles_y;
	render_target_ctx.tile_count = tile_count;

	// tile_buffer: GPU writes (shader storage), CPU reads (persistently mapped)
	{
		VkBufferCreateInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size  = sizeof(TileData) * tile_count;
		bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
		ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo info;
		vmaCreateBuffer(alloc, &bi, &ai, &render_target_ctx.tile_buffer,
		                &render_target_ctx.tile_buffer_alloc, &info);
		render_target_ctx.tile_buffer_mapped = info.pMappedData;
	}

	// tile_render_flags: CPU writes, GPU reads (persistently mapped CPU→GPU)
	{
		VkBufferCreateInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size  = sizeof(uint32_t) * tile_count;
		bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo info;
		vmaCreateBuffer(alloc, &bi, &ai, &render_target_ctx.tile_render_flags_buffer,
		                &render_target_ctx.tile_render_flags_buffer_alloc, &info);
		render_target_ctx.tile_render_flags_mapped = info.pMappedData;
		// Start with everything enabled so the first frame renders fully
		memset(info.pMappedData, 0xFF, sizeof(uint32_t) * tile_count);
	}
}

void App::destroySwapchainResources() {
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

void App::recreateSwapchainResources() {
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
	// In recreateSwapchainResources(), after destroySwapchainResources() / createSwapchainResources():
	destroy_tile_buffers(render_target_ctx.allocator);
	{
		const uint32_t new_tiles_x = (swapchain_ctx.extent.width  + 15u) / 16u;
		const uint32_t new_tiles_y = (swapchain_ctx.extent.height + 15u) / 16u;
		create_tile_buffers(render_target_ctx.allocator, new_tiles_x, new_tiles_y);
	}
	initialize_imgui_renderer();

	// Recreate pathtracer storage/accum images at the new swapchain extent
	VmaAllocator allocator = render_target_ctx.allocator;
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.storage_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.storage_image,
	                render_target_ctx.storage_image_alloc);
	vkDestroyImageView(vulkan_ctx.device, render_target_ctx.accum_image_view, nullptr);
	vmaDestroyImage(allocator, render_target_ctx.accum_image, render_target_ctx.accum_image_alloc);

	// Also destroy object_id image/view so it will be recreated at the new extent
	if (render_target_ctx.object_id_image_view != VK_NULL_HANDLE) {
		vkDestroyImageView(vulkan_ctx.device, render_target_ctx.object_id_image_view, nullptr);
		render_target_ctx.object_id_image_view = VK_NULL_HANDLE;
	}
	if (render_target_ctx.object_id_image != VK_NULL_HANDLE) {
		vmaDestroyImage(allocator, render_target_ctx.object_id_image, render_target_ctx.object_id_image_alloc);
		render_target_ctx.object_id_image = VK_NULL_HANDLE;
	}

	auto make_storage = [&](VkImage& img, VmaAllocation& alloc, VkImageUsageFlags extra) {
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
		ii.extent        = {swapchain_ctx.extent.width, swapchain_ctx.extent.height, 1};
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

	// create storage & accum as before
	make_storage(render_target_ctx.storage_image, render_target_ctx.storage_image_alloc,
			 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	render_target_ctx.storage_width = swapchain_ctx.extent.width;
	render_target_ctx.storage_height = swapchain_ctx.extent.height;
	std::cout << "[DEBUG] created storage_image (recreate path) " << render_target_ctx.storage_width
		  << "x" << render_target_ctx.storage_height << "\n";
	make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
	         VK_IMAGE_USAGE_TRANSFER_DST_BIT);

	// create object_id image with R32_SINT format
	{
		VkImageCreateInfo ii{};
		ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ii.imageType     = VK_IMAGE_TYPE_2D;
		ii.format        = VK_FORMAT_R32_SINT;
		ii.extent        = {swapchain_ctx.extent.width, swapchain_ctx.extent.height, 1};
		ii.mipLevels     = 1;
		ii.arrayLayers   = 1;
		ii.samples       = VK_SAMPLE_COUNT_1_BIT;
		ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		vmaCreateImage(allocator, &ii, &ai, &render_target_ctx.object_id_image, &render_target_ctx.object_id_image_alloc, nullptr);
		render_target_ctx.object_id_width = swapchain_ctx.extent.width;
		render_target_ctx.object_id_height = swapchain_ctx.extent.height;
		std::cout << "[DEBUG] created object_id_image (recreate path) " << render_target_ctx.object_id_width
			  << "x" << render_target_ctx.object_id_height << "\n";
	}

	render_target_ctx.storage_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.storage_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.object_id_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.object_id_image, VK_FORMAT_R32_SINT,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
	render_target_ctx.accum_image_view = create_image_view(
	    vulkan_ctx.device, render_target_ctx.accum_image, VK_FORMAT_R8G8B8A8_UNORM,
	    VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

	// Transition object_id & accum → GENERAL so the shader can read/write them
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

	// Update descriptor set bindings 0 (output), 13 (object_id) and 1 (accum)
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

	// At the end of recreateSwapchainResources(), after the existing descriptor updates:
	{
		VkDescriptorBufferInfo tile_buf_info{render_target_ctx.tile_buffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo tile_flags_info{render_target_ctx.tile_render_flags_buffer, 0,
											VK_WHOLE_SIZE};
		VkWriteDescriptorSet tile_writes[2]{};
		tile_writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 14, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_buf_info};
		tile_writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 15, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_flags_info};
		vkUpdateDescriptorSets(vulkan_ctx.device, 2, tile_writes, 0, nullptr);
	}

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
	        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
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
	// Recreate tile buffers at new resolution
	destroy_tile_buffers(allocator);
	{
		const uint32_t new_tiles_x = (swapchain_ctx.extent.width  + 15u) / 16u;
		const uint32_t new_tiles_y = (swapchain_ctx.extent.height + 15u) / 16u;
		create_tile_buffers(allocator, new_tiles_x, new_tiles_y);
	}
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
				 VK_FORMAT_R8G8B8A8_UNORM,
				 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	render_target_ctx.storage_width = swapchain_ctx.extent.width;
	render_target_ctx.storage_height = swapchain_ctx.extent.height;
	std::cout << "[DEBUG] created storage_image (resize path) " << render_target_ctx.storage_width
			  << "x" << render_target_ctx.storage_height << "\n";
	make_storage(render_target_ctx.object_id_image, render_target_ctx.object_id_image_alloc,
	             VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	render_target_ctx.object_id_width = swapchain_ctx.extent.width;
	render_target_ctx.object_id_height = swapchain_ctx.extent.height;
	std::cout << "[DEBUG] created object_id_image (resize path) " << render_target_ctx.object_id_width
	          << "x" << render_target_ctx.object_id_height << "\n";
	make_storage(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	render_target_ctx.accum_width = swapchain_ctx.extent.width;
	render_target_ctx.accum_height = swapchain_ctx.extent.height;
	std::cout << "[DEBUG] created accum_image (resize path) " << render_target_ctx.accum_width
	          << "x" << render_target_ctx.accum_height << "\n";

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

	// Rebind tile buffers to their new post-resize allocations
	{
		VkDescriptorBufferInfo tile_buf_info{render_target_ctx.tile_buffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo tile_flags_info{render_target_ctx.tile_render_flags_buffer, 0,
											VK_WHOLE_SIZE};
		VkWriteDescriptorSet tile_writes[2]{};
		tile_writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 14, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_buf_info};
		tile_writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 15, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_flags_info};
		vkUpdateDescriptorSets(vulkan_ctx.device, 2, tile_writes, 0, nullptr);
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

// =================================================
// === Rebuild pipeline (hot shader compilation) ===
// =================================================
static bool rebuild_pipeline() {
	// Recompile the shader from disk
	std::vector<uint32_t> spirv;
	try {
		spirv = compile_slang_shader(std::string(SHADERS_DIR) + "/pathtracer.slang", "main",
		                             {VENDORS_DIR});
	} catch (const std::exception& e) {
		std::cerr << "[SHADER RELOAD] Compile failed: " << e.what() << "\n";
		return false;
	}

	VkShaderModuleCreateInfo mci{};
	mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	mci.codeSize = spirv.size() * sizeof(uint32_t);
	mci.pCode    = spirv.data();

	VkShaderModule new_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &new_module) != VK_SUCCESS) {
		std::cerr << "[SHADER RELOAD] vkCreateShaderModule failed\n";
		return false;
	}

	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = new_module;
	stage.pName  = "main";

	VkComputePipelineCreateInfo pci{};
	pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pci.stage  = stage;
	pci.layout = compute_ctx.pipeline_layout;

	VkPipeline new_pipeline = VK_NULL_HANDLE;
	if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
	                             &new_pipeline) != VK_SUCCESS) {
		vkDestroyShaderModule(vulkan_ctx.device, new_module, nullptr);
		std::cerr << "[SHADER RELOAD] vkCreateComputePipelines failed\n";
		return false;
	}

	vkDestroyShaderModule(vulkan_ctx.device, new_module, nullptr);

	// Swap in the new pipeline (device must be idle first)
	vkDeviceWaitIdle(vulkan_ctx.device);
	vkDestroyPipeline(vulkan_ctx.device, compute_ctx.pipeline, nullptr);
	compute_ctx.pipeline = new_pipeline;

	std::cout << "[SHADER RELOAD] Success (" << spirv.size() * 4 << " bytes)\n";
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

static void build_tlas(VmaAllocator alloc, VkDevice dev, VkCommandPool pool, VkQueue q,
                       const std::vector<BLAS>&                  blases,
                       const std::vector<std::unique_ptr<Mesh>>& meshes) {
	if (blases.size() != meshes.size())
		throw std::runtime_error("build_tlas: blases.size() != meshes.size()");

	std::vector<VkAccelerationStructureInstanceKHR> insts;
	insts.reserve(blases.size());

	for (uint32_t i = 0; i < (uint32_t) blases.size(); ++i) {
		VkAccelerationStructureInstanceKHR inst{};
		inst.transform           = glm_to_vk_transform(meshes[i]->m_transform.m_transform);
		inst.instanceCustomIndex = i;
		inst.mask                = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;
		inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		inst.accelerationStructureReference = blases[i].device_address;
		insts.push_back(inst);
	}

	VmaAllocation ia;
	VkBuffer      ib =
	    create_and_upload_buffer(alloc, sizeof(VkAccelerationStructureInstanceKHR) * insts.size(),
	                             insts.data(), AS_INPUT_BUFFER_USAGE, ia);

	VkAccelerationStructureGeometryInstancesDataKHR id{};
	id.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	id.data.deviceAddress = get_bda(dev, ib);

	VkAccelerationStructureGeometryKHR geom{};
	geom.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geom.geometry.instances = id;

	VkAccelerationStructureBuildGeometryInfoKHR build{};
	build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags         = AS_BUILD_FLAGS;
	build.geometryCount = 1;
	build.pGeometries   = &geom;

	const uint32_t ic = (uint32_t) insts.size();

	VkAccelerationStructureBuildSizesInfoKHR si{};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &build, &ic, &si);

	VmaAllocation sca;
	as_ctx.tlas_buffer = create_gpu_buffer(alloc, si.accelerationStructureSize, AS_BUFFER_USAGE,
	                                       as_ctx.tlas_buffer_alloc);
	VkBuffer scb       = create_gpu_buffer(alloc, si.buildScratchSize, SCRATCH_BUFFER_USAGE, sca,
	                                       vulkan_ctx.scratch_alignment);

	VkAccelerationStructureCreateInfoKHR aci{};
	aci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	aci.buffer = as_ctx.tlas_buffer;
	aci.size   = si.accelerationStructureSize;
	aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	if (vkCreateAccelerationStructureKHR(dev, &aci, nullptr, &as_ctx.tlas) != VK_SUCCESS)
		throw std::runtime_error("failed to create TLAS");

	build.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.dstAccelerationStructure  = as_ctx.tlas;
	build.scratchData.deviceAddress = get_bda(dev, scb);

	VkAccelerationStructureBuildRangeInfoKHR ri{};
	ri.primitiveCount                                   = ic;
	const VkAccelerationStructureBuildRangeInfoKHR* pri = &ri;

	VkCommandBuffer cmd = begin_one_time_cmd(dev, pool);
	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pri);
	end_one_time_cmd(dev, pool, q, cmd);

	VkAccelerationStructureDeviceAddressInfoKHR dai{};
	dai.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	dai.accelerationStructure = as_ctx.tlas;
	as_ctx.tlas_address       = vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);

	vmaDestroyBuffer(alloc, scb, sca);
	vmaDestroyBuffer(alloc, ib, ia);
}

// =======================
// === App constructor ===
// =======================
App::App() {
	// ==============================
	// === 0. Scene setup
	// ==============================
	m_scene           = std::make_unique<Scene>();
	m_scene->m_camera = Camera(glm::vec3(0.f, 20.f, 0.f), glm::vec3(0.f, 0.f, 0.f),
	                           glm::vec3(0.f, 1.f, 0.f), 60.f, 0.1f, 10000.f);
	m_scene->load_gltf("resources/scenes/ABeautifulGame/glTF-Binary/ABeautifulGame.glb");
	rebuildObjectIdMap(m_scene.get());
	// m_scene->load_gltf("resources/scenes/Sponza/glTF/Sponza.gltf");

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
		// If swapchain extent is already known (e.g. after createSwapchainResources), prefer it
		if (swapchain_ctx.extent.width > 0 && swapchain_ctx.extent.height > 0) {
			ext.width = swapchain_ctx.extent.width;
			ext.height = swapchain_ctx.extent.height;
		}
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
						   VK_FORMAT_R8G8B8A8_UNORM,
						   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		render_target_ctx.storage_width = (uint32_t) m_window->width();
		render_target_ctx.storage_height = (uint32_t) m_window->height();
		std::cout << "[DEBUG] created storage_image (initial) " << render_target_ctx.storage_width
				  << "x" << render_target_ctx.storage_height << "\n";
		make_storage_image(render_target_ctx.object_id_image,
						   render_target_ctx.object_id_image_alloc, VK_FORMAT_R32_SINT,
						   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		render_target_ctx.object_id_width = (uint32_t) m_window->width();
		render_target_ctx.object_id_height = (uint32_t) m_window->height();
		std::cout << "[DEBUG] created object_id_image (initial) " << render_target_ctx.object_id_width
				  << "x" << render_target_ctx.object_id_height << "\n";
		make_storage_image(render_target_ctx.accum_image, render_target_ctx.accum_image_alloc,
				   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		render_target_ctx.accum_width = (uint32_t) m_window->width();
		render_target_ctx.accum_height = (uint32_t) m_window->height();
		std::cout << "[DEBUG] created accum_image (initial) " << render_target_ctx.accum_width << "x"
				  << render_target_ctx.accum_height << "\n";
		render_target_ctx.storage_image_view =
			create_image_view(vulkan_ctx.device, render_target_ctx.storage_image,
							  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.object_id_image_view =
			create_image_view(vulkan_ctx.device, render_target_ctx.object_id_image,
							  VK_FORMAT_R32_SINT, VK_IMAGE_VIEW_TYPE_2D);
		render_target_ctx.accum_image_view =
			create_image_view(vulkan_ctx.device, render_target_ctx.accum_image,
							  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D);

		// Record the extents we used for these images (debug)
		render_target_ctx.storage_width = ext.width;
		render_target_ctx.storage_height = ext.height;
		render_target_ctx.object_id_width = ext.width;
		render_target_ctx.object_id_height = ext.height;
		render_target_ctx.accum_width = ext.width;
		render_target_ctx.accum_height = ext.height;
		std::cout << "[DEBUG] created initial images (use ext=" << ext.width << "x" << ext.height << ")\n";
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

		GPUCamera gc = m_scene->m_camera.pack();
		scene_ctx.camera_buffer =
		    create_and_upload_buffer(allocator, sizeof(GPUCamera), &gc, SB, scene_ctx.camera_alloc);
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
		scene_ctx.mesh_count = (uint32_t) gpu_meshes.size();
		std::cout << "[INFO] Mesh buffers: " << gpu_meshes.size() << " meshes, " << all_verts.size()
		          << " verts, " << all_idxs.size() << " idxs\n";
	}

	// ===================================================
	// === VIII.5  Upload OpenPBR LUTs + scene textures
	// ===================================================
	upload_openpbr_luts(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                    vulkan_ctx.graphics_queue);
	upload_material_textures(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                         vulkan_ctx.graphics_queue, m_scene->m_textures);

	// ===================================================
	// === VIII.6  Tile probe buffers
	// ===================================================
	{
		uint32_t wnd_w = m_window->width();
		uint32_t wnd_h = m_window->height();
		uint32_t sc_w = swapchain_ctx.extent.width;
		uint32_t sc_h = swapchain_ctx.extent.height;
		// Prefer swapchain extent for GPU-sized buffers; fall back to window size.
		uint32_t use_w = (sc_w > 0) ? sc_w : wnd_w;
		uint32_t use_h = (sc_h > 0) ? sc_h : wnd_h;
		const uint32_t tiles_x = (use_w + 15u) / 16u;
		const uint32_t tiles_y = (use_h + 15u) / 16u;
		create_tile_buffers(allocator, tiles_x, tiles_y);
		std::cout << "[INFO] Tile buffers: " << tiles_x * tiles_y << " tiles"
				  << " (window=" << wnd_w << "x" << wnd_h << ", swapchain=" << sc_w
				  << "x" << sc_h << ")\n";
	}

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
	//  12  = material sampler        SAMPLER
	//  13  = object ID image         STORAGE_IMAGE
	//  14  = tile probe buffer       STORAGE_BUFFER
	//  15  = tile list buffer        STORAGE_BUFFER
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
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
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

		// // 14 — tile_buffer
		// VkDescriptorBufferInfo tile_buf_info{render_target_ctx.tile_buffer, 0, VK_WHOLE_SIZE};
		// writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		//                   render_target_ctx.descriptor_set, 14, 0, 1,
		//                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_buf_info});

		// // 15 — tile_render_flags
		// VkDescriptorBufferInfo tile_flags_info{render_target_ctx.tile_render_flags_buffer, 0,
		//                                        VK_WHOLE_SIZE};
		// writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		//                   render_target_ctx.descriptor_set, 15, 0, 1,
		//                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_flags_info});
		
		// Re-bind tile buffers (bindings 14 and 15)
		VkDescriptorBufferInfo tile_buf_info{render_target_ctx.tile_buffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo tile_flags_info{render_target_ctx.tile_render_flags_buffer, 0, VK_WHOLE_SIZE};

		VkWriteDescriptorSet tile_writes[2]{};
		tile_writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 14, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_buf_info};
		tile_writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
						render_target_ctx.descriptor_set, 15, 0, 1,
						VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tile_flags_info};
		vkUpdateDescriptorSets(vulkan_ctx.device, 2, tile_writes, 0, nullptr);

		// 12 — sampler for material textures
		VkDescriptorImageInfo material_sampler_info{};
		material_sampler_info.sampler = render_target_ctx.material_sampler;
		writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
		                  render_target_ctx.descriptor_set, 12, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
		                  &material_sampler_info});
		vkUpdateDescriptorSets(vulkan_ctx.device, (uint32_t) writes.size(), writes.data(), 0,
		                       nullptr);
		std::cout << "[INFO] Descriptor sets updated\n";
	}

	// ============================================
	// === X. Compute Pipeline
	// ============================================
	{
		auto spirv =
		    compile_slang_shader(std::string(SHADERS_DIR) + "/HiPR.slang", "main", {VENDORS_DIR});
		std::cout << "[INFO] Shader: " << spirv.size() * 4 << " bytes SPIR-V\n";
		VkShaderModuleCreateInfo mci{};
		mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		mci.codeSize = spirv.size() * sizeof(uint32_t);
		mci.pCode    = spirv.data();
		VkShaderModule sm;
		if (vkCreateShaderModule(vulkan_ctx.device, &mci, nullptr, &sm) != VK_SUCCESS)
			throw std::runtime_error("failed to create shader module");
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
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = sm;
		stage.pName  = "main";
		VkComputePipelineCreateInfo pci{};
		pci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pci.stage  = stage;
		pci.layout = compute_ctx.pipeline_layout;
		if (vkCreateComputePipelines(vulkan_ctx.device, VK_NULL_HANDLE, 1, &pci, nullptr,
		                             &compute_ctx.pipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create compute pipeline");
		vkDestroyShaderModule(vulkan_ctx.device, sm, nullptr);
		std::cout << "[INFO] Compute pipeline created\n";
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

App::~App() {
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

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	float last_frame_time = static_cast<float>(glfwGetTime());

	FlyCamera fly_cam(m_scene->m_camera.m_position, m_scene->m_camera.m_target,
	                  m_scene->m_camera.m_fov, 0.1f);

	double   last_time    = glfwGetTime();
	uint32_t frame_number = 0;

	int prev_f6  = GLFW_RELEASE;
	int prev_f11 = GLFW_RELEASE;
	int prev_lmb = GLFW_RELEASE;

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
			frame_number = 0;
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
			overlay_ctx.show_control_panel = !overlay_ctx.show_control_panel;
		}

		bool controls_changed = false;
		if (overlay_ctx.show_control_panel) {
			controls_changed = ui::drawAudienceControlPanel(
			    &overlay_ctx.show_control_panel, overlay_ctx.controls, overlay_ctx.diagnostics);
		}

		const SelectionPanelResult selection_panel_result = drawSelectionPanel(m_scene.get());

		if (controls_changed) {
			if (m_audio_controller != nullptr) {
				audio_level = m_audio_controller->update(overlay_ctx.controls.audio, audio_input);
				overlay_ctx.diagnostics.audio = m_audio_controller->diagnostics();
			}
			applyOverlayLevel(audio_level);
			update_water_and_floaters(overlay_ctx.diagnostics.audio.normalized_level);
		}

		if (selection_panel_result.material_changed) {
			applySelectedMaterialEditor(m_scene.get(), render_target_ctx.allocator);
			frame_number = 0;
		}
		if (selection_panel_result.selection_changed) {
			frame_number = 0;
		}

		if (overlay_ctx.controls.reset_water_requested) {
			if (m_water_surface != nullptr)
				m_water_surface->requestReset();
			overlay_ctx.controls.reset_water_requested = false;
		}
		if (overlay_ctx.controls.reset_objects_requested) {
			if (m_water_surface != nullptr)
				m_water_surface->requestObjectReset();
			overlay_ctx.controls.reset_objects_requested = false;
		}

		if (overlay_ctx.controls.show_overlay) {
			ui::drawAudienceOverlay(ImGui::GetIO().DisplaySize, overlay_ctx.controls.overlay,
			                        overlay_ctx.controls.style);
		}

		// ── Tile debug overlay ───────────────────────────────────────────────
		if (probe_ctx.show_tile_debug && render_target_ctx.tile_render_flags_mapped != nullptr) {
			const auto* flags_ptr =
			    reinterpret_cast<const uint32_t*>(render_target_ctx.tile_render_flags_mapped);
			const uint32_t tiles_x_dbg = (swapchain_ctx.extent.width + 15u) / 16u;
			const uint32_t tiles_y_dbg = (swapchain_ctx.extent.height + 15u) / 16u;
			ImDrawList*    dl          = ImGui::GetForegroundDrawList();
			const float    screen_w    = static_cast<float>(swapchain_ctx.extent.width);
			const float    screen_h    = static_cast<float>(swapchain_ctx.extent.height);
			const float    tile_pw     = screen_w / static_cast<float>(tiles_x_dbg);
			const float    tile_ph     = screen_h / static_cast<float>(tiles_y_dbg);
			for (uint32_t ty = 0; ty < tiles_y_dbg; ++ty) {
				for (uint32_t tx = 0; tx < tiles_x_dbg; ++tx) {
					const uint32_t idx     = ty * tiles_x_dbg + tx;
					const bool     flagged = flags_ptr[idx] != 0u;
					const ImVec2   p0      = {tx * tile_pw, ty * tile_ph};
					const ImVec2   p1      = {(tx + 1) * tile_pw, (ty + 1) * tile_ph};
					const ImU32    col =
					    flagged ? IM_COL32(0, 255, 80, 50) : IM_COL32(255, 30, 30, 80);
					dl->AddRectFilled(p0, p1, col);
					dl->AddRect(p0, p1, IM_COL32(0, 0, 0, 40));
				}
			}
		}

		ImGui::Render();

		glfwPollEvents();

		double now = glfwGetTime();
		float  dt  = std::clamp(static_cast<float>(now - last_time), 0.0001f, 0.1f);
		last_time  = now;

		// ── Resize check ─────────────────────────────────────────────────────
		uint32_t fb_w = m_window->width();
		uint32_t fb_h = m_window->height();
		if (fb_w != swapchain_ctx.swapchain.extent.width ||
		    fb_h != swapchain_ctx.swapchain.extent.height) {
			handle_resize(frame_number, fb_w, fb_h);
			probe_ctx.valid = false;        // tile buffer resized — probe stale
		}

		// ── F11: fullscreen toggle ────────────────────────────────────────────
		int f11 = glfwGetKey(m_window->handle(), GLFW_KEY_F11);
		if (f11 == GLFW_PRESS && prev_f11 == GLFW_RELEASE) {
			m_window->toggle_fullscreen();
		}
		prev_f11 = f11;

		// ── F6: shader hot-reload ─────────────────────────────────────────────
		int f6 = glfwGetKey(m_window->handle(), GLFW_KEY_F6);
		if (f6 == GLFW_PRESS && prev_f6 == GLFW_RELEASE) {
			if (rebuild_pipeline()) {
				frame_number    = 0;
				probe_ctx.valid = false;        // new shader may behave differently
			}
		}
		prev_f6 = f6;

		// ── Fly-camera update ─────────────────────────────────────────────────
		if (fly_cam.update(m_window->handle(), dt)) {
			frame_number    = 0;
			probe_ctx.valid = false;        // camera moved — interaction map is stale
			// Clear accumulation image immediately to avoid temporal smearing
			if (render_target_ctx.accum_image != VK_NULL_HANDLE) {
				VkCommandBuffer clear_cmd =
				    begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);
				// Transition accum -> TRANSFER_DST
				transition_layout(clear_cmd, render_target_ctx.accum_image, VK_IMAGE_LAYOUT_GENERAL,
				                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
				                  VK_ACCESS_TRANSFER_WRITE_BIT,
				                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                  VK_PIPELINE_STAGE_TRANSFER_BIT);
				VkClearColorValue clear_color{};
				clear_color.float32[0] = 0.0f;
				clear_color.float32[1] = 0.0f;
				clear_color.float32[2] = 0.0f;
				clear_color.float32[3] = 0.0f;
				VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
				vkCmdClearColorImage(clear_cmd, render_target_ctx.accum_image,
				                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);
				// Transition back to GENERAL for shader access
				transition_layout(clear_cmd, render_target_ctx.accum_image,
				                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
				                  VK_PIPELINE_STAGE_TRANSFER_BIT,
				                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
				end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool,
				                 vulkan_ctx.graphics_queue, clear_cmd);
			}
		}

		const GPUCamera gpu_camera = fly_cam.pack();
		memcpy(scene_ctx.camera_mapped, &gpu_camera, sizeof(GPUCamera));

		// ── CPU pick on LMB ───────────────────────────────────────────────────
		const int current_lmb = glfwGetMouseButton(m_window->handle(), GLFW_MOUSE_BUTTON_LEFT);
		if (current_lmb == GLFW_PRESS && prev_lmb == GLFW_RELEASE && !fly_cam.isMouseCaptured() &&
		    !ImGui::GetIO().WantCaptureMouse) {
			if (selectMesh(m_scene.get(),
			               pickMeshAtCursor(m_scene.get(), m_window->handle(), gpu_camera,
			                                framebuffer_width, framebuffer_height))) {
				frame_number = 0;
				// probe_ctx.valid stays true — the probe map is still good.
				// The BFS just re-runs from the new seed, which is cheap CPU work.
			}
		}
		prev_lmb = current_lmb;

		// =====================================================================
		// === Probe pass (stage 0) — runs when stale ===
		// Submits a separate one-shot command buffer, stalls until done, then
		// reads back tile_buffer and recomputes tile_render_flags on the CPU.
		// This only fires on scene changes, not every frame.
		// =====================================================================
		const uint32_t tiles_x    = (swapchain_ctx.extent.width + 15u) / 16u;
		const uint32_t tiles_y    = (swapchain_ctx.extent.height + 15u) / 16u;
		const uint32_t tile_count = tiles_x * tiles_y;

		const bool need_probe = !probe_ctx.valid || frame_number == 0 ||
		                        probe_ctx.last_selected != selection_ctx.selected_mesh_index;

		if (need_probe) {
			// -- Submit stage 0 -----------------------------------------------
			{
				VkCommandBuffer probe_cmd =
				    begin_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool);

				// Storage image must be in GENERAL for the compute shader to write
				VkImageLayout storage_old = render_target_ctx.storage_image_initialized ?
				                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
				                                VK_IMAGE_LAYOUT_UNDEFINED;
				transition_layout(probe_cmd, render_target_ctx.storage_image, storage_old,
				                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
				                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

				vkCmdBindPipeline(probe_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline);
				vkCmdBindDescriptorSets(probe_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        compute_ctx.pipeline_layout, 0, 1,
				                        &render_target_ctx.descriptor_set, 0, nullptr);

				PathTracerPushConstants pc{};
				pc.frame               = frame_number;
				pc.material_count      = scene_ctx.material_count;
				pc.selected_mesh_index = selection_ctx.selected_mesh_index;
				pc.outline_width       = selection_ctx.outline_width;
				pc.debug_view_mode     = static_cast<int32_t>(selection_ctx.debug_view_mode);
				pc.stage               = 0u;
				pc.outline_color       = selection_ctx.outline_color;
				vkCmdPushConstants(probe_cmd, compute_ctx.pipeline_layout,
				                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

				vkCmdDispatch(probe_cmd, tiles_x, tiles_y, 1);

				// end_one_time_cmd submits and calls vkQueueWaitIdle —
				// that's the sync point before we read the mapped buffer below.
				end_one_time_cmd(vulkan_ctx.device, command_ctx.command_pool,
				                 vulkan_ctx.graphics_queue, probe_cmd);

				render_target_ctx.storage_image_initialized = true;
			}

			// -- Readback tile_buffer (persistently mapped, so just memcpy) ---
			probe_ctx.tiles.resize(tile_count);
			memcpy(probe_ctx.tiles.data(), render_target_ctx.tile_buffer_mapped,
			       sizeof(TileData) * tile_count);

			// -- BFS: compute which tiles to prioritise -----------------------
			std::vector<uint32_t> flags;
			if (probe_ctx.hipr_enabled) {
				bfs_compute_tile_flags(probe_ctx.tiles, tile_count,
				                       selection_ctx.selected_mesh_index, flags);
			} else {
				// Render everything — bypass HiPR entirely
				flags.assign(tile_count, 1u);
			}
			// -- Upload render flags (persistently mapped CPU→GPU) ------------
			memcpy(render_target_ctx.tile_render_flags_mapped, flags.data(),
			       sizeof(uint32_t) * tile_count);
			vmaFlushAllocation(render_target_ctx.allocator,
			                   render_target_ctx.tile_render_flags_buffer_alloc, 0, VK_WHOLE_SIZE);

			// Debug: report probe result summary
			size_t flagged_count = 0;
			for (uint32_t f : flags)
				if (f != 0u) ++flagged_count;
			std::cout << "[DEBUG] probe completed: selected=" << selection_ctx.selected_mesh_index
					  << ", flagged_tiles=" << flagged_count << "\n";

			probe_ctx.valid             = true;
			probe_ctx.last_selected     = selection_ctx.selected_mesh_index;
			probe_ctx.background_cursor = 0u;
		}

		// =====================================================================
		// === Background refinement — drip-feed non-BFS tiles ===
		// Each frame we enable a small batch of tiles that weren't in the BFS
		// frontier, advancing a round-robin cursor through the full tile set.
		// This ensures the whole image converges without a hard cut-over.
		// =====================================================================
		if (probe_ctx.pause_background && render_target_ctx.tile_render_flags_mapped != nullptr) {
			auto* flags_ptr =
			    reinterpret_cast<uint32_t*>(render_target_ctx.tile_render_flags_mapped);

			int added = 0;
			for (uint32_t i = 0; i < tile_count && added < BACKGROUND_TILES_PER_FRAME; ++i) {
				const uint32_t idx = (probe_ctx.background_cursor + i) % tile_count;
				if (flags_ptr[idx] == 0u) {
					flags_ptr[idx] = 1u;
					++added;
				}
			}
			probe_ctx.background_cursor =
			    (probe_ctx.background_cursor + BACKGROUND_TILES_PER_FRAME) % tile_count;

			if (added > 0) {
				vmaFlushAllocation(render_target_ctx.allocator,
				                   render_target_ctx.tile_render_flags_buffer_alloc, 0,
				                   VK_WHOLE_SIZE);
			}
		}

		// =====================================================================
		// === Per-frame render (stage 1 — path trace) ===
		// =====================================================================
		const uint32_t frame_idx =
		    frame_number % static_cast<uint32_t>(swapchain_ctx.images.size());

		vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
		vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);

		uint32_t image_index;
		VkResult acquire_result = vkAcquireNextImageKHR(
		    vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
		    sync_ctx.image_available[frame_idx], VK_NULL_HANDLE, &image_index);
		if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
			handle_resize(frame_number, fb_w, fb_h);
			probe_ctx.valid = false;
			continue;
		}

		vkResetCommandBuffer(command_ctx.command_buffer, 0);
		VkCommandBufferBeginInfo begin_info{};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);
		VkCommandBuffer cmd = command_ctx.command_buffer;

		// Storage image: probe already left it in TRANSFER_SRC (it writes then
		// end_one_time_cmd flushes, but we never transitioned it out of GENERAL
		// in the probe path). Transition from GENERAL on non-probe frames.
		VkImageLayout storage_old = render_target_ctx.storage_image_initialized ?
		                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
		                                VK_IMAGE_LAYOUT_UNDEFINED;
		// On probe frames the image ended in GENERAL (one-shot cmd didn't
		// transition it back), so we must always go from GENERAL if we just ran
		// the probe.
		if (need_probe)
			storage_old = VK_IMAGE_LAYOUT_GENERAL;

		transition_layout(cmd, render_target_ctx.storage_image, storage_old,
		                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		VkImageLayout swap_old = swapchain_ctx.image_initialized[image_index] ?
		                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
		                             VK_IMAGE_LAYOUT_UNDEFINED;
		transition_layout(cmd, swapchain_ctx.images[image_index], swap_old,
		                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_ctx.pipeline_layout, 0,
		                        1, &render_target_ctx.descriptor_set, 0, nullptr);

		PathTracerPushConstants pc{};
		pc.frame               = frame_number;
		pc.material_count      = scene_ctx.material_count;
		pc.selected_mesh_index = selection_ctx.selected_mesh_index;
		pc.outline_width       = selection_ctx.outline_width;
		pc.debug_view_mode     = static_cast<int32_t>(selection_ctx.debug_view_mode);
		pc.stage               = 1u;        // path trace pass
		pc.outline_color       = selection_ctx.outline_color;
		vkCmdPushConstants(cmd, compute_ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                   sizeof(pc), &pc);

		vkCmdDispatch(cmd, tiles_x, tiles_y, 1);

		render_target_ctx.storage_image_initialized  = true;
		swapchain_ctx.image_initialized[image_index] = true;
		frame_number++;

		// ── Blit pathtracer output → swapchain ────────────────────────────────
		transition_layout(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_GENERAL,
		                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
		                  VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                  VK_PIPELINE_STAGE_TRANSFER_BIT);

		VkImageBlit blit{};
		blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.srcOffsets[1]  = {static_cast<int32_t>(swapchain_ctx.extent.width),
		                       static_cast<int32_t>(swapchain_ctx.extent.height), 1};
		blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		blit.dstOffsets[1]  = {static_cast<int32_t>(swapchain_ctx.extent.width),
		                       static_cast<int32_t>(swapchain_ctx.extent.height), 1};
		vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		               &blit, VK_FILTER_LINEAR);

		// ── ImGui overlay ─────────────────────────────────────────────────────
		VkImageMemoryBarrier overlay_barrier{};
		overlay_barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		overlay_barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		overlay_barrier.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		overlay_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		overlay_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		overlay_barrier.image               = swapchain_ctx.images[image_index];
		overlay_barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		overlay_barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		overlay_barrier.dstAccessMask =
		    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
		                     nullptr, 1, &overlay_barrier);

		VkRenderPassBeginInfo render_pass_info{};
		render_pass_info.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_info.renderPass  = overlay_ctx.render_pass;
		render_pass_info.framebuffer = overlay_ctx.framebuffers[image_index];
		render_pass_info.renderArea  = {{0, 0}, swapchain_ctx.extent};
		vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		vkCmdEndRenderPass(cmd);

		check_vk_result(vkEndCommandBuffer(cmd));

		// ── Submit ────────────────────────────────────────────────────────────
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		VkSubmitInfo         submit{};
		submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.waitSemaphoreCount   = 1;
		submit.pWaitSemaphores      = &sync_ctx.image_available[frame_idx];
		submit.pWaitDstStageMask    = &wait_stage;
		submit.commandBufferCount   = 1;
		submit.pCommandBuffers      = &cmd;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores    = &sync_ctx.render_finished[image_index];
		vkQueueSubmit(vulkan_ctx.graphics_queue, 1, &submit, sync_ctx.in_flight);

		// ── Present ───────────────────────────────────────────────────────────
		VkPresentInfoKHR present{};
		present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present.waitSemaphoreCount = 1;
		present.pWaitSemaphores    = &sync_ctx.render_finished[image_index];
		present.swapchainCount     = 1;
		present.pSwapchains        = &swapchain_ctx.swapchain.swapchain;
		present.pImageIndices      = &image_index;
		VkResult present_result    = vkQueuePresentKHR(vulkan_ctx.graphics_queue, &present);
		if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
			handle_resize(frame_number, fb_w, fb_h);
			probe_ctx.valid = false;
		}
	}

	vkDeviceWaitIdle(vulkan_ctx.device);
}