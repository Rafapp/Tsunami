#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#	define VK_NO_PROTOTYPES
#endif

#define VOLK_IMPLEMENTATION
#include "volk.h"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "VkBootstrap.h"
#include "vk_mem_alloc.h"

#include "tsunami/core/window.h"
#include "tsunami/scene/scene.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/vulkan/vulkan.h"

using vulkan::Runtime;
using vulkan::RuntimeFrameInput;
using vulkan::RuntimeFrameOutput;

// Module headers — struct definitions + extern globals + all helper functions.
// VOLK_IMPLEMENTATION and VMA_IMPLEMENTATION are defined above in this TU only.
#include "tsunami/vulkan/internal/vk_context.h"
#include "tsunami/vulkan/vk_acceleration.h"
#include "tsunami/vulkan/vk_helpers.h"
#include "tsunami/vulkan/vk_overlay.h"
#include "tsunami/vulkan/vk_pipeline.h"
#include "tsunami/vulkan/vk_resources.h"
#include "tsunami/vulkan/vk_scene.h"

class Runtime::Impl {
  public:
	Impl(core::Window& window, const Scene& scene);
	~Impl();

	void                               createSwapchainResources();
	void                               destroySwapchainResources();
	void                               recreateSwapchainResources();
	RuntimeFrameOutput                 beginFrame();
	void                               renderFrame(const RuntimeFrameInput& input);
	simulation::WaterSurfaceCreateInfo waterSurfaceCreateInfo() const;
	void uploadMaterial(int material_index, const GPUMaterial& material);
	bool rebuildPipeline(ui::RenderDebugViewMode mode);
	void updateCamera(const GPUCamera& camera);
	void waitIdle();

	void resetHiPRObjectSampling() {
		m_hipr_object_sampling_active = false;
		m_hipr_object_sampling_done   = false;
		m_hipr_object_sampling_rank   = 0;
		m_hipr_object_sampling_frame  = 0;
		m_hipr_full_scene_frame       = 0;
	}

	void restartHiPRObjectSampling() {
		m_hipr_object_sampling_active = true;
		m_hipr_object_sampling_done   = false;
		m_hipr_object_sampling_rank   = 0;
		m_hipr_object_sampling_frame  = 0;
		m_hipr_full_scene_frame       = 0;
	}

	core::Window*                          m_window = nullptr;
	vulkan::internal::RuntimeContextBundle m_contexts{};
	float                                  m_last_frame_time        = 0.0f;
	uint32_t                               m_frame_number           = 0;
	bool                                   m_needs_visibility_pass  = true;
	bool                                   m_material_edit_mode     = false;
	bool                                   m_camera_was_moving      = false;
	bool                                   m_hipr_force_clear_order = true;
	ui::RenderDebugViewMode                m_last_render_mode       = ui::RenderDebugViewMode::HiPR;
	ui::LightingSettings                   m_last_lighting{};
	bool                                   m_hipr_object_sampling_active = false;
	bool                                   m_hipr_object_sampling_done   = false;
	uint32_t                               m_hipr_object_sampling_rank   = 0;
	uint32_t                               m_hipr_object_sampling_frame  = 0;
	uint32_t                               m_hipr_full_scene_frame       = 0;
	bool                                   m_surface_recreation_pending  = false;
	GPUCamera                              m_current_camera{};
	bool                                   m_has_current_camera = false;
	bool                                   m_camera_changed     = false;
};

namespace {

vulkan::internal::RuntimeContextBundle* g_active_runtime_contexts = nullptr;

vulkan::internal::RuntimeContextBundle& requireActiveRuntimeContexts() {
	if (g_active_runtime_contexts == nullptr) {
		throw std::runtime_error("vulkan runtime is not active");
	}
	return *g_active_runtime_contexts;
}

constexpr float kMinFrameDelta = 1.0f / 240.0f;

uint32_t sanitize_hipr_ten_step_value(uint32_t value) {
	if (value <= 1u) {
		return 1u;
	}
	if (value >= 100u) {
		return 100u;
	}

	uint32_t snapped = ((value + 5u) / 10u) * 10u;
	snapped          = std::max(snapped, 10u);
	snapped          = std::min(snapped, 100u);
	return snapped;
}

ui::AudienceRenderDiagnostics buildRenderDiagnostics(float delta_time_seconds) {
	frame_timing_history.pushFrame(delta_time_seconds);

	ui::AudienceRenderDiagnostics render{};
	render.current_fps           = frame_timing_history.current_fps;
	render.average_fps           = frame_timing_history.average_fps;
	render.min_fps               = frame_timing_history.min_fps;
	render.max_fps               = frame_timing_history.max_fps;
	render.current_frame_time_ms = frame_timing_history.current_frame_ms;
	render.average_frame_time_ms = frame_timing_history.average_frame_ms;
	render.min_frame_time_ms     = frame_timing_history.min_frame_ms;
	render.max_frame_time_ms     = frame_timing_history.max_frame_ms;
	render.frame_sample_count    = static_cast<uint32_t>(frame_timing_history.sample_count);
	render.render_width          = swapchain_ctx.extent.width;
	render.render_height         = swapchain_ctx.extent.height;
	render.swapchain_image_count = static_cast<uint32_t>(swapchain_ctx.images.size());

	if (ImGui::GetCurrentContext() != nullptr) {
		const ImGuiIO& io         = ImGui::GetIO();
		render.imgui_vertex_count = io.MetricsRenderVertices;
		render.imgui_index_count  = io.MetricsRenderIndices;
		render.imgui_window_count = io.MetricsRenderWindows;
	}

	return render;
}

}        // namespace

namespace vulkan::internal {

RuntimeContextBundle& current_contexts() {
	return requireActiveRuntimeContexts();
}

}        // namespace vulkan::internal

void FrameTimingHistory::pushFrame(float delta_time_seconds) {
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

void Runtime::Impl::createSwapchainResources() {
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
}

void Runtime::Impl::destroySwapchainResources() {
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

void Runtime::Impl::recreateSwapchainResources() {
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

	render_target_ctx.storage_image_initialized = false;
}

// =======================
// === Window resizing ===
// =======================
static void handle_resize(uint32_t& frame_number, uint32_t fb_w, uint32_t fb_h) {
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

Runtime::Impl::Impl(core::Window& window, const Scene& scene) : m_window(&window) {
	g_active_runtime_contexts = &m_contexts;
	// ========================================
	// === I. Vulkan function pointers
	// ========================================
	if (volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("failed to initialize volk");
	std::cout << "[INFO] Volk initialized\n";

	/* Window creation moved to App
	// === II. Window
	// =========================
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});

	*/
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
		GPUCamera initial_cam   = scene.m_camera.pack();
		memcpy(scene_ctx.camera_mapped, &initial_cam, sizeof(GPUCamera));
		std::vector<GPUMaterial> gms;

		for (const auto& mesh : scene.m_meshes)
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

	for (int mi = 0; mi < (int) scene.m_meshes.size(); ++mi) {
		const auto& mesh = scene.m_meshes[mi];
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
		           vulkan_ctx.graphics_queue, as_ctx.blases, scene.m_meshes);
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

	// ===================================================
	// === VIII.5  Upload OpenPBR LUTs + scene textures
	// ===================================================
	upload_openpbr_luts(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                    vulkan_ctx.graphics_queue);
	upload_material_textures(allocator, vulkan_ctx.device, command_ctx.command_pool,
	                         vulkan_ctx.graphics_queue, scene.m_textures);

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
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
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
		std::cout << "[INFO] Descriptor sets updated\n";
	}

	// ============================================
	// === X. Compute Pipeline
	// ============================================
	{
		init_pipeline_modes(vulkan_ctx.device, render_target_ctx.descriptor_set_layout);
		std::cout << "[INFO] Compute pipeline layouts created\n";

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

	m_last_frame_time = static_cast<float>(glfwGetTime());
}

Runtime::Impl::~Impl() {
	if (vulkan_ctx.device == VK_NULL_HANDLE) {
		if (g_active_runtime_contexts == &m_contexts) {
			g_active_runtime_contexts = nullptr;
		}
		return;
	}

	vkDeviceWaitIdle(vulkan_ctx.device);

	shutdown_imgui();
	destroySwapchainResources();

	destroy_pipeline_modes(vulkan_ctx.device);

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

	if (as_ctx.tlas != VK_NULL_HANDLE) {
		vkDestroyAccelerationStructureKHR(vulkan_ctx.device, as_ctx.tlas, nullptr);
		as_ctx.tlas = VK_NULL_HANDLE;
	}
	if (as_ctx.tlas_buffer != VK_NULL_HANDLE && as_ctx.tlas_buffer_alloc != VK_NULL_HANDLE) {
		vmaDestroyBuffer(render_target_ctx.allocator, as_ctx.tlas_buffer, as_ctx.tlas_buffer_alloc);
		as_ctx.tlas_buffer       = VK_NULL_HANDLE;
		as_ctx.tlas_buffer_alloc = VK_NULL_HANDLE;
	}
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

	if (g_active_runtime_contexts == &m_contexts) {
		g_active_runtime_contexts = nullptr;
	}
}

// ============================================================
// === Main loop
// ============================================================
#if 0
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
		scene_ctx.selection_voice_loudness = std::clamp(audio_level, 0.0f, 1.0f);
		const float water_audio_level      = overlay_ctx.diagnostics.audio.normalized_level;
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
			selection_panel_result = ui::drawSelectionPanel(
			    m_scene.get(), scene_ctx.selection_voice_loudness, &show_selection_panel);
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
			scene_ctx.selection_voice_loudness    = std::clamp(audio_level, 0.0f, 1.0f);
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
			handle_resize(frame_number, fb_w, fb_h);
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

		const auto          active_render_mode = ui::selection_ctx.debug_view_mode;
		const size_t        mode_idx           = render_mode_index(active_render_mode);
		const PipelineMode& active_mode        = compute_ctx.modes[mode_idx];
		const VkPipeline    active_pipeline    = active_mode.pipeline;
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
			handle_resize(frame_number, fb_w, fb_h);
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
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, active_mode.layout, 0,
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
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                   sizeof(pc), &pc);
			const uint32_t clear_count =
			    std::max(pc.hipr_object_count, std::max(pc.hipr_top_k, 1u));
			vkCmdDispatch(cmd, (clear_count + 15) / 16, 1, 1);
			hipr_force_clear_order = false;
		}

		// Stage 0: visibility pass.
		if (needs_visibility_pass) {
			pc.stage = 0;
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
				vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
			handle_resize(frame_number, fb_w, fb_h);
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
#endif

RuntimeFrameOutput Runtime::Impl::beginFrame() {
	g_active_runtime_contexts = &m_contexts;

	RuntimeFrameOutput output{};

	const uint32_t framebuffer_width  = m_window->width();
	const uint32_t framebuffer_height = m_window->height();
	if ((framebuffer_width > 0 && framebuffer_height > 0) &&
	    (m_surface_recreation_pending || swapchain_ctx.extent.width != framebuffer_width ||
	     swapchain_ctx.extent.height != framebuffer_height)) {
		m_frame_number               = 0;
		m_needs_visibility_pass      = true;
		m_hipr_force_clear_order     = true;
		m_surface_recreation_pending = false;
		resetHiPRObjectSampling();
		recreateSwapchainResources();
		output.surface_resized = true;
	}

	const float time_seconds = static_cast<float>(glfwGetTime());
	const float delta_time   = std::max(time_seconds - m_last_frame_time, kMinFrameDelta);
	m_last_frame_time        = time_seconds;
	output.render            = buildRenderDiagnostics(delta_time);

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	return output;
}

void Runtime::Impl::renderFrame(const RuntimeFrameInput& input) {
	g_active_runtime_contexts = &m_contexts;

	if (m_surface_recreation_pending || !m_has_current_camera || swapchain_ctx.images.empty()) {
		return;
	}

	const ui::SelectionContext&           selection                = input.selection;
	const ui::AudienceRenderPostSettings& render_post              = input.render_post;
	const bool                            camera_moving_this_frame = input.camera_moving;

	if (selection.debug_view_mode != m_last_render_mode) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_hipr_force_clear_order = true;
		m_last_render_mode       = selection.debug_view_mode;
		resetHiPRObjectSampling();
	}

	if (input.material_edit_active && !m_material_edit_mode) {
		m_material_edit_mode = true;
		m_frame_number       = 0;
		resetHiPRObjectSampling();
	}

	if (input.material_changed) {
		m_frame_number = 0;
		restartHiPRObjectSampling();
	}

	if (input.material_edit_just_finished) {
		m_material_edit_mode = false;
		m_frame_number       = 0;
	}

	if (input.selection_changed) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_material_edit_mode     = false;
		m_hipr_force_clear_order = true;
		resetHiPRObjectSampling();
	}

	if (input.hipr_settings_changed) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_hipr_force_clear_order = true;
		if (selection.debug_view_mode == ui::RenderDebugViewMode::HiPR &&
		    selection.selected_mesh_index >= 0) {
			restartHiPRObjectSampling();
		} else {
			resetHiPRObjectSampling();
		}
	}

	if (input.path_tracing_settings_changed) {
		m_frame_number = 0;
		if (selection.debug_view_mode == ui::RenderDebugViewMode::HiPR &&
		    selection.selected_mesh_index >= 0) {
			restartHiPRObjectSampling();
		} else {
			resetHiPRObjectSampling();
		}
	}

	if (selection.lighting.skybox_enabled != m_last_lighting.skybox_enabled ||
	    selection.lighting.directional_light_enabled != m_last_lighting.directional_light_enabled ||
	    selection.lighting.sun_elevation_deg != m_last_lighting.sun_elevation_deg ||
	    selection.lighting.sun_azimuth_deg != m_last_lighting.sun_azimuth_deg ||
	    selection.lighting.sun_intensity != m_last_lighting.sun_intensity) {
		m_frame_number  = 0;
		m_last_lighting = selection.lighting;
		resetHiPRObjectSampling();
	}

	if (m_camera_changed) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_hipr_force_clear_order = true;
		resetHiPRObjectSampling();
	}
	if (camera_moving_this_frame) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_hipr_force_clear_order = true;
		resetHiPRObjectSampling();
	}
	if (!camera_moving_this_frame && m_camera_was_moving) {
		m_frame_number = 0;
		resetHiPRObjectSampling();
	}
	m_camera_was_moving = camera_moving_this_frame;
	m_camera_changed    = false;

	const auto          active_render_mode = selection.debug_view_mode;
	const size_t        mode_idx           = render_mode_index(active_render_mode);
	const PipelineMode& active_mode        = compute_ctx.modes[mode_idx];
	const VkPipeline    active_pipeline    = active_mode.pipeline;
	if (active_pipeline == VK_NULL_HANDLE) {
		return;
	}

	const uint32_t frame_idx = m_frame_number % static_cast<uint32_t>(swapchain_ctx.images.size());

	VkResult wait_result =
	    vkWaitForFences(vulkan_ctx.device, 1, &sync_ctx.in_flight, VK_TRUE, UINT64_MAX);
	if (wait_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkWaitForFences failed: " << static_cast<int>(wait_result) << "\n";
		if (wait_result == VK_ERROR_DEVICE_LOST) {
			throw std::runtime_error("vkWaitForFences failed with VK_ERROR_DEVICE_LOST");
		}
		return;
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

	uint32_t image_index = 0;
	VkResult acquire_result =
	    vkAcquireNextImageKHR(vulkan_ctx.device, swapchain_ctx.swapchain.swapchain, UINT64_MAX,
	                          sync_ctx.image_available[frame_idx], VK_NULL_HANDLE, &image_index);
	const bool acquire_suboptimal = (acquire_result == VK_SUBOPTIMAL_KHR);
	if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
		m_surface_recreation_pending = true;
		return;
	}
	if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
		std::cerr << "[SYNC] vkAcquireNextImageKHR failed: " << static_cast<int>(acquire_result)
		          << "\n";
		return;
	}

	VkResult reset_cmd_result = vkResetCommandBuffer(command_ctx.command_buffer, 0);
	if (reset_cmd_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkResetCommandBuffer failed: " << static_cast<int>(reset_cmd_result)
		          << "\n";
		drain_acquire_semaphore(frame_idx);
		return;
	}

	VkCommandBufferBeginInfo begin_info{};
	begin_info.sType      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult begin_result = vkBeginCommandBuffer(command_ctx.command_buffer, &begin_info);
	if (begin_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkBeginCommandBuffer failed: " << static_cast<int>(begin_result)
		          << "\n";
		drain_acquire_semaphore(frame_idx);
		return;
	}

	VkCommandBuffer cmd = command_ctx.command_buffer;

	VkImageLayout storage_old = render_target_ctx.storage_image_initialized ?
	                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
	                                VK_IMAGE_LAYOUT_UNDEFINED;
	transition_layout(cmd, render_target_ctx.storage_image, storage_old, VK_IMAGE_LAYOUT_GENERAL, 0,
	                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	VkImageLayout swap_old = swapchain_ctx.image_initialized[image_index] ?
	                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
	                             VK_IMAGE_LAYOUT_UNDEFINED;
	transition_layout(cmd, swapchain_ctx.images[image_index], swap_old,
	                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, active_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, active_mode.layout, 0, 1,
	                        &render_target_ctx.descriptor_set, 0, nullptr);

	if (m_needs_visibility_pass) {
		m_frame_number = 0;
	}

	const uint32_t hipr_frames_per_object =
	    sanitize_hipr_ten_step_value(selection.hipr_debug.frames_per_object);

	PathTracerPushConstants pc{};
	pc.frame               = m_frame_number;
	pc.material_count      = scene_ctx.material_count;
	pc.selected_mesh_index = selection.selected_mesh_index;
	pc.outline_width       = selection.outline_width;
	pc.debug_view_mode     = static_cast<int32_t>(selection.debug_view_mode);
	pc.outline_color       = selection.outline_color;
	pc.spp                 = std::clamp(selection.path_tracing.spp, 1u, 1024u);
	pc.max_bounces         = std::clamp(selection.path_tracing.max_bounces, 1u, 1024u);
	pc.enable_tonemapping  = render_post.enable_tonemapping ? 1u : 0u;
	pc.exposure_bias       = render_post.exposure_bias;
	pc.hipr_object_count   = scene_ctx.mesh_count;
	pc.hipr_top_k = std::min(scene_ctx.hipr_top_k, std::max(selection.hipr_debug.rank_count, 1u));
	pc.hipr_render_rank          = -1;
	pc.hipr_incremental_sort     = selection.hipr_debug.incremental_sorting ? 1u : 0u;
	pc.hipr_clear_order          = m_hipr_force_clear_order ? 1u : 0u;
	pc.hipr_vis_enable_tint      = selection.hipr_debug.vis_enable_influence_tint ? 1u : 0u;
	pc.hipr_vis_rainbow_tint     = selection.hipr_debug.vis_rainbow_tint ? 1u : 0u;
	pc.hipr_frames_per_object    = hipr_frames_per_object;
	pc.hipr_score_blend          = std::clamp(selection.hipr_debug.score_blend, 0.05f, 1.0f);
	pc.hipr_vis_tint_strength    = std::clamp(selection.hipr_debug.vis_tint_strength, 0.0f, 1.0f);
	pc.skybox_enabled            = selection.lighting.skybox_enabled ? 1u : 0u;
	pc.directional_light_enabled = selection.lighting.directional_light_enabled ? 1u : 0u;
	{
		const float elev = selection.lighting.sun_elevation_deg * (3.14159265f / 180.0f);
		const float azim = selection.lighting.sun_azimuth_deg * (3.14159265f / 180.0f);
		pc.sun_dir_x     = std::cos(elev) * std::sin(azim);
		pc.sun_dir_y     = std::sin(elev);
		pc.sun_dir_z     = std::cos(elev) * std::cos(azim);
	}
	pc.sun_intensity = selection.lighting.sun_intensity;

	const uint32_t dispatch_w = (swapchain_ctx.extent.width + 15) / 16;
	const uint32_t dispatch_h = (swapchain_ctx.extent.height + 15) / 16;

	const auto render_mode   = selection.debug_view_mode;
	const bool hipr_mode     = render_mode == ui::RenderDebugViewMode::HiPR;
	const bool obj_id_mode   = render_mode == ui::RenderDebugViewMode::ObjectIds;
	const bool hipr_vis_mode = render_mode == ui::RenderDebugViewMode::HiPRVis;
	const bool naive_mode    = render_mode == ui::RenderDebugViewMode::Naive;

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

	const bool hipr_active = selection.selected_mesh_index >= 0;
	const bool use_hipr_ranked =
	    (hipr_mode || hipr_vis_mode) && hipr_active && !camera_moving_this_frame;
	const bool hipr_object_sampling_enabled =
	    hipr_mode && hipr_active && m_hipr_object_sampling_active && !camera_moving_this_frame;
	const bool hipr_full_scene_sampling =
	    hipr_mode && hipr_active && m_hipr_object_sampling_done && !camera_moving_this_frame;

	bool       run_stage1 = true;
	bool       run_stage2 = camera_moving_this_frame || !m_material_edit_mode || hipr_active;
	bool       run_single_rank_stage4 = false;
	bool       run_ranked_stage4_loop = false;
	int32_t    single_rank_stage4_id  = -1;
	const bool hipr_vis_reveal_complete =
	    hipr_vis_mode && pc.hipr_top_k > 0u &&
	    (m_frame_number / hipr_frames_per_object) >= (pc.hipr_top_k - 1u);
	if (obj_id_mode || naive_mode) {
		run_stage1 = false;
		run_stage2 = true;
	} else if (hipr_vis_mode) {
		run_stage1 = hipr_active;
		run_stage2 = true;
	}

	if (hipr_object_sampling_enabled) {
		const uint32_t sample_rank =
		    (pc.hipr_top_k > 0u) ? std::min(m_hipr_object_sampling_rank, pc.hipr_top_k - 1u) : 0u;
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
		run_stage1             = false;
		run_stage2             = true;
		run_ranked_stage4_loop = false;
	} else {
		if (hipr_vis_mode && hipr_vis_reveal_complete) {
			run_stage1 = false;
			run_stage2 = true;
		}
		run_ranked_stage4_loop =
		    use_hipr_ranked && run_stage2 && !(hipr_vis_mode && hipr_vis_reveal_complete);
	}

	if ((!hipr_mode || !hipr_active) &&
	    (m_hipr_object_sampling_active || m_hipr_object_sampling_done)) {
		resetHiPRObjectSampling();
	}

	bool hipr_refresh = use_hipr_ranked && m_needs_visibility_pass;
	if (hipr_full_scene_sampling) {
		hipr_refresh = false;
	}

	if (use_hipr_ranked && hipr_refresh) {
		pc.stage = 10;
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
		const uint32_t clear_count = std::max(pc.hipr_object_count, std::max(pc.hipr_top_k, 1u));
		vkCmdDispatch(cmd, (clear_count + 15) / 16, 1, 1);
		m_hipr_force_clear_order = false;
	}

	if (m_needs_visibility_pass) {
		pc.stage = 0;
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
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

		m_needs_visibility_pass = false;
	}

	if (run_stage1) {
		pc.stage            = 1;
		pc.hipr_render_rank = hipr_full_scene_sampling ? -2 : -1;
		pc.frame            = hipr_object_sampling_enabled ?
		                          m_hipr_object_sampling_frame :
		                          (hipr_full_scene_sampling ? m_hipr_full_scene_frame : m_frame_number);
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
		vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
	}

	if (use_hipr_ranked && hipr_refresh) {
		VkMemoryBarrier stats_barrier{};
		stats_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		stats_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		stats_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &stats_barrier, 0, nullptr,
		                     0, nullptr);

		pc.stage = 3;
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
		vkCmdDispatch(cmd, 1, 1, 1);
	}

	if (run_stage1 && (run_stage2 || use_hipr_ranked)) {
		VkMemoryBarrier stage_barrier{};
		stage_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		stage_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		stage_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &stage_barrier, 0, nullptr,
		                     0, nullptr);
	}

	if (run_single_rank_stage4) {
		pc.stage            = 4;
		pc.frame            = m_hipr_object_sampling_frame;
		pc.hipr_render_rank = single_rank_stage4_id;
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
		vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
	}

	if (run_ranked_stage4_loop) {
		uint32_t stage4_end_rank = pc.hipr_top_k;
		if (hipr_vis_mode) {
			const uint32_t reveal_rank_count =
			    std::min(pc.hipr_top_k, 1u + (m_frame_number / hipr_frames_per_object));
			stage4_end_rank = reveal_rank_count;
		}

		for (uint32_t rank = 1; rank < stage4_end_rank; ++rank) {
			pc.stage            = 4;
			pc.frame            = m_frame_number;
			pc.hipr_render_rank = static_cast<int32_t>(rank);
			vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
			                   &pc);
			vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
		}
	}

	if (run_stage2) {
		pc.stage            = 2;
		pc.hipr_render_rank = hipr_full_scene_sampling ? -2 : -1;
		pc.frame            = hipr_full_scene_sampling ? m_hipr_full_scene_frame : m_frame_number;
		if (hipr_full_scene_sampling) {
			pc.debug_view_mode = static_cast<int32_t>(ui::RenderDebugViewMode::Naive);
		}
		vkCmdPushConstants(cmd, active_mode.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
		                   &pc);
		vkCmdDispatch(cmd, dispatch_w, dispatch_h, 1);
	}

	if (hipr_object_sampling_enabled && pc.hipr_top_k > 0u) {
		++m_hipr_object_sampling_frame;
		if (m_hipr_object_sampling_frame >= hipr_frames_per_object) {
			m_hipr_object_sampling_frame = 0;
			++m_hipr_object_sampling_rank;
			if (m_hipr_object_sampling_rank >= pc.hipr_top_k) {
				m_hipr_object_sampling_active = false;
				m_hipr_object_sampling_done   = true;
				m_hipr_object_sampling_rank   = 0;
				m_hipr_object_sampling_frame  = 0;
				m_hipr_full_scene_frame       = 0;
			}
		}
	}
	if (hipr_full_scene_sampling) {
		++m_hipr_full_scene_frame;
	}

	VkMemoryBarrier compute_to_transfer{};
	compute_to_transfer.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	compute_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	compute_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0, 1, &compute_to_transfer, 0, nullptr, 0, nullptr);

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
	blit_region.srcOffsets[1]                 = {static_cast<int32_t>(swapchain_ctx.extent.width),
	                                             static_cast<int32_t>(swapchain_ctx.extent.height), 1};

	blit_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	blit_region.dstSubresource.mipLevel       = 0;
	blit_region.dstSubresource.baseArrayLayer = 0;
	blit_region.dstSubresource.layerCount     = 1;
	blit_region.dstOffsets[0]                 = {0, 0, 0};
	blit_region.dstOffsets[1]                 = {static_cast<int32_t>(swapchain_ctx.extent.width),
	                                             static_cast<int32_t>(swapchain_ctx.extent.height), 1};

	vkCmdBlitImage(cmd, render_target_ctx.storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	               swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
	               &blit_region, VK_FILTER_NEAREST);

	render_target_ctx.storage_image_initialized  = true;
	swapchain_ctx.image_initialized[image_index] = true;

	transition_layout(cmd, swapchain_ctx.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	VkRenderPassBeginInfo rp_begin{};
	rp_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rp_begin.renderPass        = overlay_ctx.render_pass;
	rp_begin.framebuffer       = overlay_ctx.framebuffers[image_index];
	rp_begin.renderArea.offset = {0, 0};
	rp_begin.renderArea.extent = swapchain_ctx.extent;

	vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	vkCmdEndRenderPass(cmd);

	VkResult end_result = vkEndCommandBuffer(cmd);
	if (end_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkEndCommandBuffer failed: " << static_cast<int>(end_result) << "\n";
		drain_acquire_semaphore(frame_idx);
		return;
	}

	VkResult reset_fence_result = vkResetFences(vulkan_ctx.device, 1, &sync_ctx.in_flight);
	if (reset_fence_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkResetFences failed: " << static_cast<int>(reset_fence_result)
		          << "\n";
		drain_acquire_semaphore(frame_idx);
		return;
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

		vkDestroyFence(vulkan_ctx.device, sync_ctx.in_flight, nullptr);
		VkFenceCreateInfo signaled_fence_info{};
		signaled_fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		signaled_fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		if (vkCreateFence(vulkan_ctx.device, &signaled_fence_info, nullptr, &sync_ctx.in_flight) !=
		    VK_SUCCESS) {
			throw std::runtime_error("failed to recreate in-flight fence");
		}
		return;
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
		m_surface_recreation_pending = true;
		return;
	}
	if (present_result != VK_SUCCESS) {
		std::cerr << "[SYNC] vkQueuePresentKHR failed: " << static_cast<int>(present_result)
		          << "\n";
		return;
	}

	++m_frame_number;
}

simulation::WaterSurfaceCreateInfo Runtime::Impl::waterSurfaceCreateInfo() const {
	return simulation::WaterSurfaceCreateInfo{
	    .device        = vulkan_ctx.device,
	    .allocator     = render_target_ctx.allocator,
	    .output_extent = swapchain_ctx.extent,
	};
}

void Runtime::Impl::uploadMaterial(int material_index, const GPUMaterial& material) {
	if (scene_ctx.material_mapped == nullptr || material_index < 0 ||
	    material_index >= static_cast<int>(scene_ctx.material_count)) {
		return;
	}

	auto* gpu_materials           = reinterpret_cast<GPUMaterial*>(scene_ctx.material_mapped);
	gpu_materials[material_index] = material;
	vmaFlushAllocation(render_target_ctx.allocator, scene_ctx.material_alloc,
	                   static_cast<VkDeviceSize>(material_index) * sizeof(GPUMaterial),
	                   sizeof(GPUMaterial));
}

bool Runtime::Impl::rebuildPipeline(ui::RenderDebugViewMode mode) {
	const bool rebuilt = ::rebuild_pipeline(mode);
	if (rebuilt) {
		m_frame_number           = 0;
		m_needs_visibility_pass  = true;
		m_hipr_force_clear_order = true;
		resetHiPRObjectSampling();
	}
	return rebuilt;
}

void Runtime::Impl::updateCamera(const GPUCamera& camera) {
	if (!m_has_current_camera || std::memcmp(&m_current_camera, &camera, sizeof(GPUCamera)) != 0) {
		m_current_camera     = camera;
		m_has_current_camera = true;
		m_camera_changed     = true;
	}

	if (scene_ctx.camera_mapped != nullptr) {
		memcpy(scene_ctx.camera_mapped, &camera, sizeof(GPUCamera));
	}
}

void Runtime::Impl::waitIdle() {
	if (vulkan_ctx.device != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(vulkan_ctx.device);
	}
}

Runtime::Runtime(core::Window& window, const Scene& scene) :
    m_impl(std::make_unique<Impl>(window, scene)) {
}

Runtime::~Runtime() = default;

RuntimeFrameOutput Runtime::beginFrame() {
	return m_impl->beginFrame();
}

void Runtime::renderFrame(const RuntimeFrameInput& input) {
	m_impl->renderFrame(input);
}

simulation::WaterSurfaceCreateInfo Runtime::waterSurfaceCreateInfo() const {
	return m_impl->waterSurfaceCreateInfo();
}

void Runtime::uploadMaterial(int material_index, const GPUMaterial& material) {
	m_impl->uploadMaterial(material_index, material);
}

bool Runtime::rebuildPipeline(ui::RenderDebugViewMode mode) {
	return m_impl->rebuildPipeline(mode);
}

void Runtime::updateCamera(const GPUCamera& camera) {
	m_impl->updateCamera(camera);
}

void Runtime::waitIdle() {
	m_impl->waitIdle();
}
