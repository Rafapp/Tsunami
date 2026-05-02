#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "VkBootstrap.h"
#include "volk.h"

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#	define VMA_STATIC_VULKAN_FUNCTIONS 0
#	define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif
#include "vk_mem_alloc.h"

static constexpr uint32_t NUM_LUTS              = 8;
static constexpr uint32_t MAX_MATERIAL_TEXTURES = 256;
static constexpr uint32_t HIPR_TOP_K            = 16;

struct LutTexture2D {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation alloc  = VK_NULL_HANDLE;
	VkImageView   view   = VK_NULL_HANDLE;
	uint32_t      width  = 0;
	uint32_t      height = 0;
};

struct LutTexture3D {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation alloc  = VK_NULL_HANDLE;
	VkImageView   view   = VK_NULL_HANDLE;
	uint32_t      width  = 0;
	uint32_t      height = 0;
	uint32_t      depth  = 0;
};

struct BLAS {
	VkAccelerationStructureKHR handle         = VK_NULL_HANDLE;
	VkBuffer                   buffer         = VK_NULL_HANDLE;
	VmaAllocation              buffer_alloc   = VK_NULL_HANDLE;
	VkDeviceAddress            device_address = 0;
};

struct PipelineMode {
	const char*      shader_file        = nullptr;
	const char*      label              = nullptr;
	uint32_t         push_constant_size = 0;
	VkPipelineLayout layout             = VK_NULL_HANDLE;
	VkPipeline       pipeline           = VK_NULL_HANDLE;
	bool             compiled           = false;
};

struct SceneContext {
	void*         camera_mapped               = nullptr;
	VkBuffer      camera_buffer               = VK_NULL_HANDLE;
	VmaAllocation camera_alloc                = VK_NULL_HANDLE;
	void*         material_mapped             = nullptr;
	VkBuffer      material_buffer             = VK_NULL_HANDLE;
	VmaAllocation material_alloc              = VK_NULL_HANDLE;
	uint32_t      material_count              = 0;
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
};

struct VulkanContext {
	vkb::Instance       instance{};
	vkb::PhysicalDevice phys_device{};
	vkb::Device         log_device{};
	uint32_t            scratch_alignment     = 0;
	VkDevice            device                = VK_NULL_HANDLE;
	VkSurfaceKHR        surface               = VK_NULL_HANDLE;
	VkQueue             graphics_queue        = VK_NULL_HANDLE;
	uint32_t            graphics_queue_family = 0;
};

struct SwapchainContext {
	vkb::Swapchain           swapchain{};
	std::vector<VkImage>     images;
	std::vector<VkImageView> image_views;
	std::vector<bool>        image_initialized;
	VkFormat                 image_format = VK_FORMAT_UNDEFINED;
	VkExtent2D               extent{};
};

struct RenderResourcesContext {
	VmaAllocator allocator = nullptr;
};

struct RenderTargetContext {
	VmaAllocator                       allocator                 = nullptr;
	VkImage                            storage_image             = VK_NULL_HANDLE;
	VmaAllocation                      storage_image_alloc       = VK_NULL_HANDLE;
	VkImageView                        storage_image_view        = VK_NULL_HANDLE;
	bool                               storage_image_initialized = false;
	VkImage                            object_id_image           = VK_NULL_HANDLE;
	VmaAllocation                      object_id_image_alloc     = VK_NULL_HANDLE;
	VkImageView                        object_id_image_view      = VK_NULL_HANDLE;
	VkImage                            accum_image               = VK_NULL_HANDLE;
	VmaAllocation                      accum_image_alloc         = VK_NULL_HANDLE;
	VkImageView                        accum_image_view          = VK_NULL_HANDLE;
	VkImage                            dummy_image_2d            = VK_NULL_HANDLE;
	VmaAllocation                      dummy_image_2d_alloc      = VK_NULL_HANDLE;
	VkImageView                        dummy_image_2d_view       = VK_NULL_HANDLE;
	VkImage                            dummy_image_3d            = VK_NULL_HANDLE;
	VmaAllocation                      dummy_image_3d_alloc      = VK_NULL_HANDLE;
	VkImageView                        dummy_image_3d_view       = VK_NULL_HANDLE;
	VkSampler                          lut_sampler               = VK_NULL_HANDLE;
	VkSampler                          material_sampler          = VK_NULL_HANDLE;
	std::array<LutTexture2D, NUM_LUTS> lut_textures_2d;
	std::array<LutTexture3D, NUM_LUTS> lut_textures_3d;
	std::vector<VkImage>               mat_images;
	std::vector<VmaAllocation>         mat_allocs;
	std::vector<VkImageView>           mat_views;
	VkDescriptorSetLayout              descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool                   descriptor_pool       = VK_NULL_HANDLE;
	VkDescriptorSet                    descriptor_set        = VK_NULL_HANDLE;
};

struct ComputePipelineContext {
	std::array<PipelineMode, 4> modes{};
};

struct CommandContext {
	VkCommandPool   command_pool   = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

struct SyncContext {
	std::vector<VkSemaphore> image_available;
	std::vector<VkSemaphore> render_finished;
	VkFence                  in_flight = VK_NULL_HANDLE;
};

struct OverlayContext {
	VkRenderPass               render_pass = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> framebuffers;
};

struct AccelerationStructureContext {
	std::vector<BLAS>          blases;
	VkAccelerationStructureKHR tlas              = VK_NULL_HANDLE;
	VkBuffer                   tlas_buffer       = VK_NULL_HANDLE;
	VmaAllocation              tlas_buffer_alloc = VK_NULL_HANDLE;
	VkDeviceAddress            tlas_address      = 0;
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

	void pushFrame(float delta_time_seconds);
};

namespace vulkan::internal {

struct RuntimeContextBundle {
	SceneContext                 scene{};
	VulkanContext                vulkan{};
	SwapchainContext             swapchain{};
	RenderResourcesContext       render_resources{};
	RenderTargetContext          render_targets{};
	ComputePipelineContext       compute{};
	CommandContext               commands{};
	SyncContext                  sync{};
	OverlayContext               overlay{};
	AccelerationStructureContext acceleration{};
	FrameTimingHistory           frame_timing{};
};

RuntimeContextBundle& current_contexts();

}        // namespace vulkan::internal

#define scene_ctx (::vulkan::internal::current_contexts().scene)
#define vulkan_ctx (::vulkan::internal::current_contexts().vulkan)
#define swapchain_ctx (::vulkan::internal::current_contexts().swapchain)
#define render_resources_ctx (::vulkan::internal::current_contexts().render_resources)
#define render_target_ctx (::vulkan::internal::current_contexts().render_targets)
#define compute_ctx (::vulkan::internal::current_contexts().compute)
#define command_ctx (::vulkan::internal::current_contexts().commands)
#define sync_ctx (::vulkan::internal::current_contexts().sync)
#define overlay_ctx (::vulkan::internal::current_contexts().overlay)
#define as_ctx (::vulkan::internal::current_contexts().acceleration)
#define frame_timing_history (::vulkan::internal::current_contexts().frame_timing)
