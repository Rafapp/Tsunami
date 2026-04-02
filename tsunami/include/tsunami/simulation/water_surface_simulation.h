#pragma once

#include <cstdint>

#include "volk.h"
#include "vk_mem_alloc.h"

#include "tsunami/simulation/water_surface_types.h"

namespace simulation {

struct WaterSurfaceCreateInfo {
	VkDevice     device        = VK_NULL_HANDLE;
	VmaAllocator allocator     = nullptr;
	VkExtent2D   output_extent = {};
};

class WaterSurfaceSimulation {
  public:
	explicit WaterSurfaceSimulation(const WaterSurfaceCreateInfo& create_info);
	~WaterSurfaceSimulation();

	WaterSurfaceSimulation(const WaterSurfaceSimulation&)            = delete;
	WaterSurfaceSimulation& operator=(const WaterSurfaceSimulation&) = delete;

	const WaterSurfaceDiagnostics& prepareFrame(const WaterSurfaceSettings& settings,
	                                            float audio_level, float time_seconds,
	                                            float delta_time);
	void requestReset();
	void record(VkCommandBuffer command_buffer);

	VkImage outputImage() const {
		return m_output_image;
	}

  private:
	void createImages();
	void createDescriptors();
	void createPipeline();

	VkDevice     m_device        = VK_NULL_HANDLE;
	VmaAllocator m_allocator     = nullptr;
	VkExtent2D   m_output_extent = {};

	VkImage       m_output_image       = VK_NULL_HANDLE;
	VmaAllocation m_output_allocation  = nullptr;
	VkImageView   m_output_image_view  = VK_NULL_HANDLE;

	VkImage       m_height_images[2]      = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	VmaAllocation m_height_allocations[2] = {nullptr, nullptr};
	VkImageView   m_height_image_views[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

	VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool      m_descriptor_pool       = VK_NULL_HANDLE;
	VkDescriptorSet       m_descriptor_sets[2]    = {VK_NULL_HANDLE, VK_NULL_HANDLE};

	VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
	VkPipeline       m_pipeline        = VK_NULL_HANDLE;

	WaterSurfaceDiagnostics m_diagnostics{};

	uint32_t m_active_descriptor_set_index = 0;
	bool     m_height_layout_initialized   = false;
	bool     m_output_in_transfer_src      = false;
	bool     m_clear_history_requested     = true;
	float    m_previous_audio_level        = 0.0f;
	float    m_recent_activity             = 0.0f;
	float    m_emission_accumulator        = 0.0f;
	float    m_pending_impulse             = 0.0f;
	float    m_last_prepare_time           = -1.0f;

	struct PushConstants;
	PushConstants* m_push_constants = nullptr;
};

}        // namespace simulation
