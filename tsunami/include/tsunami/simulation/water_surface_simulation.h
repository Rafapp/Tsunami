#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#ifndef VK_NO_PROTOTYPES
#	define VK_NO_PROTOTYPES
#endif

#include "vk_mem_alloc.h"
#include "volk.h"

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
	void                           requestReset();
	void                           requestObjectReset();
	void                           setFloatingObjects(std::span<const FloatingObjectSettings> objects);
	std::vector<FloatingObjectRenderData> floatingObjectRenderData() const;
	void                           record(VkCommandBuffer command_buffer);

	VkImage outputImage() const {
		return m_output_image;
	}
	VkExtent2D outputExtent() const {
		return m_output_extent;
	}
	VkImageView currentHeightImageView() const {
		return m_height_image_views[m_active_descriptor_set_index];
	}
	VkImageView previousHeightImageView() const {
		return m_height_image_views[1u - m_active_descriptor_set_index];
	}
	float heightToWorldScale() const {
		return m_height_to_world_scale;
	}

  private:
	void createImages();
	void createDescriptors();
	void createPipeline();
	void initializeFloatingObjects();

	VkDevice     m_device        = VK_NULL_HANDLE;
	VmaAllocator m_allocator     = nullptr;
	VkExtent2D   m_output_extent = {};

	VkImage       m_output_image      = VK_NULL_HANDLE;
	VmaAllocation m_output_allocation = nullptr;
	VkImageView   m_output_image_view = VK_NULL_HANDLE;

	VkImage       m_height_images[2]      = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	VmaAllocation m_height_allocations[2] = {nullptr, nullptr};
	VkImageView   m_height_image_views[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

	VkDescriptorSetLayout m_water_descriptor_set_layout  = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_object_descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool      m_descriptor_pool              = VK_NULL_HANDLE;
	VkDescriptorSet       m_water_descriptor_sets[2]     = {VK_NULL_HANDLE, VK_NULL_HANDLE};
	VkDescriptorSet       m_object_descriptor_sets[2]    = {VK_NULL_HANDLE, VK_NULL_HANDLE};

	VkPipelineLayout m_water_pipeline_layout  = VK_NULL_HANDLE;
	VkPipelineLayout m_object_pipeline_layout = VK_NULL_HANDLE;
	VkPipeline       m_water_pipeline         = VK_NULL_HANDLE;
	VkPipeline       m_object_pipeline        = VK_NULL_HANDLE;

	VkBuffer                m_floating_object_settings_buffer         = VK_NULL_HANDLE;
	VmaAllocation           m_floating_object_settings_allocation     = nullptr;
	VkBuffer                m_floating_object_states_buffer           = VK_NULL_HANDLE;
	VmaAllocation           m_floating_object_states_allocation       = nullptr;
	VkBuffer                m_floating_objects_buffer                 = VK_NULL_HANDLE;
	VmaAllocation           m_floating_objects_allocation             = nullptr;
	VkBuffer                m_floating_object_interactions_buffer     = VK_NULL_HANDLE;
	VmaAllocation           m_floating_object_interactions_allocation = nullptr;
	uint32_t                m_floating_object_count                   = 0;
	uint32_t                m_floating_object_interaction_count       = 0;
	WaterSurfaceDiagnostics m_diagnostics{};
	float                   m_height_to_world_scale = 1.0f;

	uint32_t m_active_descriptor_set_index = 0;
	bool     m_height_layout_initialized   = false;
	bool     m_output_in_transfer_src      = false;
	bool     m_clear_history_requested     = true;
	bool     m_reset_objects_requested     = true;
	float    m_previous_audio_level        = 0.0f;
	float    m_recent_activity             = 0.0f;
	float    m_emission_accumulator        = 0.0f;
	float    m_pending_impulse             = 0.0f;
	float    m_last_prepare_time           = -1.0f;

	struct WaterPushConstants;
	struct FloatingObjectPushConstants;
	WaterPushConstants*          m_water_push_constants  = nullptr;
	FloatingObjectPushConstants* m_object_push_constants = nullptr;
};

}        // namespace simulation
