#pragma once

#include <cstdint>

#include <glm/vec4.hpp>

#ifndef VK_NO_PROTOTYPES
#	define VK_NO_PROTOTYPES
#endif

#include "vk_mem_alloc.h"
#include "volk.h"

#include "tsunami/simulation/water_scene_support.h"

class Scene;

namespace simulation {
class WaterSurfaceSimulation;
}

namespace vulkan::waterbridge {

struct WaterSurfaceParamsGpu {
	glm::vec4 center_trace_half_height    = glm::vec4(0.0f);
	glm::vec4 axis_u_half_extent          = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 axis_v_half_extent          = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
	glm::vec4 normal                      = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	int32_t   water_object_id             = -1;
	uint32_t  water_enabled               = 0;
	int32_t   water_material_index        = -1;
	float     water_height_to_world_scale = 1.0f;
	int32_t   first_floating_object_id    = -1;
	uint32_t  _pad0                       = 0;
	uint32_t  _pad1                       = 0;
	uint32_t  _pad2                       = 0;
};

struct WaterSurfaceParamsBufferContext {
	void*         params_mapped = nullptr;
	VmaAllocator  allocator     = nullptr;
	VmaAllocation params_alloc  = VK_NULL_HANDLE;
};

void updateWaterSurfaceParamsBuffer(const WaterSurfaceParamsBufferContext&      context,
                                    const simulation::WaterSurfaceRenderPlacement& placement,
                                    const Scene*                                   scene,
                                    const simulation::WaterSurfaceSimulation*      water_surface,
                                    int                                            first_floating_object_id);

void updateWaterSurfaceImageDescriptors(VkDevice device, VkDescriptorSet descriptor_set,
                                        const simulation::WaterSurfaceSimulation* water_surface);

}        // namespace vulkan::waterbridge
