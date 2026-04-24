#include "tsunami/vulkan/water_surface_bridge.h"

#include <cstring>

#include "tsunami/scene/scene.h"
#include "tsunami/simulation/water_scene_support.h"
#include "tsunami/simulation/water_surface_simulation.h"

namespace vulkan::waterbridge {

void updateWaterSurfaceParamsBuffer(const WaterSurfaceParamsBufferContext&         context,
                                    const simulation::WaterSurfaceRenderPlacement& placement,
                                    const Scene*                                   scene,
                                    const simulation::WaterSurfaceSimulation*      water_surface,
                                    int first_floating_object_id) {
	if (context.params_mapped == nullptr) {
		return;
	}

	WaterSurfaceParamsGpu params{};
	params.center_trace_half_height = glm::vec4(placement.center, placement.trace_half_height);
	params.axis_u_half_extent       = glm::vec4(placement.axis_u, placement.half_extent_u);
	params.axis_v_half_extent       = glm::vec4(placement.axis_v, placement.half_extent_v);
	params.normal                   = glm::vec4(placement.normal, 0.0f);
	params.water_object_id          = simulation::waterSurfaceObjectId(scene, placement);
	params.water_enabled            = (water_surface != nullptr && placement.enabled) ? 1u : 0u;
	params.water_material_index =
	    (scene != nullptr && placement.mesh_index >= 0 &&
	     placement.mesh_index < static_cast<int>(scene->m_meshes.size())) ?
	        placement.mesh_index :
	        -1;
	params.water_height_to_world_scale =
	    water_surface != nullptr ? water_surface->heightToWorldScale() : 1.0f;
	params.first_floating_object_id = first_floating_object_id;

	std::memcpy(context.params_mapped, &params, sizeof(params));
	vmaFlushAllocation(context.allocator, context.params_alloc, 0, sizeof(params));
}

void updateWaterSurfaceImageDescriptors(VkDevice device, VkDescriptorSet descriptor_set,
                                        const simulation::WaterSurfaceSimulation* water_surface) {
	if (descriptor_set == VK_NULL_HANDLE || water_surface == nullptr) {
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
	             descriptor_set,
	             20,
	             0,
	             1,
	             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	             &current_height_info};
	writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	             nullptr,
	             descriptor_set,
	             21,
	             0,
	             1,
	             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	             &previous_height_info};
	vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

}        // namespace vulkan::waterbridge
