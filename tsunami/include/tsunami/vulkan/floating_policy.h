#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "tsunami/simulation/water_scene_support.h"
#include "tsunami/simulation/water_surface_types.h"

namespace vulkan::floating {

struct FloatingMeshGroup {
	std::string      display_name;
	std::vector<int> mesh_indices;
	int              simulation_index = -1;
};

int firstFloatingObjectId(const std::vector<FloatingMeshGroup>& groups);

float floatingTargetMajorWorldSize(const simulation::WaterSurfaceRenderPlacement& placement,
                                   const std::string& asset_name_lower);

simulation::FloatingObjectSettings makeFloatingObjectSettings(
    const simulation::WaterSurfaceRenderPlacement& placement, const std::string& asset_name,
    const glm::vec3& asset_world_size, float default_scale, uint32_t simulation_index);

glm::mat4 makeFloatingWorldPose(const simulation::WaterSurfaceRenderPlacement& placement,
                                const simulation::FloatingObjectSettings&       settings);

glm::mat4 makeFloatingWorldPose(const simulation::WaterSurfaceRenderPlacement& placement,
                                const simulation::FloatingObjectRenderData&     render_data);

}        // namespace vulkan::floating
