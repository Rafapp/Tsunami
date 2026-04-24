// Purpose: Scene-to-simulation helper interface for deriving water placement and context.
#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

#ifndef VK_NO_PROTOTYPES
#	define VK_NO_PROTOTYPES
#endif

#include "volk.h"

class Scene;

namespace simulation {

struct WaterSurfaceRenderPlacement {
	bool      enabled                    = false;
	int32_t   mesh_index                 = -1;
	glm::vec3 center                     = glm::vec3(0.0f);
	float     trace_half_height          = 0.45f;
	glm::vec3 axis_u                     = glm::vec3(1.0f, 0.0f, 0.0f);
	float     half_extent_u              = 1.0f;
	glm::vec3 axis_v                     = glm::vec3(0.0f, 0.0f, 1.0f);
	float     half_extent_v              = 1.0f;
	glm::vec3 normal                     = glm::vec3(0.0f, 1.0f, 0.0f);
	float     floating_surface_bounds    = 0.94f;
	float     floating_boundary_exponent = 2.0f;
};

int waterSurfaceObjectId(const Scene* scene, const WaterSurfaceRenderPlacement& placement);

WaterSurfaceRenderPlacement buildWaterSurfacePlacement(const Scene* scene);

std::vector<float> buildWaterSurfaceDomainMask(const Scene*                       scene,
                                               const WaterSurfaceRenderPlacement& placement,
                                               VkExtent2D                         extent);

void applyDedicatedWaterMaterial(Scene* scene, const WaterSurfaceRenderPlacement& placement);

}        // namespace simulation
