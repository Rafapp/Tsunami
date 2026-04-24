#include "tsunami/vulkan/floating_policy.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>

#include <glm/glm.hpp>

namespace vulkan::floating {

namespace {

glm::vec2 floatingAnchorForIndex(uint32_t index) {
	static constexpr std::array<glm::vec2, simulation::kMaxFloatingObjects> kAnchors = {
	    glm::vec2(-0.52f, -0.28f), glm::vec2(0.46f, -0.30f), glm::vec2(-0.18f, 0.10f),
	    glm::vec2(0.28f, 0.16f),   glm::vec2(-0.46f, 0.32f), glm::vec2(0.06f, -0.46f),
	    glm::vec2(0.54f, 0.34f),   glm::vec2(-0.08f, 0.46f),
	};
	return kAnchors[std::min<size_t>(index, kAnchors.size() - 1)];
}

float floatingYawForIndex(uint32_t index) {
	static constexpr std::array<float, simulation::kMaxFloatingObjects> kYaws = {
	    15.0f, -24.0f, 36.0f, -48.0f, 62.0f, -80.0f, 102.0f, -128.0f,
	};
	return glm::radians(kYaws[std::min<size_t>(index, kYaws.size() - 1)]);
}

float floatingDesiredDraftFraction(const std::string& asset_name_lower) {
	(void) asset_name_lower;
	// Keep all floaters at the same draft ratio as ring floaties.
	return 0.18f;
}

}        // namespace

int firstFloatingObjectId(const std::vector<FloatingMeshGroup>& groups) {
	int first_object_id = -1;
	for (const FloatingMeshGroup& group : groups) {
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

float floatingTargetMajorWorldSize(const simulation::WaterSurfaceRenderPlacement& placement,
                                   const std::string& asset_name_lower) {
	const float pool_span_world = 2.0f * std::min(placement.half_extent_u, placement.half_extent_v);
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

simulation::FloatingObjectSettings makeFloatingObjectSettings(
    const simulation::WaterSurfaceRenderPlacement& placement, const std::string& asset_name,
    const glm::vec3& asset_world_size, float default_scale, uint32_t simulation_index) {
	std::string asset_name_lower = asset_name;
	std::transform(
	    asset_name_lower.begin(), asset_name_lower.end(), asset_name_lower.begin(),
	    [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
	const bool is_ring   = asset_name_lower.find("ring") != std::string::npos;
	const bool is_duck   = asset_name_lower.find("duck") != std::string::npos;
	const bool is_teapot = asset_name_lower.find("teapot") != std::string::npos;
	const glm::vec3 scaled_world_size =
	    glm::max(asset_world_size * default_scale, glm::vec3(0.03f, 0.03f, 0.03f));
	glm::vec3 effective_world_size = scaled_world_size;
	if (is_duck) {
		// The rubber duck asset has an attached tether/tag that inflates its bounds,
		// so compact the physical hull to avoid unstable torques and collisions.
		const float planar_mean    = 0.5f * (scaled_world_size.x + scaled_world_size.z);
		const float compact_planar = std::max(planar_mean * 0.78f, 0.03f);
		effective_world_size.x     = compact_planar;
		effective_world_size.z     = compact_planar;
	}

	simulation::FloatingObjectSettings settings{};
	settings.anchor           = floatingAnchorForIndex(simulation_index);
	settings.base_height      = 0.02f + effective_world_size.y * 0.10f;
	settings.base_yaw_radians = floatingYawForIndex(simulation_index);
	settings.size.x = effective_world_size.x / std::max(placement.half_extent_u, 1.0e-4f);
	settings.size.z = effective_world_size.z / std::max(placement.half_extent_v, 1.0e-4f);
	settings.size.y = effective_world_size.y;
	const float volume = effective_world_size.x * effective_world_size.y * effective_world_size.z;
	settings.mass      = std::clamp(volume * 30.0f, 0.35f, 1.80f);
	settings.color     = glm::vec3(0.86f, 0.58f, 0.28f);
	const float desired_draft = std::max(
	    settings.size.y * floatingDesiredDraftFraction(asset_name_lower), settings.size.y * 0.12f);
	settings.buoyancy_strength =
	    std::clamp((4.5f * settings.mass) / std::max(desired_draft * 5.0f, 1.0e-4f), 16.0f, 46.0f);
	settings.buoyancy_damping      = 7.5f;
	settings.linear_damping        = 1.8f;
	settings.angular_strength      = 9.5f;
	settings.angular_damping       = 5.5f;
	settings.self_righting         = 4.5f;
	settings.max_tilt_radians      = 0.42f;
	settings.planar_drift_strength = 2.4f;
	settings.planar_damping        = 1.4f;
	settings.anchor_pull_strength  = 0.45f;
	settings.drift_radius          = 0.56f;
	settings.footprint_roundness = is_ring ? 1.0f : (is_teapot ? 0.72f : (is_duck ? 0.86f : 0.90f));
	settings.footprint_hole_ratio = is_ring ? 0.52f : 0.0f;
	settings.waterline_offset     = -settings.size.y * 0.08f;
	settings.yaw_follow_strength  = 2.2f;
	if (is_duck) {
		settings.buoyancy_damping      = 11.5f;
		settings.linear_damping        = 3.2f;
		settings.angular_strength      = 5.0f;
		settings.angular_damping       = 10.5f;
		settings.self_righting         = 8.0f;
		settings.max_tilt_radians      = 0.24f;
		settings.planar_drift_strength = 1.1f;
		settings.planar_damping        = 3.0f;
		settings.anchor_pull_strength  = 0.95f;
		settings.drift_radius          = 0.22f;
		settings.waterline_offset      = -settings.size.y * 0.06f;
		settings.yaw_follow_strength   = 1.2f;
	}
	return settings;
}

glm::mat4 makeFloatingWorldPose(const simulation::WaterSurfaceRenderPlacement& placement,
                                const simulation::FloatingObjectSettings&       settings) {
	const glm::vec3 axis_y = placement.normal;
	glm::vec3       axis_x = placement.axis_u;
	glm::vec3       axis_z = placement.axis_v;
	const float     c      = std::cos(settings.base_yaw_radians);
	const float     s      = std::sin(settings.base_yaw_radians);
	const glm::vec3 rot_x  = glm::normalize(axis_x * c + axis_z * s);
	const glm::vec3 rot_z  = glm::normalize(-axis_x * s + axis_z * c);
	const glm::vec3 center = placement.center + placement.axis_u * (settings.anchor.x * placement.half_extent_u) +
	                         placement.axis_v * (settings.anchor.y * placement.half_extent_v) +
	                         placement.normal * settings.base_height;

	glm::mat4 pose(1.0f);
	pose[0] = glm::vec4(rot_x, 0.0f);
	pose[1] = glm::vec4(axis_y, 0.0f);
	pose[2] = glm::vec4(rot_z, 0.0f);
	pose[3] = glm::vec4(center, 1.0f);
	return pose;
}

glm::mat4 makeFloatingWorldPose(const simulation::WaterSurfaceRenderPlacement& placement,
                                const simulation::FloatingObjectRenderData&     render_data) {
	const auto to_world_axis = [&](const glm::vec3& axis) {
		return glm::normalize(placement.axis_u * axis.x + placement.normal * axis.y +
		                      placement.axis_v * axis.z);
	};

	const glm::vec3 center = placement.center + placement.axis_u * (render_data.center.x * placement.half_extent_u) +
	                         placement.axis_v * (render_data.center.z * placement.half_extent_v) +
	                         placement.normal * render_data.center.y;

	glm::mat4 pose(1.0f);
	pose[0] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_x)), 0.0f);
	pose[1] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_y)), 0.0f);
	pose[2] = glm::vec4(to_world_axis(glm::vec3(render_data.axis_z)), 0.0f);
	pose[3] = glm::vec4(center, 1.0f);
	return pose;
}

}        // namespace vulkan::floating
