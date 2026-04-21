#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace simulation {

constexpr uint32_t kMaxFloatingObjects            = 8;
constexpr uint32_t kMaxFloatingObjectInteractions = kMaxFloatingObjects * 8;

struct WaterSurfaceSettings {
	float propagation          = 0.22f;
	float damping              = 0.012f;
	float restoring_force      = 0.110f;
	float height_scale         = 16.0f;
	float ripple_radius        = 0.012f;
	float base_impulse         = 0.0f;
	float audio_impulse_scale  = 0.020f;
	float orbit_radius         = 0.0f;
	float orbit_speed          = 0.0f;
	float impulse_frequency_hz = 1.80f;
};

struct WaterSurfaceDiagnostics {
	float    audio_drive_level   = 0.0f;
	float    impulse_strength    = 0.0f;
	float    emitter_u           = 0.5f;
	float    emitter_v           = 0.5f;
	uint32_t grid_width          = 0;
	uint32_t grid_height         = 0;
	uint32_t dispatch_groups_x   = 0;
	uint32_t dispatch_groups_y   = 0;
	uint32_t history_image_count = 0;
	uint64_t sample_count        = 0;
	uint64_t cell_count          = 0;
	uint64_t triangle_count      = 0;
};

struct FloatingObjectSettings {
	glm::vec2 anchor                = glm::vec2(0.0f);
	float     base_height           = 0.0f;
	float     base_yaw_radians      = 0.0f;
	glm::vec3 size                  = glm::vec3(0.25f, 0.12f, 0.18f);
	float     mass                  = 0.9f;
	glm::vec3 color                 = glm::vec3(0.85f, 0.55f, 0.30f);
	float     buoyancy_strength     = 30.0f;
	float     buoyancy_damping      = 7.5f;
	float     linear_damping        = 1.8f;
	float     angular_strength      = 12.0f;
	float     angular_damping       = 5.5f;
	float     self_righting         = 6.5f;
	float     max_tilt_radians      = 0.35f;
	float     planar_drift_strength = 2.4f;
	float     planar_damping        = 1.4f;
	float     anchor_pull_strength  = 0.45f;
	float     drift_radius          = 0.48f;
	float     footprint_roundness   = 0.85f;
	float     footprint_hole_ratio  = 0.0f;
	float     waterline_offset      = 0.01f;
	float     yaw_follow_strength   = 2.2f;
};

struct alignas(16) FloatingObjectRenderData {
	glm::vec4 center{};
	glm::vec4 axis_x{};
	glm::vec4 axis_y{};
	glm::vec4 axis_z{};
	glm::vec4 half_extents{};
	glm::vec4 color{};
};

struct alignas(16) FloatingObjectInteractionData {
	glm::vec4 center_radius_strength{};
	glm::vec4 wake_direction_length{};
};

}        // namespace simulation
