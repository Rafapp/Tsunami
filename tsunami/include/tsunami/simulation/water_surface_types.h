#pragma once

#include <cstdint>

#include <glm/vec4.hpp>

namespace simulation {

constexpr uint32_t kMaxFloatingObjects = 8;
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
	float audio_drive_level = 0.0f;
	float impulse_strength  = 0.0f;
	float emitter_u         = 0.5f;
	float emitter_v         = 0.5f;
	uint32_t grid_width     = 0;
	uint32_t grid_height    = 0;
	uint32_t dispatch_groups_x = 0;
	uint32_t dispatch_groups_y = 0;
	uint32_t history_image_count = 0;
	uint64_t sample_count       = 0;
	uint64_t cell_count         = 0;
	uint64_t triangle_count     = 0;
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
