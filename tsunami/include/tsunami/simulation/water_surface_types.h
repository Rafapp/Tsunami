#pragma once

namespace simulation {

struct WaterSurfaceSettings {
	float propagation          = 0.34f;
	float damping              = 0.008f;
	float restoring_force      = 0.110f;
	float height_scale         = 16.0f;
	float ripple_radius        = 0.012f;
	float base_impulse         = 0.0f;
	float audio_impulse_scale  = 0.028f;
	float orbit_radius         = 0.0f;
	float orbit_speed          = 0.0f;
	float impulse_frequency_hz = 2.20f;
};

struct WaterSurfaceDiagnostics {
	float audio_drive_level = 0.0f;
	float impulse_strength  = 0.0f;
	float emitter_u         = 0.5f;
	float emitter_v         = 0.5f;
};

}        // namespace simulation
