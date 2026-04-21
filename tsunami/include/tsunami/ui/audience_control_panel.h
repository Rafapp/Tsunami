#pragma once

#include <cstdint>

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_types.h"
#include "tsunami/ui/audience_overlay.h"

namespace ui {

struct AudienceRenderPostSettings {
	bool  enable_tonemapping = false;
	float exposure_bias      = 2.0f;
};

struct AudienceRenderDiagnostics {
	float    current_fps             = 0.0f;
	float    average_fps             = 0.0f;
	float    min_fps                 = 0.0f;
	float    max_fps                 = 0.0f;
	float    current_frame_time_ms   = 0.0f;
	float    average_frame_time_ms   = 0.0f;
	float    min_frame_time_ms       = 0.0f;
	float    max_frame_time_ms       = 0.0f;
	float    simulation_pass_time_ms = 0.0f;
	float    render_pass_time_ms     = 0.0f;
	float    total_pass_time_ms      = 0.0f;
	float    simulation_pass_fps     = 0.0f;
	float    render_pass_fps         = 0.0f;
	float    total_pass_fps          = 0.0f;
	uint32_t frame_sample_count      = 0;
	uint32_t render_width            = 0;
	uint32_t render_height           = 0;
	uint32_t swapchain_image_count   = 0;
	int      imgui_vertex_count      = 0;
	int      imgui_index_count       = 0;
	int      imgui_window_count      = 0;
};

struct AudienceDiagnostics {
	audio::ReactiveAudioDiagnostics     audio{};
	AudienceRenderDiagnostics           render{};
	simulation::WaterSurfaceDiagnostics water{};
};

struct AudienceControlPanelState {
	bool                             show_overlay            = true;
	bool                             water_paused            = false;
	bool                             water_voice_control_enabled = true;
	bool                             reset_water_requested   = false;
	bool                             reset_objects_requested = false;
	float                            render_scale            = 0.60f;
	audio::ReactiveAudioSettings     audio{};
	AudienceRenderPostSettings       render_post{};
	simulation::WaterSurfaceSettings water{};
	AudienceOverlayState             overlay{};
	AudienceOverlayStyle             style{};
};

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics);

}        // namespace ui
