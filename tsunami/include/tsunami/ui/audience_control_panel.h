#pragma once

#include <cstdint>

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_types.h"
#include "tsunami/ui/audience_overlay.h"

namespace ui {

struct AudienceRenderDiagnostics {
	float    current_fps           = 0.0f;
	float    average_fps           = 0.0f;
	float    min_fps               = 0.0f;
	float    max_fps               = 0.0f;
	float    current_frame_time_ms = 0.0f;
	float    average_frame_time_ms = 0.0f;
	float    min_frame_time_ms     = 0.0f;
	float    max_frame_time_ms     = 0.0f;
	uint32_t frame_sample_count    = 0;
	uint32_t render_width          = 0;
	uint32_t render_height         = 0;
	uint32_t swapchain_image_count = 0;
	int      imgui_vertex_count    = 0;
	int      imgui_index_count     = 0;
	int      imgui_window_count    = 0;
};

struct AudienceDiagnostics {
	audio::ReactiveAudioDiagnostics     audio{};
	AudienceRenderDiagnostics           render{};
	simulation::WaterSurfaceDiagnostics water{};
};

struct AudienceControlPanelState {
	bool                             show_overlay            = true;
	bool                             reset_water_requested   = false;
	bool                             reset_objects_requested = false;
	audio::ReactiveAudioSettings     audio{};
	simulation::WaterSurfaceSettings water{};
	AudienceOverlayState             overlay{};
	AudienceOverlayStyle             style{};
};

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics);

}        // namespace ui
