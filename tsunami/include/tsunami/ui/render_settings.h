#pragma once

#include <cstdint>

namespace ui {

struct AudienceRenderPostSettings {
	bool  enable_tonemapping = false;
	float exposure_bias      = 2.0f;
};

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

}        // namespace ui
