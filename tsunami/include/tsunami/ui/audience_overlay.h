#pragma once

#include <cstdint>

#include "imgui.h"

namespace ui {

struct AudienceOverlayState {
	float    volume_level    = 0.0f;        // Range: [0.0, 1.0]
	uint32_t selected_index  = 0;           // Quantized index based on volume level
	uint32_t selection_count = 5;           // Total number of discrete selections
};

struct AudienceOverlayStyle {
	float  overlay_width_ratio  = 0.82f;
	float  overlay_max_width    = 1100.0f;
	float  overlay_height       = 84.0f;
	float  overlay_bottom_inset = 40.0f;
	float  outer_padding        = 16.0f;
	float  bar_height           = 38.0f;
	float  bar_rounding         = 4.0f;
	float  outline_thickness    = 3.0f;
	float  marker_thickness     = 4.0f;
	ImVec4 fill_color    = ImVec4(18.0f / 255.0f, 18.0f / 255.0f, 18.0f / 255.0f, 220.0f / 255.0f);
	ImVec4 outline_color = ImVec4(10.0f / 255.0f, 10.0f / 255.0f, 10.0f / 255.0f, 1.0f);
	ImVec4 divider_color = ImVec4(0.0f, 0.0f, 0.0f, 55.0f / 255.0f);
	ImVec4 marker_color  = ImVec4(225.0f / 255.0f, 35.0f / 255.0f, 35.0f / 255.0f, 1.0f);
};

uint32_t quantizeSelection(float volume_level, uint32_t selection_count);
void     drawAudienceOverlay(const ImVec2& display_size, const AudienceOverlayState& state,
                             const AudienceOverlayStyle& style);

}        // namespace ui
