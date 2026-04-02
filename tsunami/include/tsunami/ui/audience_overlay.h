#pragma once

#include <cstdint>

#include "imgui.h"

namespace ui {

struct AudienceOverlayState {
	float    volume_level   = 0.0f;   // Range: [0.0, 1.0]
	uint32_t selected_index = 0;      // Quantized index based on volume level
	uint32_t selection_count = 5;     // Total number of discrete selections
};

uint32_t quantizeSelection(float volume_level, uint32_t selection_count);
void     drawAudienceOverlay(const ImVec2& display_size, const AudienceOverlayState& state);

}        // namespace ui