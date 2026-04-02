#include <algorithm>

#include "tsunami/ui/audience_overlay.h"

namespace {

constexpr float kOverlayWidthRatio 	= 0.82f;
constexpr float kOverlayMaxWidth 	= 1100.0f;
constexpr float kOverlayHeight 		= 84.0f;
constexpr float kOverlayBottomInset = 40.0f;
constexpr float kOuterPadding 		= 16.0f;
constexpr float kBarHeight 			= 38.0f;
constexpr float kBarRounding 		= 4.0f;
constexpr float kOutlineThickness 	= 3.0f;
constexpr float kMarkerThickness 	= 4.0f;

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

namespace ui {

uint32_t quantizeSelection(float volume_level, uint32_t selection_count) {
	if (selection_count == 0) {
		return 0;
	}

	const float clamped_level = clamp01(volume_level);
	const float scaled_index = clamped_level * static_cast<float>(selection_count);
	return std::min(static_cast<uint32_t>(scaled_index), selection_count - 1);
}

void drawAudienceOverlay(const ImVec2& display_size, const AudienceOverlayState& state) {
	const float overlay_width =
		std::min(display_size.x * kOverlayWidthRatio, kOverlayMaxWidth);
	const ImVec2 overlay_size(overlay_width, kOverlayHeight);
	const ImVec2 overlay_position((display_size.x - overlay_size.x) / 2.0f,
									display_size.y - overlay_size.y - kOverlayBottomInset);

	ImGui::SetNextWindowPos(overlay_position, ImGuiCond_Always);
	ImGui::SetNextWindowSize(overlay_size, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.0f);

	ImGuiWindowFlags window_flags =
		ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
		ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
		ImGuiWindowFlags_NoBringToFrontOnFocus;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("AudienceOverlay", nullptr, window_flags);

	const float bar_top = overlay_position.y + (overlay_size.y - kBarHeight) * 0.5f;
	const float bar_left = overlay_position.x + kOuterPadding;
	const float bar_right = overlay_position.x + overlay_size.x - kOuterPadding;
	const float bar_width = bar_right - bar_left;
	const float bar_bottom = bar_top + kBarHeight;

	const float clamped_level = clamp01(state.volume_level);
	const float filled_right = bar_left + (clamped_level * bar_width);

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(filled_right, bar_bottom),
							IM_COL32(18, 18, 18, 220), kBarRounding);
	draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
						IM_COL32(10, 10, 10, 255), kBarRounding, 0, kOutlineThickness);

	if (state.selection_count > 1) {
		for (uint32_t divider = 1; divider < state.selection_count; ++divider) {
			const float divider_x =
				bar_left +(bar_width * static_cast<float>(divider) / 
							static_cast<float>(state.selection_count));
			draw_list->AddLine(ImVec2(divider_x, bar_top + 2.0f),
								ImVec2(divider_x, bar_bottom - 2.0f),
								IM_COL32(0, 0, 0, 55));
		}
	}

	const uint32_t clamped_selection =
		state.selection_count == 0
			? 0
			: std::min(state.selected_index, state.selection_count - 1);
	const float selection_center = 
		state.selection_count <= 1
			? 0.5f
			: (static_cast<float>(clamped_selection) + 0.5f) /
			  	static_cast<float>(state.selection_count);
	const float selection_x = bar_left + (selection_center * bar_width);

	draw_list->AddLine(ImVec2(selection_x, bar_top),
						ImVec2(selection_x, bar_bottom),
						IM_COL32(225, 35, 35, 255), kMarkerThickness);

	ImGui::End();
	ImGui::PopStyleVar(2);
}

} 	// namespace ui
