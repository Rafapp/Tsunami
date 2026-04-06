#include <algorithm>

#include "tsunami/ui/audience_overlay.h"

namespace {

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

}        // namespace

namespace ui {

uint32_t quantizeSelection(float volume_level, uint32_t selection_count) {
	if (selection_count == 0) {
		return 0;
	}

	const float clamped_level = clamp01(volume_level);
	const float scaled_index  = clamped_level * static_cast<float>(selection_count);
	return std::min(static_cast<uint32_t>(scaled_index), selection_count - 1);
}

void drawAudienceOverlay(const ImVec2& display_size, const AudienceOverlayState& state,
                         const AudienceOverlayStyle& style) {
	const float width_ratio = std::clamp(style.overlay_width_ratio, 0.1f, 1.0f);
	const float overlay_width =
	    std::min(display_size.x,
	             std::min(display_size.x * width_ratio, std::max(style.overlay_max_width, 180.0f)));
	const float  overlay_height = std::max(style.overlay_height, 30.0f);
	const ImVec2 overlay_size(overlay_width, overlay_height);
	const float  bottom_inset = std::clamp(style.overlay_bottom_inset, 0.0f,
	                                       std::max(0.0f, display_size.y - overlay_size.y));
	const ImVec2 overlay_position((display_size.x - overlay_size.x) / 2.0f,
	                              display_size.y - overlay_size.y - bottom_inset);

	ImGui::SetNextWindowPos(overlay_position, ImGuiCond_Always);
	ImGui::SetNextWindowSize(overlay_size, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.0f);

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
	                                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
	                                ImGuiWindowFlags_NoBringToFrontOnFocus;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("AudienceOverlay", nullptr, window_flags);

	const float outer_padding     = std::clamp(style.outer_padding, 0.0f, overlay_size.x * 0.45f);
	const float bar_height        = std::clamp(style.bar_height, 6.0f, overlay_size.y);
	const float bar_rounding      = std::max(style.bar_rounding, 0.0f);
	const float outline_thickness = std::max(style.outline_thickness, 1.0f);
	const float marker_thickness  = std::max(style.marker_thickness, 1.0f);
	const float bar_top           = overlay_position.y + (overlay_size.y - bar_height) * 0.5f;
	const float bar_left          = overlay_position.x + outer_padding;
	const float bar_right         = overlay_position.x + overlay_size.x - outer_padding;
	const float bar_width         = bar_right - bar_left;
	const float bar_bottom        = bar_top + bar_height;

	const float clamped_level = clamp01(state.volume_level);
	const float filled_right  = bar_left + (clamped_level * bar_width);

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(filled_right, bar_bottom),
	                         ImGui::ColorConvertFloat4ToU32(style.fill_color), bar_rounding);
	draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
	                   ImGui::ColorConvertFloat4ToU32(style.outline_color), bar_rounding, 0,
	                   outline_thickness);

	if (state.selection_count > 1) {
		for (uint32_t divider = 1; divider < state.selection_count; ++divider) {
			const float divider_x = bar_left + (bar_width * static_cast<float>(divider) /
			                                    static_cast<float>(state.selection_count));
			draw_list->AddLine(ImVec2(divider_x, bar_top + 2.0f),
			                   ImVec2(divider_x, bar_bottom - 2.0f),
			                   ImGui::ColorConvertFloat4ToU32(style.divider_color));
		}
	}

	const uint32_t clamped_selection =
	    state.selection_count == 0 ? 0 : std::min(state.selected_index, state.selection_count - 1);
	const float selection_center = state.selection_count <= 1 ?
	                                   0.5f :
	                                   (static_cast<float>(clamped_selection) + 0.5f) /
	                                       static_cast<float>(state.selection_count);
	const float selection_x      = bar_left + (selection_center * bar_width);

	draw_list->AddLine(ImVec2(selection_x, bar_top), ImVec2(selection_x, bar_bottom),
	                   ImGui::ColorConvertFloat4ToU32(style.marker_color), marker_thickness);

	ImGui::End();
	ImGui::PopStyleVar(2);
}

}        // namespace ui
