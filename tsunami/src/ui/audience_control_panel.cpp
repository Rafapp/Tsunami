#include <algorithm>
#include <cfloat>
#include <cstdio>

#include "tsunami/ui/audience_control_panel.h"

namespace {

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

void drawMeter(const char* label, float value) {
	char overlay[32]{};
	std::snprintf(overlay, sizeof(overlay), "%.2f", clamp01(value));
	ImGui::TextUnformatted(label);
	ImGui::ProgressBar(clamp01(value), ImVec2(-FLT_MIN, 0.0f), overlay);
}

}        // namespace

namespace ui {

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics) {
	if (is_open != nullptr && !*is_open) {
		return false;
	}

	bool changed = false;

	ImGui::SetNextWindowPos(ImVec2(24.0f, 24.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(430.0f, 300.0f), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Audience Control Panel", is_open)) {
		ImGui::End();
		return false;
	}

	ImGui::TextUnformatted("F1 toggles this window.");
	changed |= ImGui::Checkbox("Show audience overlay", &state.show_overlay);
	ImGui::SameLine();
	if (ImGui::Button("Reset defaults")) {
		state   = AudienceControlPanelState{};
		changed = true;
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Audio");

	int         input_mode    = static_cast<int>(state.audio.input_mode);
	const char* input_items[] = {"Automatic", "Demo wave", "Manual level"};
	changed |= ImGui::Combo("Input source", &input_mode, input_items, IM_ARRAYSIZE(input_items));
	state.audio.input_mode = static_cast<AudienceInputMode>(input_mode);
	changed |= ImGui::SliderFloat("Noise floor", &state.audio.noise_floor, 0.0f, 0.100f, "%.3f");
	changed |= ImGui::SliderFloat("Sensitivity", &state.audio.sensitivity, 1.0f, 100.0f, "%.1f");
	changed |= ImGui::SliderFloat("Smoothing", &state.audio.smoothing, 0.01f, 1.0f, "%.2f");
	changed |=
	    ImGui::SliderFloat("Demo cycle (Hz)", &state.audio.demo_cycle_hz, 0.10f, 6.0f, "%.2f");

	ImGui::BeginDisabled(state.audio.input_mode != AudienceInputMode::Manual);
	changed |= ImGui::SliderFloat("Manual level", &state.audio.manual_level, 0.0f, 1.0f, "%.2f");
	ImGui::EndDisabled();

	ImGui::Separator();
	ImGui::TextUnformatted("Bar / Slider");

	int selection_count = static_cast<int>(state.overlay.selection_count);
	changed |= ImGui::SliderInt("Segments", &selection_count, 1, 16);
	state.overlay.selection_count = static_cast<uint32_t>(selection_count);
	changed |=
	    ImGui::SliderFloat("Width ratio", &state.style.overlay_width_ratio, 0.20f, 1.0f, "%.2f");
	changed |=
	    ImGui::SliderFloat("Max width", &state.style.overlay_max_width, 240.0f, 2000.0f, "%.0f");
	changed |=
	    ImGui::SliderFloat("Overlay height", &state.style.overlay_height, 40.0f, 220.0f, "%.0f");
	changed |=
	    ImGui::SliderFloat("Bottom inset", &state.style.overlay_bottom_inset, 0.0f, 220.0f, "%.0f");
	changed |= ImGui::SliderFloat("Bar height", &state.style.bar_height, 12.0f, 160.0f, "%.0f");
	changed |= ImGui::SliderFloat("Bar padding", &state.style.outer_padding, 0.0f, 64.0f, "%.0f");
	changed |=
	    ImGui::SliderFloat("Corner rounding", &state.style.bar_rounding, 0.0f, 24.0f, "%.1f");
	changed |= ImGui::SliderFloat("Outline thickness", &state.style.outline_thickness, 1.0f, 10.0f,
	                              "%.1f");
	changed |=
	    ImGui::SliderFloat("Marker thickness", &state.style.marker_thickness, 1.0f, 12.0f, "%.1f");

	ImGui::Separator();
	ImGui::TextUnformatted("Style");
	changed |= ImGui::ColorEdit4("Fill color", reinterpret_cast<float*>(&state.style.fill_color),
	                             ImGuiColorEditFlags_AlphaBar);
	changed |=
	    ImGui::ColorEdit4("Outline color", reinterpret_cast<float*>(&state.style.outline_color),
	                      ImGuiColorEditFlags_AlphaBar);
	changed |=
	    ImGui::ColorEdit4("Divider color", reinterpret_cast<float*>(&state.style.divider_color),
	                      ImGuiColorEditFlags_AlphaBar);
	changed |=
	    ImGui::ColorEdit4("Marker color", reinterpret_cast<float*>(&state.style.marker_color),
	                      ImGuiColorEditFlags_AlphaBar);

	ImGui::Separator();
	ImGui::TextUnformatted("Diagnostics");
	ImGui::Text("Microphone: %s", diagnostics.microphone_name);
	if (diagnostics.microphone_available) {
		ImGui::TextUnformatted("Status: live capture active");
	} else {
		ImGui::TextUnformatted("Status: microphone unavailable, auto mode falls back to demo");
	}
	ImGui::TextWrapped("%s", diagnostics.microphone_status);
	drawMeter("Raw microphone RMS", diagnostics.raw_microphone_level);
	drawMeter("Normalized input", diagnostics.normalized_level);
	drawMeter("Displayed level", state.overlay.volume_level);

	const uint32_t selected_index =
	    state.overlay.selection_count == 0 ?
	        0 :
	        std::min(state.overlay.selected_index, state.overlay.selection_count - 1) + 1;
	ImGui::Text("Selected segment: %u / %u", selected_index, state.overlay.selection_count);

	ImGui::End();
	return changed;
}

}        // namespace ui
