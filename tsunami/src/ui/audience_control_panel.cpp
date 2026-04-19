#include <algorithm>
#include <cfloat>
#include <cstdint>
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

	ImGui::TextUnformatted("F1 toggles all demo UI.");
	changed |= ImGui::Checkbox("Show audience overlay", &state.show_overlay);
	ImGui::SameLine();
	if (ImGui::Button("Reset defaults")) {
		state                         = AudienceControlPanelState{};
		state.reset_water_requested   = true;
		state.reset_objects_requested = true;
		changed                       = true;
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Render Stats");
	if (diagnostics.render.frame_sample_count == 0) {
		ImGui::TextUnformatted("Collecting frame timing samples...");
	} else {
		if (ImGui::BeginTable("render_stats_summary", 5,
		                      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV)) {
			ImGui::TableSetupColumn("Metric");
			ImGui::TableSetupColumn("Current");
			ImGui::TableSetupColumn("Avg");
			ImGui::TableSetupColumn("Min");
			ImGui::TableSetupColumn("Max");
			ImGui::TableHeadersRow();

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted("FPS");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%7.1f", diagnostics.render.current_fps);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%7.1f", diagnostics.render.average_fps);
			ImGui::TableSetColumnIndex(3);
			ImGui::Text("%7.1f", diagnostics.render.min_fps);
			ImGui::TableSetColumnIndex(4);
			ImGui::Text("%7.1f", diagnostics.render.max_fps);

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted("Frame (ms)");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%7.2f", diagnostics.render.current_frame_time_ms);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%7.2f", diagnostics.render.average_frame_time_ms);
			ImGui::TableSetColumnIndex(3);
			ImGui::Text("%7.2f", diagnostics.render.min_frame_time_ms);
			ImGui::TableSetColumnIndex(4);
			ImGui::Text("%7.2f", diagnostics.render.max_frame_time_ms);

			ImGui::EndTable();
		}
		ImGui::Text("Timing window: %u frames", diagnostics.render.frame_sample_count);

		if (ImGui::BeginTable("render_pass_breakdown", 3,
		                      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV)) {
			ImGui::TableSetupColumn("Pass");
			ImGui::TableSetupColumn("MS");
			ImGui::TableSetupColumn("FPS");
			ImGui::TableHeadersRow();

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted("Sim");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%7.2f", diagnostics.render.simulation_pass_time_ms);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%7.2f", diagnostics.render.simulation_pass_fps);

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted("Render");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%7.2f", diagnostics.render.render_pass_time_ms);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%7.2f", diagnostics.render.render_pass_fps);

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted("Total");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%7.2f", diagnostics.render.total_pass_time_ms);
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%7.2f", diagnostics.render.total_pass_fps);

			ImGui::EndTable();
		}
	}

	if (ImGui::CollapsingHeader("Advanced Stats")) {
		ImGui::Text("Render target: %u x %u", diagnostics.render.render_width,
		            diagnostics.render.render_height);
		ImGui::Text("Swapchain images: %u", diagnostics.render.swapchain_image_count);
		ImGui::Text("Surface grid: %u x %u (%llu samples)", diagnostics.water.grid_width,
		            diagnostics.water.grid_height,
		            static_cast<unsigned long long>(diagnostics.water.sample_count));
		ImGui::Text("Equivalent mesh: %llu quads | %llu triangles",
		            static_cast<unsigned long long>(diagnostics.water.cell_count),
		            static_cast<unsigned long long>(diagnostics.water.triangle_count));
		ImGui::Text("Compute dispatch: %u x %u workgroups", diagnostics.water.dispatch_groups_x,
		            diagnostics.water.dispatch_groups_y);
		ImGui::Text("Ping-pong height images: %u", diagnostics.water.history_image_count);
		ImGui::Text("ImGui geometry: %d vertices | %d indices | %d windows",
		            diagnostics.render.imgui_vertex_count, diagnostics.render.imgui_index_count,
		            diagnostics.render.imgui_window_count);
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Color Pipeline");
	changed |= ImGui::Checkbox("Enable tonemapping", &state.render_post.enable_tonemapping);
	changed |=
	    ImGui::SliderFloat("Exposure bias", &state.render_post.exposure_bias, 0.1f, 8.0f, "%.2f");

	ImGui::Separator();
	ImGui::TextUnformatted("Audio");

	int         input_mode    = static_cast<int>(state.audio.input_mode);
	const char* input_items[] = {"Automatic", "Demo wave", "Manual level"};
	changed |= ImGui::Combo("Input source", &input_mode, input_items, IM_ARRAYSIZE(input_items));
	state.audio.input_mode = static_cast<audio::ReactiveAudioInputMode>(input_mode);
	changed |= ImGui::SliderFloat("Noise floor", &state.audio.noise_floor, 0.0f, 0.100f, "%.3f");
	changed |= ImGui::SliderFloat("Sensitivity", &state.audio.sensitivity, 1.0f, 100.0f, "%.1f");
	changed |= ImGui::SliderFloat("Smoothing", &state.audio.smoothing, 0.01f, 1.0f, "%.2f");
	changed |=
	    ImGui::SliderFloat("Demo cycle (Hz)", &state.audio.demo_cycle_hz, 0.10f, 6.0f, "%.2f");

	ImGui::BeginDisabled(state.audio.input_mode != audio::ReactiveAudioInputMode::Manual);
	changed |= ImGui::SliderFloat("Manual level", &state.audio.manual_level, 0.0f, 1.0f, "%.2f");
	ImGui::EndDisabled();

	ImGui::Separator();
	ImGui::TextUnformatted("Water Surface");
	changed |= ImGui::SliderFloat("Propagation", &state.water.propagation, 0.01f, 0.28f, "%.3f");
	changed |= ImGui::SliderFloat("Damping", &state.water.damping, 0.004f, 0.12f, "%.3f");
	changed |=
	    ImGui::SliderFloat("Restoring force", &state.water.restoring_force, 0.0f, 0.35f, "%.3f");
	changed |= ImGui::SliderFloat("Height scale", &state.water.height_scale, 1.0f, 40.0f, "%.1f");
	changed |=
	    ImGui::SliderFloat("Ripple radius", &state.water.ripple_radius, 0.005f, 0.20f, "%.3f");
	changed |= ImGui::SliderFloat("Base impulse", &state.water.base_impulse, 0.0f, 0.015f, "%.4f");
	changed |=
	    ImGui::SliderFloat("Audio impulse", &state.water.audio_impulse_scale, 0.0f, 0.080f, "%.4f");
	changed |= ImGui::SliderFloat("Emitter orbit (0=fixed)", &state.water.orbit_radius, 0.0f, 0.45f,
	                              "%.2f");
	changed |= ImGui::SliderFloat("Orbit speed (Hz)", &state.water.orbit_speed, 0.0f, 1.5f, "%.2f");
	changed |= ImGui::SliderFloat("Impulse rate (Hz)", &state.water.impulse_frequency_hz, 0.1f,
	                              8.0f, "%.2f");
	if (ImGui::Button(state.water_paused ? "Resume water simulation" : "Pause water simulation")) {
		state.water_paused = !state.water_paused;
		changed            = true;
	}
	ImGui::SameLine();
	ImGui::TextUnformatted(state.water_paused ? "Paused at current frame" : "Live");
	if (ImGui::Button("Reset water state")) {
		state.reset_water_requested = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset object positions")) {
		state.reset_objects_requested = true;
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Performance");
	changed |= ImGui::SliderFloat("Render scale", &state.render_scale, 0.50f, 1.0f, "%.2f");

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

	ImGui::TextUnformatted("Diagnostics");
	ImGui::Text("Microphone: %s", diagnostics.audio.source_name.c_str());
	if (diagnostics.audio.source_available) {
		ImGui::TextUnformatted("Status: live capture active");
	} else {
		ImGui::TextUnformatted("Status: microphone unavailable, auto mode falls back to demo");
	}
	ImGui::TextWrapped("%s", diagnostics.audio.source_status.c_str());
	drawMeter("Raw microphone RMS", diagnostics.audio.raw_level);
	drawMeter("Normalized input", diagnostics.audio.normalized_level);
	drawMeter("Displayed level", state.overlay.volume_level);
	drawMeter("Water drive", diagnostics.water.audio_drive_level);
	drawMeter("Water impulse", diagnostics.water.impulse_strength * 12.0f);
	ImGui::Text("Emitter UV: (%.2f, %.2f)", diagnostics.water.emitter_u,
	            diagnostics.water.emitter_v);

	const uint32_t selected_index =
	    state.overlay.selection_count == 0 ?
	        0 :
	        std::min(state.overlay.selected_index, state.overlay.selection_count - 1) + 1;
	ImGui::Text("Selected segment: %u / %u", selected_index, state.overlay.selection_count);

	ImGui::End();
	return changed;
}

}        // namespace ui
