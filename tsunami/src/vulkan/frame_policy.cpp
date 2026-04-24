#include "tsunami/vulkan/frame_policy.h"

#include <algorithm>

#include "tsunami/audio/audio_input_utils.h"
#include "tsunami/simulation/water_audio_gate.h"
#include "tsunami/ui/audience_overlay.h"

namespace vulkan::framepolicy {

void handleGuiVisibilityHotkey(bool f1_pressed, bool& show_all_gui, bool& show_control_panel,
                               bool& show_selection_panel) {
	if (!f1_pressed) {
		return;
	}

	show_all_gui = !show_all_gui;
	if (show_all_gui) {
		show_control_panel  = true;
		show_selection_panel = true;
	}
}

void applyOverlayLevelFromAudio(float value, ui::AudienceControlPanelState& controls) {
	controls.overlay.volume_level   = std::clamp(value, 0.0f, 1.0f);
	controls.overlay.selected_index = ui::quantizeSelection(
	    controls.overlay.volume_level, controls.overlay.selection_count);
}

AudioWaterUpdateOutput updateAudioWaterPolicy(
    audio::ReactiveAudioController*       audio_controller,
    simulation::WaterSurfaceSimulation*   water_surface,
    ui::AudienceControlPanelState&        controls,
    ui::AudienceDiagnostics&              diagnostics,
    const audio::ReactiveAudioInputFrame& input_frame,
    float                                  time_seconds,
    float                                  delta_time,
    float&                                 selection_voice_loudness,
    bool&                                  auto_water_paused) {
	AudioWaterUpdateOutput output{};

	if (audio_controller != nullptr) {
		output.audio_level = audio_controller->update(controls.audio, input_frame, delta_time);
		diagnostics.audio  = audio_controller->diagnostics();
	}

	selection_voice_loudness = audio::updateSelectionVoiceLoudness(
	    controls.audio, input_frame, selection_voice_loudness, delta_time);

	output.water_audio_level =
	    controls.water_voice_control_enabled ? diagnostics.audio.smoothed_level : 0.0f;
	auto_water_paused = simulation::shouldAutoPauseWaterWhenCalm(
	    controls.water.drive_mode, output.water_audio_level, auto_water_paused);
	output.effective_water_paused = controls.water_paused || auto_water_paused;

	applyOverlayLevelFromAudio(output.audio_level, controls);
	if (water_surface != nullptr && !output.effective_water_paused) {
		diagnostics.water =
		    water_surface->prepareFrame(controls.water, output.water_audio_level, time_seconds, delta_time);
	} else {
		diagnostics.water.audio_drive_level = 0.0f;
		diagnostics.water.impulse_strength  = 0.0f;
	}

	return output;
}

}        // namespace vulkan::framepolicy
