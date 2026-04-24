#pragma once

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_simulation.h"
#include "tsunami/ui/audience_control_panel.h"

namespace vulkan::framepolicy {

struct AudioWaterUpdateOutput {
	float audio_level            = 0.0f;
	float water_audio_level      = 0.0f;
	bool  effective_water_paused = false;
};

void handleGuiVisibilityHotkey(bool f1_pressed, bool& show_all_gui, bool& show_control_panel,
                               bool& show_selection_panel);

void applyOverlayLevelFromAudio(float value, ui::AudienceControlPanelState& controls);

AudioWaterUpdateOutput updateAudioWaterPolicy(
    audio::ReactiveAudioController*     audio_controller,
    simulation::WaterSurfaceSimulation* water_surface, ui::AudienceControlPanelState& controls,
    ui::AudienceDiagnostics& diagnostics, const audio::ReactiveAudioInputFrame& input_frame,
    float time_seconds, float delta_time, float& selection_voice_loudness, bool& auto_water_paused);

}        // namespace vulkan::framepolicy
