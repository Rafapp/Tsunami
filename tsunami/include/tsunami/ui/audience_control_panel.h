#pragma once

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_types.h"
#include "tsunami/ui/audience_overlay.h"

namespace ui {

struct AudienceDiagnostics {
	audio::ReactiveAudioDiagnostics  audio{};
	simulation::WaterSurfaceDiagnostics water{};
};

struct AudienceControlPanelState {
	bool                         show_overlay           = true;
	bool                         reset_water_requested  = false;
	audio::ReactiveAudioSettings audio{};
	simulation::WaterSurfaceSettings water{};
	AudienceOverlayState         overlay{};
	AudienceOverlayStyle         style{};
};

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics);

}        // namespace ui
