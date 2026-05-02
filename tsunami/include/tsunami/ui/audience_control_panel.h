#pragma once

#include <cstdint>

#include "tsunami/audio/reactive_audio_controller.h"
#include "tsunami/simulation/water_surface_types.h"
#include "tsunami/ui/audience_overlay.h"
#include "tsunami/ui/render_settings.h"

namespace ui {

struct AudienceDiagnostics {
	audio::ReactiveAudioDiagnostics     audio{};
	AudienceRenderDiagnostics           render{};
	simulation::WaterSurfaceDiagnostics water{};
};

struct AudienceControlPanelState {
	bool                             show_overlay            = true;
	bool                             reset_water_requested   = false;
	bool                             reset_objects_requested = false;
	audio::ReactiveAudioSettings     audio{};
	AudienceRenderPostSettings       render_post{};
	simulation::WaterSurfaceSettings water{};
	AudienceOverlayState             overlay{};
	AudienceOverlayStyle             style{};
};

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics);

}        // namespace ui
