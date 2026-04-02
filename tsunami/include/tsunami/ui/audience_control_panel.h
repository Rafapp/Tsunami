#pragma once

#include "tsunami/ui/audience_overlay.h"

namespace ui {

enum class AudienceInputMode : int {
	Automatic = 0,
	Demo      = 1,
	Manual    = 2,
};

struct AudienceAudioSettings {
	AudienceInputMode input_mode    = AudienceInputMode::Automatic;
	float             noise_floor   = 0.015f;
	float             sensitivity   = 50.0f;
	float             smoothing     = 0.18f;
	float             demo_cycle_hz = 1.35f;
	float             manual_level  = 0.5f;
};

struct AudienceDiagnostics {
	bool        microphone_available = false;
	float       raw_microphone_level = 0.0f;
	float       normalized_level     = 0.0f;
	const char* microphone_name      = "Unavailable";
	const char* microphone_status    = "Microphone capture is unavailable.";
};

struct AudienceControlPanelState {
	bool                  show_overlay = true;
	AudienceAudioSettings audio{};
	AudienceOverlayState  overlay{};
	AudienceOverlayStyle  style{};
};

bool drawAudienceControlPanel(bool* is_open, AudienceControlPanelState& state,
                              const AudienceDiagnostics& diagnostics);

}        // namespace ui
