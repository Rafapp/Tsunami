#pragma once

#include <string>

namespace audio {

enum class ReactiveAudioInputMode : int {
	Automatic = 0,
	Demo      = 1,
	Manual    = 2,
};

struct ReactiveAudioSettings {
	ReactiveAudioInputMode input_mode    = ReactiveAudioInputMode::Automatic;
	float                  noise_floor   = 0.015f;
	float                  sensitivity   = 50.0f;
	float                  smoothing     = 0.18f;
	float                  demo_cycle_hz = 1.35f;
	float                  manual_level  = 0.5f;
};

struct ReactiveAudioInputFrame {
	bool        source_available = false;
	float       raw_level        = 0.0f;
	float       time_seconds     = 0.0f;
	std::string source_name      = "Unavailable";
	std::string source_status    = "Microphone capture is unavailable.";
};

struct ReactiveAudioDiagnostics {
	bool        source_available = false;
	float       raw_level        = 0.0f;
	float       normalized_level = 0.0f;
	float       smoothed_level   = 0.0f;
	std::string source_name      = "Unavailable";
	std::string source_status    = "Microphone capture is unavailable.";
};

class ReactiveAudioController {
  public:
	float update(const ReactiveAudioSettings& settings, const ReactiveAudioInputFrame& input_frame,
	             float delta_time);
	void  reset();

	const ReactiveAudioDiagnostics& diagnostics() const {
		return m_diagnostics;
	}

  private:
	ReactiveAudioDiagnostics m_diagnostics{};
};

}        // namespace audio
