#include <algorithm>
#include <cmath>

#include "tsunami/audio/reactive_audio_controller.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

float clamp01(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

float calculateDemoLevel(float time_seconds, float cycle_hz) {
	const float angular_frequency = std::max(cycle_hz, 0.0f) * kPi * 2.0f;
	return 0.5f + (0.45f * std::sin(time_seconds * angular_frequency));
}

float normalizeMicrophoneLevel(float raw_level, const audio::ReactiveAudioSettings& settings) {
	const float noise_floor = clamp01(settings.noise_floor);
	const float sensitivity = std::max(settings.sensitivity, 0.0f);
	return clamp01((raw_level - noise_floor) * sensitivity);
}

}        // namespace

namespace audio {

float ReactiveAudioController::update(const ReactiveAudioSettings&   settings,
                                      const ReactiveAudioInputFrame& input_frame) {
	m_diagnostics.source_available = input_frame.source_available;
	m_diagnostics.raw_level        = input_frame.source_available ? input_frame.raw_level : 0.0f;
	m_diagnostics.source_name      = input_frame.source_name;
	m_diagnostics.source_status    = input_frame.source_status;

	float target_level = 0.0f;
	switch (settings.input_mode) {
		case ReactiveAudioInputMode::Automatic:
			target_level = input_frame.source_available ?
			                   normalizeMicrophoneLevel(input_frame.raw_level, settings) :
			                   calculateDemoLevel(input_frame.time_seconds, settings.demo_cycle_hz);
			break;
		case ReactiveAudioInputMode::Demo:
			target_level = calculateDemoLevel(input_frame.time_seconds, settings.demo_cycle_hz);
			break;
		case ReactiveAudioInputMode::Manual:
			target_level = clamp01(settings.manual_level);
			break;
	}

	const float smoothing          = std::clamp(settings.smoothing, 0.01f, 1.0f);
	m_diagnostics.normalized_level = clamp01(target_level);
	m_diagnostics.smoothed_level +=
	    (m_diagnostics.normalized_level - m_diagnostics.smoothed_level) * smoothing;
	m_diagnostics.smoothed_level = clamp01(m_diagnostics.smoothed_level);

	return m_diagnostics.smoothed_level;
}

void ReactiveAudioController::reset() {
	m_diagnostics.raw_level        = 0.0f;
	m_diagnostics.normalized_level = 0.0f;
	m_diagnostics.smoothed_level   = 0.0f;
}

}        // namespace audio
