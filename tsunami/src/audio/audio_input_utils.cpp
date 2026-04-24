#include "tsunami/audio/audio_input_utils.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "tsunami/audio/microphone_input.h"

namespace audio {

namespace {

float clampUnit(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

float normalizeMicrophoneLevelForSelection(const ReactiveAudioSettings& settings, float raw_level) {
	const float noise_floor = clampUnit(settings.noise_floor);
	const float sensitivity = std::max(settings.sensitivity, 0.0f);
	return clampUnit((raw_level - noise_floor) * sensitivity);
}

}        // namespace

ReactiveAudioInputFrame buildAudioInputFrame(const MicrophoneInput* microphone,
                                             float                  time_seconds) {
	ReactiveAudioInputFrame input_frame{};
	input_frame.source_available = microphone != nullptr && microphone->isAvailable();
	input_frame.raw_level        = input_frame.source_available ? microphone->latestLevel() : 0.0f;
	input_frame.time_seconds     = time_seconds;
	input_frame.source_name =
	    microphone != nullptr ? microphone->deviceName() : std::string("Unavailable");
	input_frame.source_status = microphone != nullptr ?
	                                microphone->statusMessage() :
	                                std::string("Microphone capture is unavailable.");
	return input_frame;
}

float updateSelectionVoiceLoudness(const ReactiveAudioSettings&   settings,
                                   const ReactiveAudioInputFrame& input_frame,
                                   float previous_smoothed_loudness, float delta_time) {
	const float target_loudness =
	    input_frame.source_available ?
	        normalizeMicrophoneLevelForSelection(settings, input_frame.raw_level) :
	        0.0f;
	const float smoothing = std::clamp(settings.smoothing, 0.01f, 1.0f);
	const float dt_steps  = std::clamp(delta_time * 60.0f, 0.0f, 8.0f);
	const float alpha     = 1.0f - std::pow(1.0f - smoothing, dt_steps);
	const float smoothed_loudness =
	    previous_smoothed_loudness + (target_loudness - previous_smoothed_loudness) * alpha;
	return clampUnit(smoothed_loudness);
}

}        // namespace audio
