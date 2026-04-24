#include "tsunami/simulation/water_audio_gate.h"

#include <algorithm>

namespace simulation {

namespace {

float clampUnit(float value) {
	return std::clamp(value, 0.0f, 1.0f);
}

}        // namespace

float gatedWaterLevel(float water_audio_level) {
	return clampUnit((water_audio_level - 0.010f) / 0.990f);
}

bool shouldAutoPauseWaterWhenCalm(WaterDriveMode drive_mode, float water_audio_level,
                                  bool currently_auto_paused) {
	if (drive_mode != WaterDriveMode::ArtistLinear) {
		return false;
	}

	const float     gated_level      = gatedWaterLevel(water_audio_level);
	constexpr float kPauseThreshold  = 0.002f;
	constexpr float kResumeThreshold = 0.010f;
	if (currently_auto_paused) {
		return gated_level < kResumeThreshold;
	}
	return gated_level <= kPauseThreshold;
}

}        // namespace simulation
