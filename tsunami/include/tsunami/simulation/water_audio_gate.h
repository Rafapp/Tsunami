// Purpose: Audio-gating policy interface for auto-pausing or resuming water simulation.
#pragma once

#include "tsunami/simulation/water_surface_types.h"

namespace simulation {

float gatedWaterLevel(float water_audio_level);

bool shouldAutoPauseWaterWhenCalm(WaterDriveMode drive_mode, float water_audio_level,
                                  bool currently_auto_paused);

}        // namespace simulation
