// Purpose: Internal audio utility declarations shared by frame policy and audio subsystems.
#pragma once

#include "tsunami/audio/reactive_audio_controller.h"

namespace audio {

class MicrophoneInput;

ReactiveAudioInputFrame buildAudioInputFrame(const MicrophoneInput* microphone, float time_seconds);

float updateSelectionVoiceLoudness(const ReactiveAudioSettings&   settings,
                                   const ReactiveAudioInputFrame& input_frame,
                                   float previous_smoothed_loudness, float delta_time);

}        // namespace audio
