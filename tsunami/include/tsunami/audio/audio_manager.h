#pragma once

#include "tsunami/audio/microphone_input.h"
#include "tsunami/audio/reactive_audio_controller.h"

namespace audio {

class AudioManager {
  public:
	AudioManager();
	~AudioManager() = default;

	AudioManager(const AudioManager&)            = delete;
	AudioManager& operator=(const AudioManager&) = delete;

	ReactiveAudioInputFrame captureFrame(float time_seconds) const;

  private:
	MicrophoneInput m_microphone;
};

}        // namespace audio
