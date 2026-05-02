#include "tsunami/audio/audio_manager.h"

namespace audio {

AudioManager::AudioManager() = default;

ReactiveAudioInputFrame AudioManager::captureFrame(float time_seconds) const {
	ReactiveAudioInputFrame frame{};
	frame.source_available = m_microphone.isAvailable();
	frame.raw_level        = m_microphone.latestLevel();
	frame.time_seconds     = time_seconds;
	frame.source_name      = m_microphone.deviceName();
	frame.source_status    = m_microphone.statusMessage();
	return frame;
}

}        // namespace audio
