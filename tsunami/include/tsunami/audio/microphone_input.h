// Purpose: Microphone capture interface that exposes normalized live levels and device status.
#pragma once

#include <string>

namespace audio {

class MicrophoneInput {
  public:
	MicrophoneInput();
	~MicrophoneInput();

	MicrophoneInput(const MicrophoneInput&)            = delete;
	MicrophoneInput& operator=(const MicrophoneInput&) = delete;

	bool               isAvailable() const;
	float              latestLevel() const;
	const std::string& statusMessage() const;
	const std::string& deviceName() const;

  private:
	struct Impl;
	Impl* m_impl = nullptr;
};

}        // namespace audio
