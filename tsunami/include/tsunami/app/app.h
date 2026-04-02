#pragma once

#include <memory>
#include <stdexcept>

#include "tsunami/core/window.h"

namespace audio {
class MicrophoneInput;
}

class App {
  public:
	App();
	~App();

	void run();

  private:
	void                                    MainLoop();
	std::unique_ptr<audio::MicrophoneInput> m_microphone;
	std::unique_ptr<core::Window>           m_window;
};
