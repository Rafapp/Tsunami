#pragma once

#include <memory>
#include <stdexcept>

#include "tsunami/core/window.h"

namespace audio {
class MicrophoneInput;
class ReactiveAudioController;
}        // namespace audio

namespace simulation {
class WaterSurfaceSimulation;
}

class App {
  public:
	App();
	~App();

	void run();

  private:
	void                                                createSwapchainResources();
	void                                                destroySwapchainResources();
	void                                                recreateSwapchainResources();
	void                                                MainLoop();
	std::unique_ptr<audio::ReactiveAudioController>     m_audio_controller;
	std::unique_ptr<audio::MicrophoneInput>             m_microphone;
	std::unique_ptr<simulation::WaterSurfaceSimulation> m_water_surface;
	std::unique_ptr<core::Window>                       m_window;
};
