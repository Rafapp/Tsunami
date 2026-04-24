// Purpose: Public Vulkan runtime interface that owns renderer startup and main-loop execution.
#pragma once

#include <memory>
#include <string>

namespace audio {
class MicrophoneInput;
class ReactiveAudioController;
}        // namespace audio

namespace core {
class Window;
}

namespace simulation {
class WaterSurfaceSimulation;
}

class Scene;

namespace vulkan {

class Runtime {
  public:
	explicit Runtime(const std::string& scene_argument = "");
	~Runtime();

	Runtime(const Runtime&)            = delete;
	Runtime& operator=(const Runtime&) = delete;
	Runtime(Runtime&&)                 = delete;
	Runtime& operator=(Runtime&&)      = delete;

	void runMainLoop();

  private:
	void createSwapchainResources();
	void destroySwapchainResources();
	void recreateSwapchainResources();
	void runLoop();

	std::unique_ptr<audio::ReactiveAudioController>     m_audio_controller;
	std::unique_ptr<audio::MicrophoneInput>             m_microphone;
	std::unique_ptr<simulation::WaterSurfaceSimulation> m_water_surface;
	std::unique_ptr<core::Window>                       m_window;
	std::unique_ptr<Scene>                              m_scene;
};

}        // namespace vulkan
