#pragma once

#include <memory>
#include <string>

namespace audio {
struct ReactiveAudioInputFrame;
}

namespace core {
class Window;
}

namespace renderer {

class Renderer {
  public:
	Renderer(core::Window& window, const std::string& scene_argument = "");
	~Renderer();

	Renderer(const Renderer&)            = delete;
	Renderer& operator=(const Renderer&) = delete;
	Renderer(Renderer&&)                 = delete;
	Renderer& operator=(Renderer&&)      = delete;

	void tick(float time_seconds, float delta_time,
	          const audio::ReactiveAudioInputFrame& audio_frame);

  private:
	class Impl;
	std::unique_ptr<Impl> m_impl;
};

}        // namespace renderer
