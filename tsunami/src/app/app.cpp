#include "tsunami/app/app.h"

#include <algorithm>
#include <memory>
#include <utility>

#include <GLFW/glfw3.h>

#include "tsunami/audio/audio_manager.h"
#include "tsunami/core/window.h"
#include "tsunami/renderer/renderer.h"

class App::Impl {
  public:
	explicit Impl(const std::string& scene_argument) {
		core::WindowConfig window_config{};
		m_window        = std::make_unique<core::Window>(window_config);
		m_audio_manager = std::make_unique<audio::AudioManager>();
		m_renderer      = std::make_unique<renderer::Renderer>(*m_window, scene_argument);
		m_last_time     = static_cast<float>(glfwGetTime());
	}

	void run() {
		while (!m_window->shouldClose()) {
			m_window->pollEvents();

			if (m_window->width() == 0 || m_window->height() == 0) {
				m_window->waitEvents();
				m_last_time = static_cast<float>(glfwGetTime());
				continue;
			}

			const float time_seconds = static_cast<float>(glfwGetTime());
			const float delta_time   = std::clamp(time_seconds - m_last_time, 0.0001f, 0.1f);
			m_last_time              = time_seconds;

			const auto audio_frame = m_audio_manager->captureFrame(time_seconds);
			m_renderer->tick(time_seconds, delta_time, audio_frame);
		}
	}

	std::unique_ptr<core::Window>        m_window;
	std::unique_ptr<audio::AudioManager> m_audio_manager;
	std::unique_ptr<renderer::Renderer>  m_renderer;
	float                                m_last_time = 0.0f;
};

App::App(const std::string& scene_argument) : m_impl(std::make_unique<Impl>(scene_argument)) {
}

App::~App() = default;

App::App(App&&) noexcept            = default;
App& App::operator=(App&&) noexcept = default;

void App::run() {
	m_impl->run();
}
