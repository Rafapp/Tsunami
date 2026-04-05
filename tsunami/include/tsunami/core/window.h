#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>

namespace core {

struct WindowConfig {
	uint32_t    width  = 1280;
	uint32_t    height = 720;
	std::string title  = "tsunami 🌊";
};

class Window {
  public:
	explicit Window(const WindowConfig& config = {});
	~Window();

	Window(const Window&)            = delete;
	Window& operator=(const Window&) = delete;

	bool shouldClose() const;
	void pollEvents() const;

	GLFWwindow* handle() const {
		return m_window;
	}
	uint32_t width() const {
		return m_width;
	}
	uint32_t height() const {
		return m_height;
	}

	// Resize / fullscreen
	bool was_resized() const {
		return m_resized;
	}
	void clear_resized() {
		m_resized = false;
	}
	void toggle_fullscreen();
	void maximize();

  private:
	static void framebuffer_size_cb(GLFWwindow* w, int width, int height);

	WindowConfig m_config;
	GLFWwindow*  m_window  = nullptr;
	uint32_t     m_width   = 0;
	uint32_t     m_height  = 0;
	bool         m_resized = false;

	// Saved windowed state so we can restore when leaving fullscreen
	bool m_fullscreen = false;
	int  m_windowed_x = 100;
	int  m_windowed_y = 100;
	int  m_windowed_w = 1280;
	int  m_windowed_h = 720;
};

}        // namespace core