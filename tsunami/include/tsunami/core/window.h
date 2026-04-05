#pragma once
#define GLFW_INCLUDE_VULKAN

#include <string>

#include <GLFW/glfw3.h>

namespace core {

struct WindowConfig {
	uint32_t width 			= 1280;
	uint32_t height 		= 720;
	std::string title 		= "Tsunami 🌊";
	bool resizable 			= true;
};

class Window {
	public:
	explicit Window(const WindowConfig& config = {});
	~Window();

	// Non-copyable and non-movable
	Window(const Window&)            = delete;
	Window& operator=(const Window&) = delete;

	bool shouldClose() const;
	void pollEvents() const;
	void waitEvents() const;

	GLFWwindow* handle() const {
		return m_window;
	}

	uint32_t width() const {
		if (m_window == nullptr) {
			return m_config.width;
		}

		int framebuffer_width = 0;
		int framebuffer_height = 0;
		glfwGetFramebufferSize(m_window, &framebuffer_width, &framebuffer_height);
		return framebuffer_width > 0 ? static_cast<uint32_t>(framebuffer_width) : 0u;
	}

	uint32_t height() const {
		if (m_window == nullptr) {
			return m_config.height;
		}

		int framebuffer_width = 0;
		int framebuffer_height = 0;
		glfwGetFramebufferSize(m_window, &framebuffer_width, &framebuffer_height);
		return framebuffer_height > 0 ? static_cast<uint32_t>(framebuffer_height) : 0u;
	}

	private:
	WindowConfig  m_config;
	GLFWwindow*   m_window = nullptr;
};

} // namespace core