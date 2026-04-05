#include <stdexcept>

#include "tsunami/core/window.h"

namespace core {

Window::Window(const WindowConfig& config) : m_config(config) {
	if (!glfwInit())
		throw std::runtime_error("glfwInit failed");

	// No OpenGL context, using Vulkan
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, m_config.resizable ? GLFW_TRUE : GLFW_FALSE);

	m_window = glfwCreateWindow(static_cast<int>(m_config.width), static_cast<int>(m_config.height),
	                            m_config.title.c_str(), nullptr, nullptr);

	if (!m_window)
		throw std::runtime_error("glfwCreateWindow failed");
}

Window::~Window() {
	if (m_window)
		glfwDestroyWindow(m_window);
	glfwTerminate();
}

bool Window::shouldClose() const {
	return glfwWindowShouldClose(m_window);
}

void Window::pollEvents() const {
	glfwPollEvents();
}

void Window::waitEvents() const {
	glfwWaitEvents();
}

}        // namespace core
