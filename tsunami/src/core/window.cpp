#include "tsunami/core/window.h"
#include <stdexcept>

namespace core {

Window::Window(const WindowConfig& config) : m_config(config) {
	if (!glfwInit())
		throw std::runtime_error("glfwInit failed");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, m_config.resizable ? GLFW_TRUE : GLFW_FALSE);

	m_window = glfwCreateWindow(static_cast<int>(config.width), static_cast<int>(config.height),
	                            config.title.c_str(), nullptr, nullptr);
	if (!m_window)
		throw std::runtime_error("glfwCreateWindow failed");

	m_width  = config.width;
	m_height = config.height;

	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, framebuffer_size_cb);
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

void Window::toggle_fullscreen() {
	if (!m_fullscreen) {
		// Save current windowed state before going fullscreen
		glfwGetWindowPos(m_window, &m_windowed_x, &m_windowed_y);
		glfwGetWindowSize(m_window, &m_windowed_w, &m_windowed_h);

		GLFWmonitor*       mon  = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(mon);
		glfwSetWindowMonitor(m_window, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
		m_fullscreen = true;
	} else {
		glfwSetWindowMonitor(m_window, nullptr, m_windowed_x, m_windowed_y, m_windowed_w,
		                     m_windowed_h, 0);
		m_fullscreen = false;
	}
}

void Window::maximize() {
	if (!m_fullscreen)
		glfwMaximizeWindow(m_window);
}

/*static*/ void Window::framebuffer_size_cb(GLFWwindow* w, int width, int height) {
	auto* win      = static_cast<Window*>(glfwGetWindowUserPointer(w));
	win->m_width   = static_cast<uint32_t>(width);
	win->m_height  = static_cast<uint32_t>(height);
	win->m_resized = true;
}

}        // namespace core
