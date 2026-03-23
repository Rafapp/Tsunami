#include <iostream>

#include "tsunami/app.h"

App::App() {
	m_window = std::make_unique<core::Window>(
	    core::WindowConfig{.width = 1280, .height = 720, .title = "tsunami 🌊"});

	std::cout << "[tsunami] window created " << m_window->width() << "x" << m_window->height()
	          << "\n";
}

void App::run() {
	MainLoop();
}

void App::MainLoop() {
	while (!m_window->shouldClose()) {
		m_window->pollEvents();
	}
}