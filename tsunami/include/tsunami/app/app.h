#pragma once

#include <memory>
#include <stdexcept>

#include "tsunami/core/window.h"

class App {
  public:
	App();
	~App() = default;

	void run();

  private:
	void                          MainLoop();
	std::unique_ptr<core::Window> m_window;
};