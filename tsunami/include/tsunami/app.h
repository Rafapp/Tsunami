#pragma once

#include <stdexcept>
#include <memory>

#include "volk.h"
#include "VkBootstrap.h"

#include "tsunami/core/window.h"

class App {
  public:
	App();
	~App() = default;

	void run();

  private:
	void MainLoop();

	std::unique_ptr<core::Window> m_window;
};