#pragma once

#include <memory>
#include <string>

class App {
  public:
	explicit App(const std::string& scene_argument = "");
	~App();

	App(const App&)            = delete;
	App& operator=(const App&) = delete;
	App(App&&) noexcept;
	App& operator=(App&&) noexcept;

	void run();

  private:
	class Impl;
	std::unique_ptr<Impl> m_impl;
};
