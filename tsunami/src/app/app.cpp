#include "tsunami/app/app.h"

#include <memory>
#include <utility>

#include "tsunami/vulkan/vulkan.h"

class App::Impl {
  public:
	explicit Impl(const std::string& scene_argument) :
	    runtime(std::make_unique<vulkan::Runtime>(scene_argument)) {
	}

	std::unique_ptr<vulkan::Runtime> runtime;
};

App::App(const std::string& scene_argument) : m_impl(std::make_unique<Impl>(scene_argument)) {
}

App::~App() = default;

App::App(App&&) noexcept            = default;
App& App::operator=(App&&) noexcept = default;

void App::run() {
	m_impl->runtime->runMainLoop();
}
