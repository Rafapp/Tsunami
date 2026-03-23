#pragma once
#define GLFW_INCLUDE_VULKAN

#include <string>

#include <GLFW/glfw3.h>

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

    // non-copyable, non-movable
    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    bool        shouldClose() const;
    void        pollEvents()  const;

    GLFWwindow* handle()      const { return m_window; }
    uint32_t    width()       const { return m_config.width; }
    uint32_t    height()      const { return m_config.height; }

private:
    WindowConfig m_config;
    GLFWwindow*  m_window = nullptr;
};

}