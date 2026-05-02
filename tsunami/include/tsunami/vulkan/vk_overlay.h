#pragma once

// Forward-declare GLFWwindow to avoid pulling in GLFW in every TU
struct GLFWwindow;

void create_overlay_render_pass();
void destroy_overlay_render_resources();

void initialize_imgui_context(GLFWwindow* window);
void initialize_imgui_renderer();
void shutdown_imgui_renderer();
void shutdown_imgui();
