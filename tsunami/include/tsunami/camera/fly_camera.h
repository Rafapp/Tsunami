#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Forward-declared so callers don't need to include the full camera header
struct GPUCamera;

// ---------------------------------------------------------------------------
//  FlyCamera  –  first-person fly camera driven by GLFW input
//
//  Controls (while right-mouse is held to capture the cursor):
//    Right-click       – toggle mouse capture on/off
//    ESC               – release mouse capture
//    WASD              – move forward / backward / strafe
//    Space / Ctrl      – move up / down
//    Shift (held)      – 4× speed multiplier
//    Q / E             – halve / double the base speed
// ---------------------------------------------------------------------------
class FlyCamera {
  public:
	FlyCamera() = default;

	// Construct from an existing scene camera position and look-at target
	FlyCamera(glm::vec3 position, glm::vec3 target, float fov_deg = 60.f, float speed = 10.f);

	// Call once per frame.  dt is seconds elapsed since last frame.
	// Returns true if the camera moved (so the caller can reset accumulation).
	bool update(GLFWwindow* window, float dt);

	// Pack current state into the GPU-side camera buffer struct
	GPUCamera pack() const;

	// ---- tweakable parameters ----
	float speed       = 0.5f;           // units / second
	float sensitivity = 0.0018f;        // radians / pixel
	float fov         = 60.f;           // vertical field-of-view in degrees

  private:
	glm::vec3 m_position{0.f, 50.f, 0.f};
	float     m_yaw   = 0.f;        // radians, around world-Y
	float     m_pitch = 0.f;        // radians, clamped to ±89°

	bool  m_mouse_captured = false;
	float m_last_x         = 0.f;
	float m_last_y         = 0.f;
	bool  m_first_mouse    = true;

	// Cache previous right-button state to detect press edges
	int m_prev_rmb = GLFW_RELEASE;

	glm::vec3 forward() const;
	glm::vec3 right() const;
};