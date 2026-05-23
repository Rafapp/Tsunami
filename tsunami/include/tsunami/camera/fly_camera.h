#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Forward-declared so callers don't need to include the full camera header
struct GPUCamera;

// ---------------------------------------------------------------------------
//  FlyCamera - first-person fly camera driven by GLFW input
//
//  Controls (while right-mouse is held to capture the cursor):
//    Right-click       - toggle mouse capture on/off
//    ESC               - release mouse capture
//    WASD              - move forward / backward / strafe
//    Space / Ctrl      - move up / down
//    Shift (held)      - 4x speed multiplier
//    Q / E             - halve / double the base speed
// ---------------------------------------------------------------------------
class FlyCamera {
  public:
	// Construct from a camera position and look-at target.
	// Defaults: position (0, 0.5, 0), target (0, 0, 0), speed 0.5, sensitivity 0.0005,
	// near 0.1, far 10000, up +Y.
	FlyCamera(glm::vec3 position = glm::vec3(0.f, 0.5f, 0.f),
	          glm::vec3 target = glm::vec3(0.f, 0.f, 0.f), float fov_deg = 60.f, float speed = 0.5f,
	          float sensitivity = 0.0005f, float near_clip = 0.1f, float far_clip = 10000.f,
	          glm::vec3 up = glm::vec3(0.f, 1.f, 0.f));

	// Call once per frame. dt is seconds elapsed since last frame.
	// Returns true if the camera moved (so the caller can reset accumulation).
	bool update(GLFWwindow* window, float dt);

	// Pack current state into the GPU-side camera buffer struct
	GPUCamera pack() const;

	bool isMouseCaptured() const {
		return m_mouse_captured;
	}

	void applyControllerInput(float move_x, float move_y, float look_x, float look_y, float dt);

	// ---- tweakable parameters ----
	float     m_speed       = 0.5f;           // units / second
	float     m_sensitivity = 0.0005f;        // radians / pixel
	float     m_fov         = 60.f;           // vertical field-of-view in degrees
	float     m_near_clip   = 0.1f;
	float     m_far_clip    = 10000.f;
	glm::vec3 m_position{0.f, 0.5f, 0.f};
	float     m_yaw   = 0.f;        // radians, around world-Y
	float     m_pitch = 0.f;        // radians, clamped to +/-89 deg

  private:
	glm::vec3 m_up{0.f, 1.f, 0.f};

	bool  m_mouse_captured = false;
	float m_last_x         = 0.f;
	float m_last_y         = 0.f;
	bool  m_first_mouse    = true;

	// Cache previous right-button state to detect press edges
	int m_prev_rmb = GLFW_RELEASE;

	glm::vec3 forward() const;
	glm::vec3 right() const;
};
