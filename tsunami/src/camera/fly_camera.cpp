#include "tsunami/camera/fly_camera.h"
#include "tsunami/camera/camera.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

FlyCamera::FlyCamera(glm::vec3 position, glm::vec3 target, float fov_deg, float speed,
                     float sensitivity, float near_clip, float far_clip, glm::vec3 up) :
    m_speed(speed),
    m_sensitivity(sensitivity),
    m_fov(fov_deg),
    m_near_clip(near_clip),
    m_far_clip(far_clip),
    m_position(position) {
	const float up_len = glm::length(up);
	m_up               = (up_len > 1e-6f) ? (up / up_len) : glm::vec3{0.f, 1.f, 0.f};

	// Derive initial yaw/pitch from the look direction
	glm::vec3 dir = target - position;
	if (glm::length(dir) < 1e-6f) {
		dir = glm::vec3{0.f, 0.f, -1.f};
	} else {
		dir = glm::normalize(dir);
	}
	m_pitch = std::asin(dir.y);
	m_yaw   = std::atan2(dir.x, dir.z);
}

glm::vec3 FlyCamera::forward() const {
	return {std::sin(m_yaw) * std::cos(m_pitch), std::sin(m_pitch),
	        std::cos(m_yaw) * std::cos(m_pitch)};
}

glm::vec3 FlyCamera::right() const {
	const glm::vec3 side = glm::cross(forward(), m_up);
	if (glm::length(side) > 1e-6f) {
		return glm::normalize(side);
	}
	return glm::normalize(glm::cross(forward(), glm::vec3{0.f, 0.f, 1.f}));
}

void FlyCamera::applyControllerInput(float move_x, float move_y,
                                     float look_x, float look_y, float dt) {
    constexpr float kMoveSpeed = 3.0f;
    constexpr float kLookSpeed = 1.5f;

    glm::vec3 move{0.f};
    move += forward() * move_y;   
    move += right()   * move_x;   

    if (glm::length(move) > 0.001f)
        m_position += glm::normalize(move) * (kMoveSpeed * m_speed * dt);

    // Stick look
    if (look_x != 0.f || look_y != 0.f) {
        m_yaw   -= look_x * kLookSpeed * dt;
        m_pitch -= look_y * kLookSpeed * dt;
        constexpr float kLimit = glm::radians(89.f);
        m_pitch = std::clamp(m_pitch, -kLimit, kLimit);
    }
}

bool FlyCamera::update(GLFWwindow* window, float dt) {
	bool moved = false;

	// ---- right-mouse toggle capture ----
	int rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (rmb == GLFW_PRESS && m_prev_rmb == GLFW_RELEASE) {
		m_mouse_captured = !m_mouse_captured;
		if (m_mouse_captured) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			m_first_mouse = true;        // avoid jump on first captured frame
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
	m_prev_rmb = rmb;

	// ESC releases the mouse without closing the window
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && m_mouse_captured) {
		m_mouse_captured = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// ---- mouse look (only when captured) ----
	if (m_mouse_captured) {
		double mx, my;
		glfwGetCursorPos(window, &mx, &my);
		float fx = static_cast<float>(mx);
		float fy = static_cast<float>(my);

		if (m_first_mouse) {
			m_last_x      = fx;
			m_last_y      = fy;
			m_first_mouse = false;
		}

		float dx = (fx - m_last_x) * m_sensitivity;
		float dy = (fy - m_last_y) * m_sensitivity;
		m_last_x = fx;
		m_last_y = fy;

		if (dx != 0.f || dy != 0.f) {
			m_yaw -= dx;
			m_pitch -= dy;
			// Clamp pitch to ±89°
			constexpr float kLimit = glm::radians(89.f);
			m_pitch                = std::clamp(m_pitch, -kLimit, kLimit);
			moved                  = true;
		}
	}

	// ---- keyboard movement ----
	float spd = m_speed;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
	    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
		spd *= 4.f;

	glm::vec3 fwd = forward();
	glm::vec3 rgt = right();
	glm::vec3 up  = m_up;

	glm::vec3 move{0.f};
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		move += fwd;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		move -= fwd;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		move -= rgt;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		move += rgt;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		move += up;
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		move -= up;

	if (glm::length(move) > 0.001f) {
		m_position += glm::normalize(move) * (spd * dt);
		moved = true;
	}

	// ---- speed adjust (Q / E) ----
	static int prev_q = GLFW_RELEASE, prev_e = GLFW_RELEASE;
	int        cur_q = glfwGetKey(window, GLFW_KEY_Q);
	int        cur_e = glfwGetKey(window, GLFW_KEY_E);
	if (cur_q == GLFW_PRESS && prev_q == GLFW_RELEASE)
		m_speed = std::max(0.01f, m_speed * 0.5f);
	if (cur_e == GLFW_PRESS && prev_e == GLFW_RELEASE)
		m_speed *= 2.f;
	prev_q = cur_q;
	prev_e = cur_e;

	return moved;
}

GPUCamera FlyCamera::pack() const {
	glm::vec3 target = m_position + forward();
	return GPUCamera{glm::vec4(m_position, 0.f), glm::vec4(target, 0.f), glm::vec4(m_up, 0.f),
	                 glm::vec4(m_fov, m_near_clip, m_far_clip, 0.f)};
}
