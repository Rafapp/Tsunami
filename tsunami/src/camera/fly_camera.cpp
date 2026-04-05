#include "tsunami/camera/fly_camera.h"
#include "tsunami/camera/camera.h"

#include <glm/trigonometric.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>

FlyCamera::FlyCamera(glm::vec3 position, glm::vec3 target,
                     float fov_deg, float speed_)
    : speed(speed_), fov(fov_deg), m_position(position) {
    // Derive initial yaw/pitch from the look direction
    glm::vec3 dir = glm::normalize(target - position);
    m_pitch = std::asin(dir.y);
    m_yaw   = std::atan2(dir.x, dir.z);
}

glm::vec3 FlyCamera::forward() const {
    return {
         std::sin(m_yaw) * std::cos(m_pitch),
         std::sin(m_pitch),
         std::cos(m_yaw) * std::cos(m_pitch)
    };
}

glm::vec3 FlyCamera::right() const {
    return glm::normalize(glm::cross(forward(), glm::vec3{0.f, 1.f, 0.f}));
}

bool FlyCamera::update(GLFWwindow* window, float dt) {
    bool moved = false;

    // ---- right-mouse toggle capture ----
    int rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
    if (rmb == GLFW_PRESS && m_prev_rmb == GLFW_RELEASE) {
        m_mouse_captured = !m_mouse_captured;
        if (m_mouse_captured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            m_first_mouse = true;   // avoid jump on first captured frame
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
            m_last_x = fx;
            m_last_y = fy;
            m_first_mouse = false;
        }

        float dx = (fx - m_last_x) * sensitivity;
        float dy = (fy - m_last_y) * sensitivity;
        m_last_x = fx;
        m_last_y = fy;

        if (dx != 0.f || dy != 0.f) {
            m_yaw   -= dx;
            m_pitch -= dy;
            // Clamp pitch to ±89°
            constexpr float kLimit = glm::radians(89.f);
            m_pitch = std::clamp(m_pitch, -kLimit, kLimit);
            moved = true;
        }
    }

    // ---- keyboard movement ----
    float spd = speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
        spd *= 4.f;

    glm::vec3 fwd = forward();
    glm::vec3 rgt = right();
    glm::vec3 up  = {0.f, 1.f, 0.f};

    glm::vec3 move{0.f};
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)     move += fwd;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)     move -= fwd;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)     move -= rgt;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)     move += rgt;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)       move += up;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) move -= up;

    if (glm::length(move) > 0.001f) {
        m_position += glm::normalize(move) * (spd * dt);
        moved = true;
    }

    // ---- speed adjust (Q / E) ----
    static int prev_q = GLFW_RELEASE, prev_e = GLFW_RELEASE;
    int cur_q = glfwGetKey(window, GLFW_KEY_Q);
    int cur_e = glfwGetKey(window, GLFW_KEY_E);
    if (cur_q == GLFW_PRESS && prev_q == GLFW_RELEASE) speed = std::max(1.f, speed * 0.5f);
    if (cur_e == GLFW_PRESS && prev_e == GLFW_RELEASE) speed *= 2.f;
    prev_q = cur_q;
    prev_e = cur_e;

    return moved;
}

GPUCamera FlyCamera::pack() const {
    glm::vec3 target = m_position + forward();
    return GPUCamera{
        glm::vec4(m_position, 0.f),
        glm::vec4(target,     0.f),
        glm::vec4(0.f, 1.f, 0.f, 0.f),
        glm::vec4(fov, 0.1f, 10000.f, 0.f)
    };
}