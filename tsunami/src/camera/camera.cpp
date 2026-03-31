#include "tsunami/camera/camera.h"

Camera::Camera()
    : m_position(0.0f, 0.0f, 3.0f),
      m_target(0.0f, 0.0f, 0.0f),
      m_up(0.0f, 1.0f, 0.0f),
      m_fov(45.0f),
      m_nearClip(0.1f),
      m_farClip(100.0f) {}

GPUCamera Camera::pack() const {
    return GPUCamera {
        glm::vec4(m_position, 0.0f),
        glm::vec4(m_target,   0.0f),
        glm::vec4(m_up,       0.0f),
        glm::vec4(m_fov, m_nearClip, m_farClip, 0.0f)
    };
}

void Camera::setTransform(const Transform& transform) {
    m_position = transform.m_position;
    m_target   = m_position + glm::vec3(
        sin(glm::radians(transform.m_rotation.y)) * cos(glm::radians(transform.m_rotation.x)),
        sin(glm::radians(transform.m_rotation.x)),
        cos(glm::radians(transform.m_rotation.y)) * cos(glm::radians(transform.m_rotation.x))
    );
}

Transform Camera::getTransform() const {
    return Transform(m_position, glm::vec3(0.0f), glm::vec3(1.0f));
}