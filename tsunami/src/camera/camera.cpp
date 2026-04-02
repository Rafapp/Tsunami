#include "tsunami/camera/camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up,
               float fov, float nearClip, float farClip) :
    m_position(position),
    m_target(target),
    m_up(up),
    m_fov(fov),
    m_nearClip(nearClip),
    m_farClip(farClip) {}

GPUCamera Camera::pack() const {
	return GPUCamera{
	    glm::vec4(m_position, 0.0f),
	    glm::vec4(m_target,   0.0f),
	    glm::vec4(m_up,       0.0f),
	    glm::vec4(m_fov, m_nearClip, m_farClip, 0.0f)
	};
}