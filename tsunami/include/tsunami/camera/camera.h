#pragma once
#include <glm/glm.hpp>

struct alignas(16) GPUCamera {
	glm::vec4 position;
	glm::vec4 target;
	glm::vec4 up;
	glm::vec4 fov_near_far; // x: fov, y: near clip, z: far clip
};

class Camera {
  public:
	Camera() = default;
	Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up,
	       float fov, float nearClip, float farClip);
	~Camera() = default;

	GPUCamera pack() const;

	glm::vec3 m_position = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 m_target   = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 m_up       = glm::vec3(0.0f, 1.0f, 0.0f);
	float     m_fov      = 45.0f;
	float     m_nearClip = 0.1f;
	float     m_farClip  = 100.0f;
};