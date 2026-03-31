#pragma once

#include "tsunami/scene/transform.h"

struct alignas(16) GPUCamera {
	glm::vec4 position;
	glm::vec4 target;
	glm::vec4 up;
	glm::vec4 fov_near_far;        // x: fov, y: near clip, z: far clip
};

class Camera {
  public:
	Camera();
	~Camera() = default;

	GPUCamera pack() const;
	void      setTransform(const Transform& transform);
	Transform getTransform() const;

  private:
	glm::vec3 m_position;
	glm::vec3 m_target;
	glm::vec3 m_up;
	float     m_fov;
	float     m_nearClip;
	float     m_farClip;
};