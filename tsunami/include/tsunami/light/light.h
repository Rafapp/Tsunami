// Purpose: Base lighting interface shared by scene light implementations.
#pragma once

#include <glm/glm.hpp>

struct alignas(16) GPULight {
	glm::vec4 position;               // xyz: position, w: type (1: point light)
	glm::vec4 color_intensity;        // rgb: color, a: intensity
};

class Light {
  public:
	Light()          = default;
	virtual ~Light() = default;

	virtual GPULight pack() const = 0;
	glm::vec3        m_position;
	glm::vec3        m_color;
	float            m_intensity;
};