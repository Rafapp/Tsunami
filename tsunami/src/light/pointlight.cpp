#include "tsunami/light/pointlight.h"

PointLight::PointLight(const glm::vec3& position, const glm::vec3& color, float intensity) {
	m_position  = position;
	m_color     = color;
	m_intensity = intensity;
}

GPULight PointLight::pack() const {
	return GPULight{
	    glm::vec4(m_position, 1.0f),           // w: 1 for point light
	    glm::vec4(m_color, m_intensity)        // a: intensity
	};
}
