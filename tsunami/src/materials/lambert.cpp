#include "tsunami/materials/lambert.h"

Lambert::Lambert(glm::vec3 albedo, glm::vec3 emission, float emissionIntensity) {
	m_albedo            = albedo;
	m_emission          = emission;
	m_emissionIntensity = emissionIntensity;
}

GPUMaterial Lambert::pack() const {
	return GPUMaterial{glm::vec4(m_albedo, 0.0f),        // a: 0 = lambert type flag
	                   glm::vec4(m_emission, m_emissionIntensity)};
}