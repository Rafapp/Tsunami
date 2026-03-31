#include "tsunami/materials/lambert.h"

Lambert::Lambert(glm::vec3 albedo) {
	m_albedo = albedo;
}

GPUMaterial Lambert::pack() const {
	return GPUMaterial{
	    glm::vec4(m_albedo, 0.0f)        // a: 0 = lambert type flag
	};
}