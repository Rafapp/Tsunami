#pragma once

#include <glm/glm.hpp>

struct alignas(16) GPUMaterial {
	glm::vec4 albedo_type;               // rgb: color, a: type (0: lambert)
	glm::vec4 emission_intensity;        // rgb: emission color, a: intensity
};

class Material {
  public:
	Material()          = default;
	virtual ~Material() = default;

	virtual GPUMaterial pack() const = 0;
	glm::vec3           m_albedo;
	glm::vec3           m_emission;
	float               m_emissionIntensity;
};