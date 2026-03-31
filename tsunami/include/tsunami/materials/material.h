#pragma once

#include <glm/glm.hpp>

struct alignas(16) GPUMaterial {
	glm::vec4 albedo;        // rgb: color, a: type (0: lambert)
};

class Material {
  public:
	Material()          = default;
	virtual ~Material() = default;

	virtual GPUMaterial pack() const = 0;
	glm::vec3           m_albedo;
};