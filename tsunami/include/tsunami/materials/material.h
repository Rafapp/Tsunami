#pragma once
#include <glm/glm.hpp>

struct alignas(16) GPUMaterial {
	glm::vec4 albedo_type;               // rgb: color,     a: type (0=lambert, 1=pbr)
	glm::vec4 emission_intensity;        // rgb: emission,  a: intensity
	glm::vec4 pbr;                       // r: roughness,   g: metallic, b: ior, a: unused
	// Texture indices — 0xFFFFFFFF means no texture, use flat color
	uint32_t albedo_tex_index;
	uint32_t normal_tex_index;
	uint32_t roughness_tex_index;
	uint32_t emissive_tex_index;
};

class Material {
  public:
	Material()                       = default;
	virtual ~Material()              = default;
	virtual GPUMaterial pack() const = 0;
	glm::vec3           m_albedo;
	glm::vec3           m_emission;
	float               m_emissionIntensity;
};