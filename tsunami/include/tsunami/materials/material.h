#pragma once
#include <cstdint>
#include <glm/glm.hpp>

struct alignas(16) GPUMaterial {
	// Base layer
	glm::vec4 base_color;                    // rgb: color, a: weight (0-1)
	float     base_metalness;                // 0=dielectric, 1=metal
	float     base_diffuse_roughness;        // 0=smooth diffuse, 1=rough diffuse
	float     _pad0[2];

	// Specular layer
	glm::vec4 specular_color;             // rgb: color, a: weight (0-1)
	float     specular_roughness;         // 0=mirror, 1=fully rough
	float     specular_ior;               // index of refraction (1.5 default)
	float     specular_anisotropy;        // 0=isotropic, 1=fully anisotropic
	float     _pad1;

	// Transmission
	float     transmission_weight;        // 0=opaque, 1=fully transmissive
	float     _pad2[3];
	glm::vec4 transmission_color;        // rgb: tint, a: unused

	// Emission
	glm::vec4 emission_color;            // rgb: color, a: weight
	float     emission_luminance;        // cd/m^2 equivalent scale
	float     _pad3[3];

	// Coat
	float     coat_weight;        // 0=no coat, 1=full coat
	float     coat_roughness;
	float     coat_ior;
	float     _pad4;
	glm::vec4 coat_color;        // rgb: tint, a: unused

	// Fuzz/sheen
	float     fuzz_weight;
	float     fuzz_roughness;
	float     _pad5[2];
	glm::vec4 fuzz_color;        // rgb: tint, a: unused

	// Thin film
	float thin_film_weight;
	float thin_film_ior;
	float thin_film_thickness;        // nanometres
	float _pad6;

	// Texture indices into the bindless material_textures[] array.
	// 0xFFFFFFFF = no texture; fall back to the flat colour field above.
	uint32_t albedo_tex_index;
	uint32_t normal_tex_index;
	uint32_t roughness_tex_index;
	uint32_t emissive_tex_index;
};

// Single material class — set only the fields you need; the rest default to
// OpenPBR 1.0 spec values.  No subclasses needed.
class Material {
  public:
	Material() {
		set_defaults();
	}
	virtual ~Material() = default;
	GPUMaterial pack() const;

	// Convenience fluent setters
	Material& albedo(glm::vec3 c) {
		m_gpu.base_color = glm::vec4(c, 1.f);
		return *this;
	}
	Material& metalness(float m) {
		m_gpu.base_metalness = m;
		return *this;
	}
	Material& roughness(float r) {
		m_gpu.specular_roughness = r;
		return *this;
	}
	Material& ior(float i) {
		m_gpu.specular_ior = i;
		return *this;
	}
	Material& emission(glm::vec3 e, float lum = 1.f) {
		m_gpu.emission_color     = glm::vec4(e, 1.f);
		m_gpu.emission_luminance = lum;
		return *this;
	}
	Material& coat(float w, float r = 0.f, float i = 1.6f) {
		m_gpu.coat_weight    = w;
		m_gpu.coat_roughness = r;
		m_gpu.coat_ior       = i;
		return *this;
	}
	Material& transmission(float w) {
		m_gpu.transmission_weight = w;
		return *this;
	}
	Material& fuzz(float w, float r = 0.5f) {
		m_gpu.fuzz_weight    = w;
		m_gpu.fuzz_roughness = r;
		return *this;
	}

	// Direct texture-index setters (used by scene loader)
	Material& albedo_tex(uint32_t idx) {
		m_gpu.albedo_tex_index = idx;
		return *this;
	}
	Material& normal_tex(uint32_t idx) {
		m_gpu.normal_tex_index = idx;
		return *this;
	}
	Material& roughness_tex(uint32_t idx) {
		m_gpu.roughness_tex_index = idx;
		return *this;
	}
	Material& emissive_tex(uint32_t idx) {
		m_gpu.emissive_tex_index = idx;
		return *this;
	}

	GPUMaterial m_gpu{};

  private:
	void set_defaults() {
		// OpenPBR 1.0 spec defaults
		m_gpu.base_color             = glm::vec4(0.8f, 0.8f, 0.8f, 1.f);
		m_gpu.base_metalness         = 0.f;
		m_gpu.base_diffuse_roughness = 0.f;
		m_gpu.specular_color         = glm::vec4(1.f, 1.f, 1.f, 1.f);
		m_gpu.specular_roughness     = 0.3f;
		m_gpu.specular_ior           = 1.5f;
		m_gpu.specular_anisotropy    = 0.f;
		m_gpu.transmission_weight    = 0.f;
		m_gpu.transmission_color     = glm::vec4(1.f);
		m_gpu.emission_color         = glm::vec4(0.f, 0.f, 0.f, 0.f);
		m_gpu.emission_luminance     = 0.f;
		m_gpu.coat_weight            = 0.f;
		m_gpu.coat_roughness         = 0.f;
		m_gpu.coat_ior               = 1.6f;
		m_gpu.coat_color             = glm::vec4(1.f);
		m_gpu.fuzz_weight            = 0.f;
		m_gpu.fuzz_roughness         = 0.5f;
		m_gpu.fuzz_color             = glm::vec4(1.f);
		m_gpu.thin_film_weight       = 0.f;
		m_gpu.thin_film_ior          = 1.5f;
		m_gpu.thin_film_thickness    = 0.f;
		m_gpu._pad6                  = 0.f;
		// No textures by default
		m_gpu.albedo_tex_index    = 0xFFFFFFFFu;
		m_gpu.normal_tex_index    = 0xFFFFFFFFu;
		m_gpu.roughness_tex_index = 0xFFFFFFFFu;
		m_gpu.emissive_tex_index  = 0xFFFFFFFFu;
	}
};