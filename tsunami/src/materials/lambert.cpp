#include "tsunami/materials/lambert.h"

Lambert::Lambert(glm::vec3 albedo, glm::vec3 emission, float emissionIntensity) {
    m_albedo            = albedo;
    m_emission          = emission;
    m_emissionIntensity = emissionIntensity;
}

GPUMaterial Lambert::pack() const {
    GPUMaterial g{};
    g.albedo_type        = glm::vec4(m_albedo, 0.0f);
    g.emission_intensity = glm::vec4(m_emission, m_emissionIntensity);
    g.pbr                = glm::vec4(1.0f, 0.0f, 1.5f, 0.0f); // rough=1, metallic=0
    g.albedo_tex_index       = 0xFFFFFFFF;
    g.normal_tex_index       = 0xFFFFFFFF;
    g.roughness_tex_index    = 0xFFFFFFFF;
    g.emissive_tex_index     = 0xFFFFFFFF;
    return g;
}