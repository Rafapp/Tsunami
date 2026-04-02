#pragma once

#include "tsunami/materials/material.h"

class Lambert : public Material {
  public:
	Lambert(glm::vec3 albedo, glm::vec3 emission = glm::vec3(0.0f), float emissionIntensity = 0.0f);
	~Lambert() = default;
	GPUMaterial pack() const override;
};