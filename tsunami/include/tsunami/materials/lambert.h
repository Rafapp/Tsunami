#pragma once

#include "tsunami/materials/material.h"

class Lambert : public Material {
  public:
	Lambert(glm::vec3 albedo);
	~Lambert() = default;
	GPUMaterial pack() const override;
};