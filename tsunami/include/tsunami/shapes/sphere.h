#pragma once

#include "tsunami/shapes/shape.h"

class Sphere : public Shape {
  public:
	Sphere(Transform transform, Material* material, float radius);
	~Sphere() = default;
	GPUShape pack(int matIndex) const override;

  private:
	float m_radius;
};