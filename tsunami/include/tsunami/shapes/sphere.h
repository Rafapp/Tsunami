#pragma once

#include "tsunami/shapes/shape.h"

class Sphere : public Shape {
  public:
	Sphere(Transform transform, std::shared_ptr<Material> material);
	~Sphere() = default;
	GPUShape pack(int matIndex) const override;
};