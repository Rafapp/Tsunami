#pragma once

#include "tsunami/shapes/shape.h"

class Quad : public Shape {
  public:
	Quad(Transform transform, std::shared_ptr<Material> material);
	~Quad() = default;
	GPUShape pack(int matIndex) const override;
};