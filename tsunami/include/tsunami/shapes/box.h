#pragma once

#include "tsunami/shapes/shape.h"

class Box : public Shape {
  public:
	Box(Transform transform, Material* material);
	~Box() = default;
	GPUShape pack(int matIndex) const override;
};