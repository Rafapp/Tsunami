// Purpose: Procedural quad primitive interface for generating planar meshes.
#pragma once

#include "tsunami/shapes/shape.h"

class Quad : public Shape {
  public:
	Quad(Transform transform, std::shared_ptr<Material> material);
	~Quad() = default;
	GPUShape pack(int matIndex) const override;
};