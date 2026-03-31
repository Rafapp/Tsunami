#pragma once

#include "tsunami/shapes/shape.h"

class Quad : public Shape {
  public:
	Quad(Transform transform, Material* material, float width, float height);
	~Quad() = default;
	GPUShape pack(int matIndex) const override;

  private:
	float m_width;
	float m_height;
};