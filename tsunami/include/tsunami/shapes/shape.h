// Purpose: Common base interface for procedural shape generators.
#pragma once

#include <glm/glm.hpp>
#include <memory>

#include "tsunami/materials/material.h"
#include "tsunami/scene/transform.h"

struct alignas(16) GPUShape {
	glm::mat4 transform;
	glm::mat4 inverseTransform;        // pre-computed on CPU
	int       type;                    // 0: quad, 1: box, 2: sphere
	int       matIndex;
	float     _pad[2];
};

class Shape {
  public:
	Shape(Transform transform, std::shared_ptr<Material> material) :
	    m_transform(transform), m_material(material) {
	}
	virtual ~Shape()                          = default;
	virtual GPUShape pack(int matIndex) const = 0;

	Transform                 m_transform;
	std::shared_ptr<Material> m_material;
};