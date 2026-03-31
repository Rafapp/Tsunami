#pragma once

#include "tsunami/materials/material.h"
#include "tsunami/scene/transform.h"
#include <glm/glm.hpp>

struct alignas(16) GPUShape {
	glm::mat4 transform;
	glm::vec4 params0;        // repurposed per shape type
	glm::vec4 params1;
	int       type;            // 0: sphere, 1: quad, 2: cube
	int       matIndex;        // index into materials buffer
	float     _pad[2];         // keep 16 byte alignment
};

class Shape {
  public:
	Shape(Transform transform, Material* material) :
	    m_transform(transform), m_material(material) {};
	virtual ~Shape()                          = default;
	virtual GPUShape pack(int matIndex) const = 0;

	Transform m_transform;
	Material* m_material;
};