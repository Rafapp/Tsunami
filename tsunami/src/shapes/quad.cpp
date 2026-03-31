#include "tsunami/shapes/quad.h"

Quad::Quad(Transform transform, Material* material, float width, float height) :
    Shape(transform, material), m_width(width), m_height(height) {
}

GPUShape Quad::pack(int matIndex) const {
	return GPUShape{m_transform.pack(),
	                glm::vec4(m_width, m_height, 0.0f, 0.0f),        // params0: dimensions
	                glm::vec4(0.0f),
	                1,        // type: quad
	                matIndex,
	                {0.0f, 0.0f}};
}