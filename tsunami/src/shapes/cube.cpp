#include "tsunami/shapes/cube.h"

Cube::Cube(Transform transform, Material* material, float width, float height, float depth) :
    Shape(transform, material), m_width(width), m_height(height), m_depth(depth) {
}

GPUShape Cube::pack(int matIndex) const {
	return GPUShape{m_transform.pack(),
	                glm::vec4(m_width, m_height, m_depth, 0.0f),        // params0: dimensions
	                glm::vec4(0.0f),
	                2,        // type: cube
	                matIndex,
	                {0.0f, 0.0f}};
}