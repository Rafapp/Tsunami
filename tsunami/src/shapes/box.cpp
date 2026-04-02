#include "tsunami/shapes/box.h"

Box::Box(Transform transform, Material* material) : 
	Shape(transform, material) {}

GPUShape Box::pack(int matIndex) const {
	return GPUShape{
	    m_transform.pack(),
		glm::inverse(m_transform.pack()),
	    1, // type: box 
	    matIndex
	};
}