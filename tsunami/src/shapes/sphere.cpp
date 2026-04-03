#include "tsunami/shapes/sphere.h"

Sphere::Sphere(Transform transform, std::shared_ptr<Material> material) : Shape(transform, material) {
}

GPUShape Sphere::pack(int matIndex) const {
	return GPUShape{m_transform.pack(), glm::inverse(m_transform.pack()),
	                2,        // type: sphere
	                matIndex};
}