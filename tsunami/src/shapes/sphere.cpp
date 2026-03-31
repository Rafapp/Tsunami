#include "tsunami/shapes/sphere.h"

Sphere::Sphere(Transform transform, Material* material, float radius)
    : Shape(transform, material), m_radius(radius) {}

GPUShape Sphere::pack(int matIndex) const {
    return GPUShape {
        m_transform.pack(),
        glm::vec4(m_radius, 0.0f, 0.0f, 0.0f),
        glm::vec4(0.0f),
        0,
        matIndex,
        {0.0f, 0.0f}
    };
}