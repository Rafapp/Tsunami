#include "tsunami/shapes/quad.h"

Quad::Quad(Transform transform, Material* material) :
    Shape(transform, material) {}

GPUShape Quad::pack(int matIndex) const {
    return GPUShape{
        m_transform.pack(),
        glm::inverse(m_transform.pack()),
        0,
        matIndex
    };
}