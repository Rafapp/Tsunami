#pragma once

#include "tsunami/shapes/shape.h"

class Cube : public Shape {
public:
    Cube(Transform transform, Material* material, float width, float height, float depth);
    ~Cube() = default;
    GPUShape pack(int matIndex) const override;

private:
    float m_width;
    float m_height;
    float m_depth;
};