#pragma once

#include "tsunami/light/light.h"

class PointLight : public Light {
    public:
        PointLight(const glm::vec3& position, const glm::vec3& color, float intensity);
        ~PointLight() = default;
        GPULight pack() const override;
};