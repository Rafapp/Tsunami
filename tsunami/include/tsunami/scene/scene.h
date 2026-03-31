#pragma once
#include "tsunami/camera/camera.h"
#include "tsunami/light/light.h"
#include "tsunami/materials/material.h"
#include "tsunami/shapes/shape.h"
#include <vector>

struct GPUScene {
	GPUCamera                camera;
	std::vector<GPULight>    lights;
	std::vector<GPUMaterial> materials;
	std::vector<GPUShape>    shapes;
};

class Scene {
  public:
	Scene(Camera camera, std::vector<Light*> lights, std::vector<Shape*> shapes);
	~Scene() = default;
	GPUScene pack() const;

	Camera              m_camera;
	std::vector<Light*> m_lights;
	std::vector<Shape*> m_shapes;
};