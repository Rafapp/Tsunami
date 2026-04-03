#pragma once
#include "tsunami/camera/camera.h"
#include "tsunami/light/light.h"

#include "tsunami/materials/lambert.h"
#include "tsunami/materials/material.h"

#include "tsunami/shapes/shape.h"
#include "tsunami/shapes/quad.h"
#include "tsunami/shapes/box.h"
#include "tsunami/shapes/sphere.h"
#include "tsunami/shapes/mesh.h"

#include <vector>

struct GPUScene {
	GPUCamera                camera;
	std::vector<GPULight>    lights;
	std::vector<GPUMaterial> materials;
	std::vector<GPUShape>    shapes;
	std::vector<GPUMesh>  meshes;
};

class Scene {
  public:
	Scene()  = default;
	~Scene() = default;
	GPUScene pack() const;

	Camera                              m_camera;
	std::vector<std::unique_ptr<Light>> m_lights;
	std::vector<std::unique_ptr<Shape>> m_shapes;
	std::vector<std::unique_ptr<Mesh>>  m_meshes;
};