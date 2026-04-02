#include "tsunami/scene/scene.h"

GPUScene Scene::pack() const {
	GPUScene packed;

	packed.camera = m_camera.pack();

	for (const auto& light : m_lights)
		packed.lights.push_back(light->pack());

	for (int i = 0; i < m_shapes.size(); i++) {
		Shape* shape = m_shapes[i].get();
		packed.materials.push_back(shape->m_material->pack());
		packed.shapes.push_back(shape->pack(i));        // i is the material index
	}

	return packed;
}