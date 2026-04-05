#include "tsunami/scene/scene.h"
#include "tsunami/shapes/mesh.h"
#include <iostream>

GPUScene Scene::pack() const {
    GPUScene packed;
    packed.camera = m_camera.pack();
    for (const auto& light : m_lights)
        packed.lights.push_back(light->pack());
    for (int i = 0; i < (int)m_shapes.size(); i++) {
        Shape* shape = m_shapes[i].get();
        packed.materials.push_back(shape->m_material->pack());
        packed.shapes.push_back(shape->pack(i));
    }
    return packed;
}

void Scene::load_gltf(const std::string& path) {
    auto meshes = Mesh::load_gltf(path);
    if (meshes.empty()) {
        std::cerr << "[Scene::load_gltf] No meshes loaded from: " << path << "\n";
        return;
    }
    for (auto& mesh : meshes)
        m_meshes.push_back(std::move(mesh));
    std::cout << "[Scene] Total meshes after load: " << m_meshes.size() << "\n";
}