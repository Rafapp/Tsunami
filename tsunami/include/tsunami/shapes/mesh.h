// Purpose: Mesh data interface used by scene loading and rendering systems.
#pragma once
#include "tsunami/materials/material.h"
#include "tsunami/scene/transform.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

struct alignas(16) GPUVertex {
	glm::vec3 position;
	float     _pad0;
	glm::vec3 normal;
	float     _pad1;
	glm::vec4 tangent;        // xyz = tangent, w = handedness
	glm::vec2 uv;
	glm::vec2 _pad2;
};

struct alignas(16) GPUMesh {
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	uint32_t  blasHandle_lo;
	uint32_t  blasHandle_hi;
	int       matIndex;
	int       vertexOffset;
	int       indexOffset;
	int       indexCount;
	float     _pad[3];
};

class Mesh {
  public:
	// Load a single .obj (collapses all sub-meshes as before)
	Mesh(const std::string& path, Transform transform, std::shared_ptr<Material> material);

	// Construct from already-built buffers (used by load_gltf)
	Mesh(std::vector<GPUVertex> vertices, std::vector<uint32_t> indices, Transform transform,
	     std::shared_ptr<Material> material, std::string name = {});

	~Mesh() = default;

	// Load a glTF file via assimp — returns one Mesh per primitive,
	// each with its own material resolved from the glTF.
	// Texture paths are stored in the material but not uploaded yet (stb stub).
	static std::vector<std::unique_ptr<Mesh>> load_gltf(const std::string& path);

	GPUMesh pack(int matIndex, int vertexOffset, int indexOffset) const;

	std::vector<GPUVertex>    gpuVertices;
	std::vector<uint32_t>     gpuIndices;
	Transform                 m_transform;
	std::shared_ptr<Material> m_material;
	std::string               m_name;
	glm::vec3                 m_local_bounds_min = glm::vec3(0.0f);
	glm::vec3                 m_local_bounds_max = glm::vec3(0.0f);

  private:
	bool load_obj(const std::string& path);
	void compute_local_bounds();
};
