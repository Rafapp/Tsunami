#pragma once

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

#include "tsunami/materials/material.h"
#include "tsunami/scene/transform.h"

struct alignas(16) GPUVertex {
	glm::vec3 position;
	float     _pad0;
	glm::vec3 normal;
	float     _pad1;
};

struct alignas(16) GPUMesh {
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	uint64_t  blasHandle;
	int       matIndex;
	int       vertexOffset;
	int       indexOffset;
	int       indexCount;
	float     _pad[3];
};

class Mesh {
  public:
	Mesh(const std::string& path, Transform transform, std::shared_ptr<Material> material);
	~Mesh() = default;

	GPUMesh pack(int matIndex, int vertexOffset, int indexOffset) const;

	std::vector<glm::vec3>  vertices;
	std::vector<glm::uvec3> triangles;
	std::vector<glm::vec3>  normals;
	std::vector<GPUVertex>  gpuVertices;
	std::vector<uint32_t>   gpuIndices;

	Transform                 m_transform;
	std::shared_ptr<Material> m_material;

  private:
	bool load(const std::string& path);
};