#include <iostream>

#include "tsunami/shapes/mesh.h"

Mesh::Mesh(const std::string& path, Transform transform, std::shared_ptr<Material> material) {
	m_transform = transform;
	m_material  = material;
	if (!load(path))
		std::cerr << "[Mesh] Failed to load: " << path << "\n";
}

bool Mesh::load(const std::string& path) {
	Assimp::Importer importer;

	const aiScene* scene =
	    importer.ReadFile(path,
	                      aiProcess_Triangulate |                 // guarantee tris only
	                          aiProcess_GenSmoothNormals |        // generate normals if missing
	                          aiProcess_JoinIdenticalVertices);

	if (!scene || !scene->HasMeshes()) {
		std::cerr << "[Mesh] Assimp error: " << importer.GetErrorString() << "\n";
		return false;
	}

	// Collapse sub-meshes to a single mesh
	// TODO: Sub-mesh and per-submesh material support
	for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
		const aiMesh* mesh = scene->mMeshes[m];

		const uint32_t vertexBase = static_cast<uint32_t>(vertices.size());

		// Vertices / normals
		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			const auto& v = mesh->mVertices[i];
			const auto& n = mesh->mNormals[i];

			vertices.push_back({v.x, v.y, v.z});
			normals.push_back({n.x, n.y, n.z});

			GPUVertex gv{};
			gv.position = {v.x, v.y, v.z};
			gv.normal   = {n.x, n.y, n.z};
			gpuVertices.push_back(gv);
		}

		// Triangles
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i];

			uint32_t i0 = vertexBase + face.mIndices[0];
			uint32_t i1 = vertexBase + face.mIndices[1];
			uint32_t i2 = vertexBase + face.mIndices[2];

			triangles.push_back({i0, i1, i2});

			gpuIndices.push_back(i0);
			gpuIndices.push_back(i1);
			gpuIndices.push_back(i2);
		}
	}

	return true;
}

GPUMesh Mesh::pack(int matIndex, int vertexOffset, int indexOffset) const {
	GPUMesh g{};
	g.transform        = m_transform.m_transform;
	g.inverseTransform = m_transform.m_inverseTransform;
	g.blasHandle       = 0;        // filled in after BLAS build
	g.matIndex         = matIndex;
	g.vertexOffset     = vertexOffset;
	g.indexOffset      = indexOffset;
	g.indexCount       = static_cast<int>(gpuIndices.size());
	return g;
}