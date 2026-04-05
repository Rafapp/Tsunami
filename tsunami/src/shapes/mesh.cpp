#include "tsunami/shapes/mesh.h"
#include "tsunami/materials/lambert.h"
#include <assimp/material.h>
#include <iostream>

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

static constexpr uint32_t ASSIMP_OBJ_FLAGS = aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_CalcTangentSpace;

static constexpr uint32_t ASSIMP_GLTF_FLAGS =
    aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
    aiProcess_CalcTangentSpace |
    aiProcess_FlipUVs;        // glTF UVs are top-left origin; flip to match Vulkan

// Recursively collect world transforms for each mesh node.
static void collect_node_transforms(const aiNode* node, const aiMatrix4x4& parent_transform,
                                    std::vector<aiMatrix4x4>& out_transforms, uint32_t mesh_count) {
	aiMatrix4x4 world = parent_transform * node->mTransformation;
	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		uint32_t mesh_idx = node->mMeshes[i];
		if (mesh_idx < mesh_count)
			out_transforms[mesh_idx] = world;
	}
	for (unsigned int i = 0; i < node->mNumChildren; ++i)
		collect_node_transforms(node->mChildren[i], world, out_transforms, mesh_count);
}

// ----------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------

Mesh::Mesh(const std::string& path, Transform transform, std::shared_ptr<Material> material) {
	m_transform = transform;
	m_material  = material;
	if (!load_obj(path))
		std::cerr << "[Mesh] Failed to load: " << path << "\n";
}

Mesh::Mesh(std::vector<GPUVertex> verts, std::vector<uint32_t> indices, Transform transform,
           std::shared_ptr<Material> material) :
    gpuVertices(std::move(verts)),
    gpuIndices(std::move(indices)),
    m_transform(transform),
    m_material(std::move(material)) {
}

// ----------------------------------------------------------------
// .obj loader (existing behaviour, now also fills uv)
// ----------------------------------------------------------------

bool Mesh::load_obj(const std::string& path) {
	Assimp::Importer importer;
	const aiScene*   scene = importer.ReadFile(path, ASSIMP_OBJ_FLAGS);
	if (!scene || !scene->HasMeshes()) {
		std::cerr << "[Mesh] Assimp error: " << importer.GetErrorString() << "\n";
		return false;
	}
	for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
		const aiMesh*  mesh     = scene->mMeshes[m];
		const uint32_t vertBase = static_cast<uint32_t>(gpuVertices.size());
		const bool     has_uv   = mesh->HasTextureCoords(0);

		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			GPUVertex gv{};
			gv.position = {mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};
			gv.normal   = {mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z};
			gv.uv = has_uv ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) :
			                 glm::vec2(0.0f);
			gpuVertices.push_back(gv);
		}
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& f = mesh->mFaces[i];
			gpuIndices.push_back(vertBase + f.mIndices[0]);
			gpuIndices.push_back(vertBase + f.mIndices[1]);
			gpuIndices.push_back(vertBase + f.mIndices[2]);
		}
	}
	return true;
}

// ----------------------------------------------------------------
// glTF factory
// ----------------------------------------------------------------

std::vector<std::unique_ptr<Mesh>> Mesh::load_gltf(const std::string& path) {
	Assimp::Importer importer;
	const aiScene*   scene = importer.ReadFile(path, ASSIMP_GLTF_FLAGS);
	if (!scene || !scene->HasMeshes()) {
		std::cerr << "[Mesh::load_gltf] Assimp error: " << importer.GetErrorString() << "\n";
		return {};
	}

	// Collect per-mesh world transforms by walking the node tree
	std::vector<aiMatrix4x4> world_transforms(scene->mNumMeshes);
	aiMatrix4x4              identity;
	collect_node_transforms(scene->mRootNode, identity, world_transforms, scene->mNumMeshes);

	std::vector<std::unique_ptr<Mesh>> result;
	result.reserve(scene->mNumMeshes);

	for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
		const aiMesh* ai_mesh = scene->mMeshes[m];
		const bool    has_uv  = ai_mesh->HasTextureCoords(0);

		// --- Vertices ---
		std::vector<GPUVertex> verts;
		verts.reserve(ai_mesh->mNumVertices);
		for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
			GPUVertex gv{};
			gv.position = {ai_mesh->mVertices[i].x, ai_mesh->mVertices[i].y,
			               ai_mesh->mVertices[i].z};
			gv.normal   = {ai_mesh->mNormals[i].x, ai_mesh->mNormals[i].y, ai_mesh->mNormals[i].z};
			gv.uv       = has_uv ? glm::vec2(ai_mesh->mTextureCoords[0][i].x,
			                                 ai_mesh->mTextureCoords[0][i].y) :
			                       glm::vec2(0.0f);
			verts.push_back(gv);
		}

		// --- Indices ---
		std::vector<uint32_t> indices;
		indices.reserve(ai_mesh->mNumFaces * 3);
		for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i) {
			const aiFace& f = ai_mesh->mFaces[i];
			indices.push_back(f.mIndices[0]);
			indices.push_back(f.mIndices[1]);
			indices.push_back(f.mIndices[2]);
		}

		// --- Material ---
		// Resolve from the glTF material; stub textures with 0xFFFFFFFF for now.
		glm::vec3 albedo(0.8f);
		glm::vec3 emission(0.0f);
		float     emission_intensity = 0.0f;

		if (scene->HasMaterials() && ai_mesh->mMaterialIndex < scene->mNumMaterials) {
			const aiMaterial* mat = scene->mMaterials[ai_mesh->mMaterialIndex];

			aiColor4D base_color;
			if (mat->Get(AI_MATKEY_BASE_COLOR, base_color) == AI_SUCCESS)
				albedo = {base_color.r, base_color.g, base_color.b};
			else if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, base_color) == AI_SUCCESS)
				albedo = {base_color.r, base_color.g, base_color.b};

			aiColor3D emissive;
			if (mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
				emission           = {emissive.r, emissive.g, emissive.b};
				emission_intensity = glm::length(emission) > 0.0f ? 1.0f : 0.0f;
			}

			// TODO (stb): resolve texture paths and upload
			// aiString tex_path;
			// if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &tex_path) == AI_SUCCESS)
			//     ... stb_load(tex_path.C_Str()) ...
		}

		auto material = std::make_shared<Lambert>(albedo, emission, emission_intensity);

		// --- Transform from node tree ---
		const aiMatrix4x4& t             = world_transforms[m];
		glm::mat4          glm_transform = {
		    t.a1, t.b1, t.c1, t.d1, t.a2, t.b2, t.c2, t.d2,
		    t.a3, t.b3, t.c3, t.d3, t.a4, t.b4, t.c4, t.d4,
		};
		// Wrap in a Transform — pass glm_transform directly if Transform supports it,
		// otherwise use the identity Transform and bake into vertices at load time.
		Transform transform{};

		result.push_back(std::make_unique<Mesh>(std::move(verts), std::move(indices), transform,
		                                        std::move(material)));
	}

	std::cout << "[Mesh::load_gltf] Loaded " << result.size() << " primitives from " << path
	          << "\n";
	return result;
}

// ----------------------------------------------------------------
// Pack
// ----------------------------------------------------------------

GPUMesh Mesh::pack(int matIndex, int vertexOffset, int indexOffset) const {
	GPUMesh g{};
	g.transform        = m_transform.m_transform;
	g.inverseTransform = m_transform.m_inverseTransform;
	g.blasHandle       = 0;
	g.matIndex         = matIndex;
	g.vertexOffset     = vertexOffset;
	g.indexOffset      = indexOffset;
	g.indexCount       = static_cast<int>(gpuIndices.size());
	return g;
}