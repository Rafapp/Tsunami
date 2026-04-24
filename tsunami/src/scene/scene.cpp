// Purpose: Implements scene loading plus mesh/material/light ownership and bookkeeping.
#include "tsunami/scene/scene.h"
#include "tsunami/shapes/mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

GPUScene Scene::pack() const {
	GPUScene packed;
	packed.camera = m_camera.pack();

	for (const auto& light : m_lights)
		packed.lights.push_back(light->pack());

	// Analytic shapes only.
	// Triangle meshes are flattened / packed later in app.cpp once
	// vertexOffset and indexOffset are known.
	for (int i = 0; i < (int) m_shapes.size(); ++i) {
		Shape* shape = m_shapes[i].get();
		packed.materials.push_back(shape->m_material->pack());
		packed.shapes.push_back(shape->pack(i));
	}

	return packed;
}

int Scene::load_texture_if_needed(const std::shared_ptr<Texture>& texture) {
	if (!texture || !texture->valid())
		return -1;
	auto it = m_texture_lookup.find(texture->source_path);
	if (it != m_texture_lookup.end())
		return it->second;
	const int idx                          = static_cast<int>(m_textures.size());
	m_texture_lookup[texture->source_path] = idx;
	m_textures.push_back(texture);
	return idx;
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Fetch the texture index for a given material slot, loading the texture if
// it has not been seen before.  Returns 0xFFFFFFFF when no texture exists.
static uint32_t get_tex_index(const aiScene* ai_scene, const std::string& scene_source_path,
                              aiMaterial* ai_mat, aiTextureType type, bool srgb, Scene& dst) {
	if (!ai_mat || ai_mat->GetTextureCount(type) == 0)
		return 0xFFFFFFFFu;
	aiString tex_path;
	if (ai_mat->GetTexture(type, 0, &tex_path) != aiReturn_SUCCESS)
		return 0xFFFFFFFFu;
	auto tex = Texture::load_from_assimp(ai_scene, scene_source_path, tex_path, srgb);
	if (!tex || !tex->valid())
		return 0xFFFFFFFFu;
	int idx = dst.load_texture_if_needed(tex);
	return idx >= 0 ? static_cast<uint32_t>(idx) : 0xFFFFFFFFu;
}

// Load every material texture from the Assimp scene so they are all present in
// m_textures before we wire up the per-material indices.
static void load_all_material_textures(const aiScene*     ai_scene,
                                       const std::string& scene_source_path, Scene& dst) {
	if (!ai_scene)
		return;

	auto load_slot = [&](aiMaterial* mat, aiTextureType type, bool srgb) {
		const unsigned int count = mat->GetTextureCount(type);
		for (unsigned int i = 0; i < count; ++i) {
			aiString tex_path;
			if (mat->GetTexture(type, i, &tex_path) != aiReturn_SUCCESS)
				continue;
			auto tex = Texture::load_from_assimp(ai_scene, scene_source_path, tex_path, srgb);
			if (tex && tex->valid()) {
				dst.load_texture_if_needed(tex);
				std::cout << "[Scene] Loaded texture: " << tex->source_path << " (" << tex->width
				          << "x" << tex->height << ")\n";
			}
		}
	};

	for (unsigned int mi = 0; mi < ai_scene->mNumMaterials; ++mi) {
		aiMaterial* mat = ai_scene->mMaterials[mi];
		if (!mat)
			continue;
		// sRGB colour textures
		load_slot(mat, aiTextureType_BASE_COLOR, true);
		load_slot(mat, aiTextureType_DIFFUSE, true);
		load_slot(mat, aiTextureType_EMISSIVE, true);
		// Linear data textures
		load_slot(mat, aiTextureType_NORMALS, false);
		load_slot(mat, aiTextureType_HEIGHT, false);
		load_slot(mat, aiTextureType_METALNESS, false);
		load_slot(mat, aiTextureType_DIFFUSE_ROUGHNESS, false);
		load_slot(mat, aiTextureType_AMBIENT_OCCLUSION, false);
		load_slot(mat, aiTextureType_SPECULAR, false);
		load_slot(mat, aiTextureType_OPACITY, false);
		load_slot(mat, aiTextureType_UNKNOWN, false);
	}
}

// Wire per-mesh texture indices.  We iterate ai_scene->mMeshes in the same
// traversal order that Mesh::load_gltf uses so that mesh_idx here matches the
// position in m_meshes.
static void wire_texture_indices_node_recursive(const aiScene* ai_scene, const aiNode* node,
                                                const std::string& scene_source_path,
                                                std::vector<std::unique_ptr<Mesh>>& meshes,
                                                Scene& dst, int& mesh_idx) {
	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		const unsigned int src_mesh_idx = node->mMeshes[i];
		if (src_mesh_idx >= ai_scene->mNumMeshes)
			continue;
		if (mesh_idx >= (int) meshes.size())
			return;

		const aiMesh* ai_mesh = ai_scene->mMeshes[src_mesh_idx];
		if (!ai_mesh || !(ai_mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
			continue;

		auto& mat = meshes[mesh_idx]->m_material;
		if (mat) {
			aiMaterial* ai_mat = ai_scene->mMaterials[ai_mesh->mMaterialIndex];

			uint32_t alb = get_tex_index(ai_scene, scene_source_path, ai_mat,
			                             aiTextureType_BASE_COLOR, true, dst);
			if (alb == 0xFFFFFFFFu)
				alb = get_tex_index(ai_scene, scene_source_path, ai_mat, aiTextureType_DIFFUSE,
				                    true, dst);
			mat->m_gpu.albedo_tex_index = alb;

			uint32_t nrm = get_tex_index(ai_scene, scene_source_path, ai_mat, aiTextureType_NORMALS,
			                             false, dst);
			if (nrm == 0xFFFFFFFFu)
				nrm = get_tex_index(ai_scene, scene_source_path, ai_mat, aiTextureType_HEIGHT,
				                    false, dst);
			mat->m_gpu.normal_tex_index = nrm;

			uint32_t rgh = get_tex_index(ai_scene, scene_source_path, ai_mat,
			                             aiTextureType_DIFFUSE_ROUGHNESS, false, dst);
			if (rgh == 0xFFFFFFFFu)
				rgh = get_tex_index(ai_scene, scene_source_path, ai_mat, aiTextureType_UNKNOWN,
				                    false, dst);
			mat->m_gpu.roughness_tex_index = rgh;

			mat->m_gpu.emissive_tex_index = get_tex_index(ai_scene, scene_source_path, ai_mat,
			                                              aiTextureType_EMISSIVE, true, dst);
		}

		++mesh_idx;
	}

	for (unsigned int c = 0; c < node->mNumChildren; ++c) {
		wire_texture_indices_node_recursive(ai_scene, node->mChildren[c], scene_source_path, meshes,
		                                    dst, mesh_idx);
	}
}

static void wire_texture_indices(const aiScene* ai_scene, const std::string& scene_source_path,
                                 std::vector<std::unique_ptr<Mesh>>& meshes, Scene& dst) {
	int mesh_idx = 0;
	wire_texture_indices_node_recursive(ai_scene, ai_scene->mRootNode, scene_source_path, meshes,
	                                    dst, mesh_idx);
}

// ─── Scene::load_gltf ─────────────────────────────────────────────────────────

void Scene::load_gltf(const std::string& path) {
	m_meshes.clear();
	m_textures.clear();
	m_texture_lookup.clear();
	append_gltf(path);
}

void Scene::append_gltf(const std::string& path) {
	// Pass 1: load mesh geometry (handled by Mesh::load_gltf which does its
	//         own Assimp import with FlipUVs etc.).
	auto meshes = Mesh::load_gltf(path);
	if (meshes.empty()) {
		std::cerr << "[Scene::append_gltf] No meshes loaded from: " << path << "\n";
		return;
	}
	std::cout << "[Scene] Appending " << meshes.size() << " meshes from " << path << "\n";

	// Pass 2: Assimp import for texture discovery + index wiring.
	//         We use the same post-process flags as the texture scan so the
	//         primitive ordering matches Pass 1.
	static constexpr unsigned int TEXTURE_FLAGS =
	    aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
	    aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality | aiProcess_SortByPType;

	Assimp::Importer importer;
	const aiScene*   ai_scene = importer.ReadFile(path, TEXTURE_FLAGS);
	if (!ai_scene || !ai_scene->mRootNode) {
		std::cerr << "[Scene::append_gltf] Assimp texture pass failed: "
		          << importer.GetErrorString() << "\n";
		return;
	}

	// Load all referenced textures into m_textures (deduplication via m_texture_lookup).
	load_all_material_textures(ai_scene, path, *this);

	// Wire per-mesh texture indices now that every texture has a stable index.
	wire_texture_indices(ai_scene, path, meshes, *this);

	for (auto& mesh : meshes)
		m_meshes.push_back(std::move(mesh));

	std::cout << "[Scene] Meshes: " << m_meshes.size() << "  |  Textures: " << m_textures.size()
	          << "\n";
}
