// Purpose: Implements mesh container behavior and geometry helper routines.
#include "tsunami/shapes/mesh.h"
#include <assimp/material.h>
#include <cmath>
#include <glm/geometric.hpp>
#include <iostream>
#include <utility>

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

static constexpr uint32_t ASSIMP_OBJ_FLAGS = aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_CalcTangentSpace;

static constexpr uint32_t ASSIMP_GLTF_FLAGS = aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                                              aiProcess_JoinIdenticalVertices |
                                              aiProcess_CalcTangentSpace | aiProcess_FlipUVs;

// Assimp -> GLM conversion.
// This mapping is the safe/standard one for copying aiMatrix4x4 into glm::mat4.
static glm::mat4 ai_to_glm(const aiMatrix4x4& m) {
	glm::mat4 out;
	out[0][0] = m.a1;
	out[1][0] = m.a2;
	out[2][0] = m.a3;
	out[3][0] = m.a4;
	out[0][1] = m.b1;
	out[1][1] = m.b2;
	out[2][1] = m.b3;
	out[3][1] = m.b4;
	out[0][2] = m.c1;
	out[1][2] = m.c2;
	out[2][2] = m.c3;
	out[3][2] = m.c4;
	out[0][3] = m.d1;
	out[1][3] = m.d2;
	out[2][3] = m.d3;
	out[3][3] = m.d4;
	return out;
}

static glm::vec3 safe_normalize(const glm::vec3& v, const glm::vec3& fallback) {
	const float len2 = glm::dot(v, v);
	if (len2 > 1.0e-12f) {
		return v * glm::inversesqrt(len2);
	}
	return fallback;
}

static glm::vec3 build_orthonormal_tangent(const glm::vec3& n) {
	const glm::vec3 axis =
	    (std::abs(n.y) < 0.999f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
	return safe_normalize(glm::cross(axis, n), glm::vec3(1.0f, 0.0f, 0.0f));
}

static std::string build_gltf_mesh_name(const aiNode* node, const aiMesh* ai_mesh,
                                        unsigned int mesh_idx) {
	const std::string node_name = (node != nullptr) ? node->mName.C_Str() : "";
	const std::string mesh_name = (ai_mesh != nullptr) ? ai_mesh->mName.C_Str() : "";

	if (!node_name.empty() && !mesh_name.empty() && node_name != mesh_name) {
		return node_name + " / " + mesh_name;
	}
	if (!mesh_name.empty()) {
		return mesh_name;
	}
	if (!node_name.empty()) {
		return node_name;
	}
	return "Mesh " + std::to_string(mesh_idx);
}

static void append_gltf_node_meshes(const aiScene* scene, const aiNode* node,
                                    const aiMatrix4x4&                  parent_transform,
                                    std::vector<std::unique_ptr<Mesh>>& result) {
	aiMatrix4x4     world         = parent_transform * node->mTransformation;
	const glm::mat4 glm_transform = ai_to_glm(world);
	const glm::mat3 linear_transform(glm_transform);
	const glm::mat3 normal_matrix = glm::transpose(glm::inverse(linear_transform));

	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		const unsigned int mesh_idx = node->mMeshes[i];
		if (mesh_idx >= scene->mNumMeshes)
			continue;

		const aiMesh* ai_mesh = scene->mMeshes[mesh_idx];
		if (!ai_mesh)
			continue;

		const bool has_uv       = ai_mesh->HasTextureCoords(0);
		const bool has_normals  = ai_mesh->HasNormals();
		const bool has_tangents = ai_mesh->HasTangentsAndBitangents();

		// --- Vertices ---
		std::vector<GPUVertex> verts;
		verts.reserve(ai_mesh->mNumVertices);
		for (unsigned int v = 0; v < ai_mesh->mNumVertices; ++v) {
			GPUVertex       gv{};
			const glm::vec3 local_pos(ai_mesh->mVertices[v].x, ai_mesh->mVertices[v].y,
			                          ai_mesh->mVertices[v].z);
			const glm::vec4 world_pos = glm_transform * glm::vec4(local_pos, 1.0f);
			gv.position               = glm::vec3(world_pos);

			if (has_normals) {
				gv.normal = safe_normalize(normal_matrix * glm::vec3(ai_mesh->mNormals[v].x,
				                                                     ai_mesh->mNormals[v].y,
				                                                     ai_mesh->mNormals[v].z),
				                           glm::vec3(0.0f, 1.0f, 0.0f));
			} else {
				gv.normal = glm::vec3(0.0f, 1.0f, 0.0f);
			}

			gv.uv = has_uv ? glm::vec2(ai_mesh->mTextureCoords[0][v].x,
			                           ai_mesh->mTextureCoords[0][v].y) :
			                 glm::vec2(0.0f);

			// --- Tangent ---
			if (has_tangents) {
				const auto& t = ai_mesh->mTangents[v];
				const auto& b = ai_mesh->mBitangents[v];

				glm::vec3 tangent   = safe_normalize(linear_transform * glm::vec3(t.x, t.y, t.z),
				                                     build_orthonormal_tangent(gv.normal));
				glm::vec3 bitangent = safe_normalize(linear_transform * glm::vec3(b.x, b.y, b.z),
				                                     glm::cross(gv.normal, tangent));
				glm::vec3 normal    = gv.normal;
				tangent             = safe_normalize(tangent - normal * glm::dot(normal, tangent),
				                                     build_orthonormal_tangent(normal));

				float handedness =
				    (glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;

				gv.tangent = glm::vec4(tangent, handedness);
			} else {
				gv.tangent = glm::vec4(build_orthonormal_tangent(gv.normal), 1.0f);
			}
			verts.push_back(gv);
		}

		// --- Indices ---
		std::vector<uint32_t> indices;
		indices.reserve(ai_mesh->mNumFaces * 3);
		for (unsigned int f = 0; f < ai_mesh->mNumFaces; ++f) {
			const aiFace& face = ai_mesh->mFaces[f];
			if (face.mNumIndices != 3)
				continue;
			indices.push_back(face.mIndices[0]);
			indices.push_back(face.mIndices[1]);
			indices.push_back(face.mIndices[2]);
		}

		// --- Material ---
		glm::vec3 albedo(0.8f);
		glm::vec3 emission(0.0f);
		float     emission_intensity = 0.0f;
		float     metalness          = 0.f;
		float     roughness          = 0.3f;
		float     ior                = 1.5f;
		float     transmission       = 0.f;

		const aiMaterial* mat = nullptr;

		if (scene->HasMaterials() && ai_mesh->mMaterialIndex < scene->mNumMaterials) {
			mat = scene->mMaterials[ai_mesh->mMaterialIndex];

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

			mat->Get(AI_MATKEY_METALLIC_FACTOR, metalness);
			mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
			mat->Get(AI_MATKEY_REFRACTI, ior);
			mat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission);
		}

		auto material = std::make_shared<Material>();
		material->albedo(albedo)
		    .emission(emission, emission_intensity)
		    .metalness(metalness)
		    .roughness(roughness)
		    .ior(ior)
		    .transmission(transmission);

		// Transmission stuff
		material->m_gpu.transmission_depth = 0.f;
		material->m_gpu.transmission_color =
		    glm::vec4(albedo, 1.f);        // default to albedo if no transmission color provided

		Transform transform{};
		// World transform is baked into vertices for glTF meshes.
		// Using identity here avoids TLAS/shader transform convention mismatches.
		transform.m_transform        = glm::mat4(1.0f);
		transform.m_inverseTransform = glm::mat4(1.0f);

		result.push_back(std::make_unique<Mesh>(std::move(verts), std::move(indices), transform,
		                                        std::move(material),
		                                        build_gltf_mesh_name(node, ai_mesh, mesh_idx)));

		std::cout << "[Mesh::load_gltf] node=" << node->mName.C_Str() << " mesh_idx=" << mesh_idx
		          << " (transform baked)\n";
	}

	for (unsigned int c = 0; c < node->mNumChildren; ++c) {
		append_gltf_node_meshes(scene, node->mChildren[c], world, result);
	}
}

// ----------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------

Mesh::Mesh(const std::string& path, Transform transform, std::shared_ptr<Material> material) {
	m_transform = transform;
	m_material  = material;
	m_name      = path;
	if (!load_obj(path))
		std::cerr << "[Mesh] Failed to load: " << path << "\n";
}

Mesh::Mesh(std::vector<GPUVertex> verts, std::vector<uint32_t> indices, Transform transform,
           std::shared_ptr<Material> material, std::string name) :
    gpuVertices(std::move(verts)),
    gpuIndices(std::move(indices)),
    m_transform(transform),
    m_material(std::move(material)),
    m_name(std::move(name)) {
	compute_local_bounds();
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
		const aiMesh*  mesh         = scene->mMeshes[m];
		const uint32_t vertBase     = static_cast<uint32_t>(gpuVertices.size());
		const bool     has_uv       = mesh->HasTextureCoords(0);
		const bool     has_normals  = mesh->HasNormals();
		const bool     has_tangents = mesh->HasTangentsAndBitangents();

		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			GPUVertex gv{};
			gv.position = {mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};
			if (has_normals) {
				gv.normal = safe_normalize(
				    glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z),
				    glm::vec3(0.0f, 1.0f, 0.0f));
			} else {
				gv.normal = glm::vec3(0.0f, 1.0f, 0.0f);
			}
			gv.uv = has_uv ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) :
			                 glm::vec2(0.0f);
			if (has_tangents) {
				const auto& t = mesh->mTangents[i];
				const auto& b = mesh->mBitangents[i];
				glm::vec3   tangent =
				    safe_normalize(glm::vec3(t.x, t.y, t.z), build_orthonormal_tangent(gv.normal));
				glm::vec3 bitangent =
				    safe_normalize(glm::vec3(b.x, b.y, b.z), glm::cross(gv.normal, tangent));
				tangent = safe_normalize(tangent - gv.normal * glm::dot(gv.normal, tangent),
				                         build_orthonormal_tangent(gv.normal));
				const float handedness =
				    (glm::dot(glm::cross(gv.normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
				gv.tangent = glm::vec4(tangent, handedness);
			} else {
				gv.tangent = glm::vec4(build_orthonormal_tangent(gv.normal), 1.0f);
			}
			gpuVertices.push_back(gv);
		}
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& f = mesh->mFaces[i];
			gpuIndices.push_back(vertBase + f.mIndices[0]);
			gpuIndices.push_back(vertBase + f.mIndices[1]);
			gpuIndices.push_back(vertBase + f.mIndices[2]);
		}
	}
	compute_local_bounds();
	return true;
}

// ----------------------------------------------------------------
// glTF factory
// ----------------------------------------------------------------

std::vector<std::unique_ptr<Mesh>> Mesh::load_gltf(const std::string& path) {
	Assimp::Importer importer;
	const aiScene*   scene = importer.ReadFile(path, ASSIMP_GLTF_FLAGS);
	if (!scene || !scene->HasMeshes() || !scene->mRootNode) {
		std::cerr << "[Mesh::load_gltf] Assimp error: " << importer.GetErrorString() << "\n";
		return {};
	}

	std::vector<std::unique_ptr<Mesh>> result;
	aiMatrix4x4                        identity;
	append_gltf_node_meshes(scene, scene->mRootNode, identity, result);

	std::cout << "[Mesh::load_gltf] Loaded " << result.size() << " mesh instances from " << path
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
	g.blasHandle_lo    = 0;
	g.blasHandle_hi    = 0;
	g.matIndex         = matIndex;
	g.vertexOffset     = vertexOffset;
	g.indexOffset      = indexOffset;
	g.indexCount       = static_cast<int>(gpuIndices.size());
	return g;
}

void Mesh::compute_local_bounds() {
	if (gpuVertices.empty()) {
		m_local_bounds_min = glm::vec3(0.0f);
		m_local_bounds_max = glm::vec3(0.0f);
		return;
	}

	m_local_bounds_min = gpuVertices.front().position;
	m_local_bounds_max = gpuVertices.front().position;

	for (const GPUVertex& vertex : gpuVertices) {
		m_local_bounds_min = glm::min(m_local_bounds_min, vertex.position);
		m_local_bounds_max = glm::max(m_local_bounds_max, vertex.position);
	}
}
