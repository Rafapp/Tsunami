#include "tsunami/simulation/water_scene_support.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "tsunami/scene/scene.h"

namespace simulation {

namespace {

constexpr std::string_view kWaterMeshLabel              = "water";
constexpr float            kWaterTraceHalfHeightWorld   = 0.45f;
constexpr float            kWaterBoundaryInsetWorld     = 0.02f;
constexpr float            kWaterBoundaryExponentCircle = 2.0f;
constexpr float            kWaterBoundaryExponentSquare = 16.0f;

constexpr glm::vec3 kWaterBaseColor              = glm::vec3(0.02f, 0.12f, 0.16f);
constexpr glm::vec3 kWaterTransmissionColor      = glm::vec3(0.72f, 0.92f, 0.98f);
constexpr float     kWaterSpecularRoughness      = 0.015f;
constexpr float     kWaterTransmissionWeight     = 1.0f;
constexpr float     kWaterTransmissionDepth      = 6.0f;
constexpr float     kWaterTransmissionScatter    = 0.004f;
constexpr float     kWaterTransmissionAnisotropy = 0.0f;
constexpr float     kWaterIor                    = 1.333f;

std::string toLowerCopy(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
		return static_cast<char>(std::tolower(character));
	});
	return value;
}

std::string trimAsciiWhitespace(std::string value) {
	const auto is_space = [](unsigned char character) { return std::isspace(character) != 0; };
	while (!value.empty() && is_space(static_cast<unsigned char>(value.front()))) {
		value.erase(value.begin());
	}
	while (!value.empty() && is_space(static_cast<unsigned char>(value.back()))) {
		value.pop_back();
	}
	return value;
}

glm::vec3 transformPoint(const glm::mat4& transform, const glm::vec3& point) {
	return glm::vec3(transform * glm::vec4(point, 1.0f));
}

int resolveWaterSurfaceMeshIndex(const Scene* scene) {
	if (scene == nullptr) {
		return -1;
	}

	const auto name_matches = [&](const std::string& mesh_name) {
		const std::string lowered = toLowerCopy(mesh_name);
		size_t            start   = 0;
		while (start <= lowered.size()) {
			const size_t slash = lowered.find('/', start);
			std::string  part = (slash == std::string::npos) ? lowered.substr(start) :
			                                                   lowered.substr(start, slash - start);
			part              = trimAsciiWhitespace(part);
			if (part == kWaterMeshLabel) {
				return true;
			}
			if (slash == std::string::npos) {
				break;
			}
			start = slash + 1;
		}
		return false;
	};

	int  resolved_mesh_index = -1;
	bool warned_multi_match  = false;
	for (int mesh_index = 0; mesh_index < static_cast<int>(scene->m_meshes.size()); ++mesh_index) {
		const auto& mesh = scene->m_meshes[mesh_index];
		if (mesh == nullptr || !name_matches(mesh->m_name)) {
			continue;
		}
		if (resolved_mesh_index < 0) {
			resolved_mesh_index = mesh_index;
			continue;
		}
		if (!warned_multi_match) {
			std::cout << "[WARN] Multiple meshes named '" << kWaterMeshLabel
			          << "' found. Using first match at object " << resolved_mesh_index << "\n";
			warned_multi_match = true;
		}
	}

	return resolved_mesh_index;
}

float signedTriangleArea2D(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void rasterizeTriangleMask(std::vector<float>& mask, uint32_t width, uint32_t height,
                           const glm::vec2& p0, const glm::vec2& p1, const glm::vec2& p2) {
	if (width == 0 || height == 0) {
		return;
	}

	const float area = signedTriangleArea2D(p0, p1, p2);
	if (std::abs(area) <= 1.0e-8f) {
		return;
	}

	const float min_xf = std::min({p0.x, p1.x, p2.x});
	const float max_xf = std::max({p0.x, p1.x, p2.x});
	const float min_yf = std::min({p0.y, p1.y, p2.y});
	const float max_yf = std::max({p0.y, p1.y, p2.y});

	const int min_x = std::max(0, static_cast<int>(std::floor(min_xf - 0.5f)));
	const int max_x =
	    std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_xf - 0.5f)));
	const int min_y = std::max(0, static_cast<int>(std::floor(min_yf - 0.5f)));
	const int max_y =
	    std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(max_yf - 0.5f)));
	if (min_x > max_x || min_y > max_y) {
		return;
	}

	const bool positive_area = area > 0.0f;
	for (int y = min_y; y <= max_y; ++y) {
		for (int x = min_x; x <= max_x; ++x) {
			const glm::vec2 sample(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
			const float     e0     = signedTriangleArea2D(p1, p2, sample);
			const float     e1     = signedTriangleArea2D(p2, p0, sample);
			const float     e2     = signedTriangleArea2D(p0, p1, sample);
			const bool      inside = positive_area ? (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) :
			                                         (e0 <= 0.0f && e1 <= 0.0f && e2 <= 0.0f);
			if (!inside) {
				continue;
			}
			mask[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
			    1.0f;
		}
	}
}

glm::vec2 projectWaterVertexToDomainPixel(const glm::vec3&                   world_position,
                                          const WaterSurfaceRenderPlacement& placement,
                                          VkExtent2D                         extent) {
	const glm::vec3 delta = world_position - placement.center;
	const float     u =
	    0.5f *
	    (glm::dot(delta, placement.axis_u) / std::max(placement.half_extent_u, 1.0e-4f) + 1.0f);
	const float v =
	    0.5f *
	    (glm::dot(delta, placement.axis_v) / std::max(placement.half_extent_v, 1.0e-4f) + 1.0f);
	return glm::vec2(u * static_cast<float>(extent.width), v * static_cast<float>(extent.height));
}

}        // namespace

int waterSurfaceObjectId(const Scene* scene, const WaterSurfaceRenderPlacement& placement) {
	return (scene != nullptr && placement.enabled) ? placement.mesh_index : -1;
}

WaterSurfaceRenderPlacement buildWaterSurfacePlacement(const Scene* scene) {
	WaterSurfaceRenderPlacement placement{};

	const int mesh_index = resolveWaterSurfaceMeshIndex(scene);
	if (mesh_index < 0 || scene == nullptr ||
	    mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[mesh_index] == nullptr) {
		std::cout << "[WARN] Unable to resolve water mesh placement. "
		             "Add a mesh named 'water' in the scene.\n";
		return placement;
	}

	const Mesh&     mesh       = *scene->m_meshes[mesh_index];
	const glm::mat4 transform  = mesh.m_transform.m_transform;
	const glm::vec3 local_min  = mesh.m_local_bounds_min;
	const glm::vec3 local_max  = mesh.m_local_bounds_max;
	const glm::vec3 local_size = glm::max(local_max - local_min, glm::vec3(0.0f));

	int normal_axis = 0;
	if (local_size.y < local_size.x) {
		normal_axis = 1;
	}
	if (local_size.z < local_size[normal_axis]) {
		normal_axis = 2;
	}

	std::array<int, 2> surface_axes{};
	for (int axis = 0, surface_axis_count = 0; axis < 3; ++axis) {
		if (axis == normal_axis) {
			continue;
		}
		surface_axes[surface_axis_count++] = axis;
	}

	std::array<glm::vec3, 3> world_axes = {
	    glm::vec3(transform[0]),
	    glm::vec3(transform[1]),
	    glm::vec3(transform[2]),
	};
	std::array<float, 3> world_axis_scales = {
	    glm::length(world_axes[0]),
	    glm::length(world_axes[1]),
	    glm::length(world_axes[2]),
	};

	for (int axis = 0; axis < 3; ++axis) {
		if (world_axis_scales[axis] <= 1.0e-5f) {
			std::cout << "[WARN] Water mesh has a degenerate transform axis at object "
			          << mesh_index << "\n";
			return placement;
		}
		world_axes[axis] /= world_axis_scales[axis];
	}

	glm::vec3  normal                = world_axes[normal_axis];
	const bool normal_axis_points_up = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f)) >= 0.0f;
	if (!normal_axis_points_up) {
		normal = -normal;
	}

	glm::vec3 axis_u = world_axes[surface_axes[0]];
	glm::vec3 axis_v = world_axes[surface_axes[1]];
	if (glm::dot(glm::cross(axis_u, axis_v), normal) < 0.0f) {
		axis_v = -axis_v;
	}

	const float half_extent_u =
	    0.5f * local_size[surface_axes[0]] * world_axis_scales[surface_axes[0]];
	const float half_extent_v =
	    0.5f * local_size[surface_axes[1]] * world_axis_scales[surface_axes[1]];
	if (half_extent_u <= 1.0e-4f || half_extent_v <= 1.0e-4f) {
		std::cout << "[WARN] Water mesh produced invalid half-extents for object " << mesh_index
		          << "\n";
		return placement;
	}

	glm::vec3 surface_local = 0.5f * (local_min + local_max);
	surface_local[normal_axis] =
	    normal_axis_points_up ? local_max[normal_axis] : local_min[normal_axis];
	glm::vec3 center_world = transformPoint(transform, surface_local);

	int   corner_vertex_count = 0;
	float max_l1_span         = 0.0f;
	for (const GPUVertex& vertex : mesh.gpuVertices) {
		const glm::vec3 world_position = transformPoint(transform, vertex.position);
		const glm::vec3 delta          = world_position - center_world;
		const float u_norm = std::abs(glm::dot(delta, axis_u) / std::max(half_extent_u, 1.0e-4f));
		const float v_norm = std::abs(glm::dot(delta, axis_v) / std::max(half_extent_v, 1.0e-4f));
		max_l1_span        = std::max(max_l1_span, u_norm + v_norm);
		if (u_norm > 0.85f && v_norm > 0.85f) {
			++corner_vertex_count;
		}
	}

	const bool  likely_square_boundary = corner_vertex_count >= 4 || max_l1_span > 1.70f;
	const float min_half_extent        = std::max(std::min(half_extent_u, half_extent_v), 1.0e-4f);
	const float boundary_inset_world = std::min(kWaterBoundaryInsetWorld, min_half_extent * 0.25f);
	const float floating_surface_bounds =
	    std::clamp(1.0f - boundary_inset_world / min_half_extent, 0.70f, 0.99f);
	const float floating_boundary_exponent =
	    likely_square_boundary ? kWaterBoundaryExponentSquare : kWaterBoundaryExponentCircle;

	placement.enabled                    = true;
	placement.mesh_index                 = mesh_index;
	placement.center                     = center_world;
	placement.trace_half_height          = kWaterTraceHalfHeightWorld;
	placement.axis_u                     = axis_u;
	placement.half_extent_u              = half_extent_u;
	placement.axis_v                     = axis_v;
	placement.half_extent_v              = half_extent_v;
	placement.normal                     = normal;
	placement.floating_surface_bounds    = floating_surface_bounds;
	placement.floating_boundary_exponent = floating_boundary_exponent;
	return placement;
}

std::vector<float> buildWaterSurfaceDomainMask(const Scene*                       scene,
                                               const WaterSurfaceRenderPlacement& placement,
                                               VkExtent2D                         extent) {
	const size_t texel_count = static_cast<size_t>(extent.width) * extent.height;
	if (texel_count == 0) {
		return {};
	}

	std::vector<float> mask(texel_count, 1.0f);
	if (scene == nullptr || !placement.enabled || placement.mesh_index < 0 ||
	    placement.mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[placement.mesh_index] == nullptr) {
		return mask;
	}

	mask.assign(texel_count, 0.0f);
	const Mesh& mesh = *scene->m_meshes[placement.mesh_index];

	if (mesh.gpuVertices.empty()) {
		return std::vector<float>(texel_count, 1.0f);
	}

	const auto project_vertex = [&](uint32_t vertex_index) {
		const glm::vec3 world_position =
		    transformPoint(mesh.m_transform.m_transform, mesh.gpuVertices[vertex_index].position);
		return projectWaterVertexToDomainPixel(world_position, placement, extent);
	};

	bool had_triangle = false;
	if (!mesh.gpuIndices.empty()) {
		for (size_t index = 0; index + 2 < mesh.gpuIndices.size(); index += 3) {
			const uint32_t i0 = mesh.gpuIndices[index + 0];
			const uint32_t i1 = mesh.gpuIndices[index + 1];
			const uint32_t i2 = mesh.gpuIndices[index + 2];
			if (i0 >= mesh.gpuVertices.size() || i1 >= mesh.gpuVertices.size() ||
			    i2 >= mesh.gpuVertices.size()) {
				continue;
			}
			rasterizeTriangleMask(mask, extent.width, extent.height, project_vertex(i0),
			                      project_vertex(i1), project_vertex(i2));
			had_triangle = true;
		}
	} else {
		for (size_t index = 0; index + 2 < mesh.gpuVertices.size(); index += 3) {
			rasterizeTriangleMask(mask, extent.width, extent.height,
			                      project_vertex(static_cast<uint32_t>(index + 0)),
			                      project_vertex(static_cast<uint32_t>(index + 1)),
			                      project_vertex(static_cast<uint32_t>(index + 2)));
			had_triangle = true;
		}
	}

	if (!had_triangle) {
		return std::vector<float>(texel_count, 1.0f);
	}

	size_t active_texels = 0;
	for (float value : mask) {
		if (value > 0.5f) {
			++active_texels;
		}
	}
	if (active_texels == 0) {
		return std::vector<float>(texel_count, 1.0f);
	}

	// Dilate one texel to avoid tiny raster gaps around triangle edges.
	std::vector<float> dilated = mask;
	for (uint32_t y = 0; y < extent.height; ++y) {
		for (uint32_t x = 0; x < extent.width; ++x) {
			const size_t index = static_cast<size_t>(y) * extent.width + x;
			if (mask[index] > 0.5f) {
				continue;
			}
			bool neighbor_active = false;
			for (int oy = -1; oy <= 1 && !neighbor_active; ++oy) {
				for (int ox = -1; ox <= 1; ++ox) {
					const int nx = static_cast<int>(x) + ox;
					const int ny = static_cast<int>(y) + oy;
					if (nx < 0 || ny < 0 || nx >= static_cast<int>(extent.width) ||
					    ny >= static_cast<int>(extent.height)) {
						continue;
					}
					const size_t neighbor_index =
					    static_cast<size_t>(ny) * extent.width + static_cast<size_t>(nx);
					if (mask[neighbor_index] > 0.5f) {
						neighbor_active = true;
						break;
					}
				}
			}
			if (neighbor_active) {
				dilated[index] = 1.0f;
			}
		}
	}

	return dilated;
}

void applyDedicatedWaterMaterial(Scene* scene, const WaterSurfaceRenderPlacement& placement) {
	if (scene == nullptr || !placement.enabled) {
		return;
	}

	const int mesh_index = placement.mesh_index;
	if (mesh_index < 0 || mesh_index >= static_cast<int>(scene->m_meshes.size()) ||
	    scene->m_meshes[mesh_index] == nullptr ||
	    scene->m_meshes[mesh_index]->m_material == nullptr) {
		return;
	}

	GPUMaterial& water_material           = scene->m_meshes[mesh_index]->m_material->m_gpu;
	water_material.base_color             = glm::vec4(kWaterBaseColor, 1.0f);
	water_material.base_metalness         = 0.0f;
	water_material.base_diffuse_roughness = 0.0f;
	water_material.specular_color         = glm::vec4(1.0f);
	water_material.specular_roughness     = kWaterSpecularRoughness;
	water_material.specular_ior           = kWaterIor;
	water_material.specular_anisotropy    = 0.0f;
	water_material.transmission_weight    = kWaterTransmissionWeight;
	water_material.transmission_depth     = kWaterTransmissionDepth;
	water_material.transmission_color     = glm::vec4(kWaterTransmissionColor, 1.0f);
	water_material.transmission_scatter   = glm::vec4(
	    kWaterTransmissionScatter, kWaterTransmissionScatter, kWaterTransmissionScatter, 0.0f);
	water_material.transmission_scatter_anisotropy = kWaterTransmissionAnisotropy;
	water_material.geometry_opacity                = 1.0f;
	water_material.emission_color                  = glm::vec4(0.0f);
	water_material.emission_luminance              = 0.0f;
	water_material.coat_weight                     = 0.0f;
	water_material.fuzz_weight                     = 0.0f;
	water_material.thin_film_weight                = 0.0f;
	water_material.albedo_tex_index                = std::numeric_limits<uint32_t>::max();
	water_material.normal_tex_index                = std::numeric_limits<uint32_t>::max();
	water_material.roughness_tex_index             = std::numeric_limits<uint32_t>::max();
	water_material.emissive_tex_index              = std::numeric_limits<uint32_t>::max();
}

}        // namespace simulation
