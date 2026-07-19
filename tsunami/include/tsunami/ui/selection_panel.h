#pragma once

#include "tsunami/materials/material.h"
#include "tsunami/scene/scene.h"

#include "vk_mem_alloc.h"

#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace ui {

enum class MaterialEditMode : int {
	Gui   = 0,
	Voice = 1,
};

enum class VoiceDrivenParameter : int {
	BaseTint          = 0,
	EmissionColor     = 1,
	Metalness         = 2,
	Roughness         = 3,
	Transmission      = 4,
	Ior               = 5,
	EmissionIntensity = 6,
	ObjectScale       = 7,
	ObjectTranslateX  = 8,
	ObjectTranslateY  = 9,
	ObjectTranslateZ  = 10,
	ObjectRotateX     = 11,
	ObjectRotateY     = 12,
	ObjectRotateZ     = 13,
	ChameleonMaterial = 14,
};

enum class RenderDebugViewMode : int {
	HiPR      = 0,
	ObjectIds = 1,
	HiPRVis   = 2,
	Naive     = 3,
};

struct ObjectIdEntry {
	int         object_id = -1;
	std::string display_name;
	int         mesh_index     = -1;
	int         material_index = -1;
};

struct HiPRDebugSettings {
	uint32_t rank_count                = 8;
	uint32_t frames_per_object         = 20;
	bool     incremental_sorting       = true;
	float    score_blend               = 0.25f;
	bool     vis_enable_influence_tint = true;
	bool     vis_rainbow_tint          = true;
	float    vis_tint_strength         = 0.50f;
};

struct LightingSettings {
	bool  skybox_enabled            = true;
	bool  directional_light_enabled = true;
	float sun_elevation_deg         = 90.0f;
	float sun_azimuth_deg           = 0.0f;
	float sun_intensity             = 10.0f;
};

struct CameraSettings {
	float fov_deg = 60.0f;
};

struct PathTracingSettings {
	uint32_t spp                     = 1;
	uint32_t max_bounces             = 8;
	uint32_t hipr_water_spp_override = 0;
};

struct SelectionContext {
	int                        selected_mesh_index = -1;
	MaterialEditMode           material_edit_mode  = MaterialEditMode::Gui;
	VoiceDrivenParameter       voice_parameter     = VoiceDrivenParameter::BaseTint;
	RenderDebugViewMode        debug_view_mode     = RenderDebugViewMode::HiPR;
	HiPRDebugSettings          hipr_debug{};
	LightingSettings           lighting{};
	CameraSettings             camera{};
	PathTracingSettings        path_tracing{};
	GPUMaterial                editor_material{};
	float                      editor_scale        = 1.0f;
	glm::vec3                  editor_translation  = glm::vec3(0.0f);
	glm::vec3                  editor_rotation_deg = glm::vec3(0.0f);
	glm::vec4                  outline_color       = glm::vec4(1.0f, 0.65f, 0.15f, 1.0f);
	uint32_t                   outline_width       = 1;
	std::vector<ObjectIdEntry> object_id_map;

	SelectionContext();
};

struct SelectionPanelResult {
	bool material_changed              = false;
	bool material_edit_active          = false;
	bool material_edit_just_finished   = false;
	bool transform_changed             = false;
	bool selection_changed             = false;
	bool hipr_settings_changed         = false;
	bool path_tracing_settings_changed = false;
};

extern SelectionContext selection_ctx;

std::string meshDisplayName(const Scene* scene, int mesh_index);
void        rebuildObjectIdMap(const Scene* scene);
bool        selectMesh(const Scene* scene, int mesh_index);
bool        syncSelectedChameleonMaterial(const Scene* scene, float voice_loudness);
void        applySelectedMaterialEditor(Scene* scene, VmaAllocator allocator, void* material_mapped,
                                        uint32_t material_count, VmaAllocation material_alloc);
SelectionPanelResult drawSelectionPanel(const Scene* scene, float voice_loudness,
                                        bool* is_open = nullptr);

}        // namespace ui
