#pragma once

#include "tsunami/materials/material.h"
#include "tsunami/scene/scene.h"

#include "vk_mem_alloc.h"

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace ui {

enum class MaterialEditMode : int {
	Gui   = 0,
	Voice = 1,
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
	uint32_t rank_count                       = 8;
	uint32_t update_period_frames             = 8;
	bool     incremental_sorting              = true;
	bool     full_resort_on_material_change   = false;
	float    score_blend                      = 0.25f;
	bool     vis_enable_influence_tint        = true;
	bool     vis_rainbow_tint                 = true;
	float    vis_tint_strength                = 0.20f;
};

struct SelectionContext {
	int                        selected_mesh_index = -1;
	MaterialEditMode           material_edit_mode  = MaterialEditMode::Gui;
	RenderDebugViewMode        debug_view_mode     = RenderDebugViewMode::HiPR;
	HiPRDebugSettings          hipr_debug{};
	GPUMaterial                editor_material{};
	glm::vec4                  outline_color = glm::vec4(1.0f, 0.65f, 0.15f, 1.0f);
	uint32_t                   outline_width = 1;
	std::vector<ObjectIdEntry> object_id_map;

	SelectionContext();
};

struct SelectionPanelResult {
	bool material_changed            = false;
	bool material_edit_active        = false;
	bool material_edit_just_finished = false;
	bool selection_changed           = false;
};

extern SelectionContext selection_ctx;

std::string meshDisplayName(const Scene* scene, int mesh_index);
void        rebuildObjectIdMap(const Scene* scene);
bool        selectMesh(const Scene* scene, int mesh_index);
void        applySelectedMaterialEditor(Scene* scene, VmaAllocator allocator, void* material_mapped,
                                        uint32_t material_count, VmaAllocation material_alloc);
SelectionPanelResult drawSelectionPanel(const Scene* scene);

}        // namespace ui
